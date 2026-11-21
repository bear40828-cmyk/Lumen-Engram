#!/usr/bin/env python3
"""Lumen Engram · 召回改造（2026-11-04）：四层，规则路由不再有否决权

设计：
  1. 专名直通    entities.py：从库里自动维护专名表，当前句命中直通专名 → 直接召回，不经任何「要不要查」的判断；
                专名命中的候选排在情绪词命中的前面。
  2. 每轮都查    FTS + 向量每轮都跑；候选池最多 20 条交给精排（rerank.judge）决定注不注、注哪条。
                寒暄由精排判 need=false，不再由正则/路由拦。
  3. 查询改写    没有明确专名时，便宜模型把这句改写成 2~3 个检索问句（query_rewrite.py），各查一遍，并进同一个池。
  4. 上下文      检索（短句补上文）与精排都带最近 3~5 轮对话。

借鉴：
  · 「候选池最多 20 → 精排闸 → 最多 2 条注入」：参考 Serein 的召回流程；本模块的 POOL_CAP / MAX_HITS 沿用。
  · mem0（Apache-2.0）：实体路仍进池，但按 Serein 不再给实体加分绕过精排。
  · Hindsight（MIT）engine/search/fusion.py：多路并行检索 → 每路先封顶 → RRF 融合 → 再精排。RRF 沿用 hybrid.py。
  · memobase（Apache-2.0）controllers/context.py pack_latest_chat：用最近几条对话当检索上下文。

聊天注入：最多 MAX_HITS=2；门槛 NORMAL_MIN/EXPLICIT_MIN/ENTITY_MIN（Serein：只看正文分；明确回忆略松）。
评测（benchmark）走 topk()：直接返回召回接口的前 10 条，不经 2 条 / ≥7 分闸。

整个流程由开关 ENGRAM_RECALL_V2 控制（默认关＝线上聊天一字不差）；查询改写另有开关 ENGRAM_RECALL_REWRITE
（默认关；每轮多一次便宜模型调用，开之前先看成本估算）。
"""
from __future__ import annotations

import re
import sys
import unicodedata
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import entities as ENT                      # noqa: E402
import flags                                # noqa: E402

POOL_CAP = 20            # 候选池上限（Serein：最多 20 → 精排）
POOL_CAP_BENCH = 30
FULL_POOL_CAP = 40
FULL_CHANNEL_CAP = 80
# Offline replay calibration only; NOT approved for enabling full-on.
FULL_POOL_MIN = 0.60  # includes spec region positive at 0.617; 0.66 rejected it
MAX_HITS = 2             # 聊天注入上限（不变）
# Serein 对齐 + 再拧（2026-11-04）：专名/cue 只进池；失败不回退关键词闸。
NORMAL_MIN = 8           # 日常无专名：need=true 且 ≥8
EXPLICIT_MIN = 7         # 「之前/昨天/还记得…」明确回忆：≥7
ENTITY_MIN = 7           # 有专名直通：专名候选 ≥7
OTHER_WITH_ENTITY_MIN = 9  # 有专名时非专名候选 / 无专名时的钉住条：≥9
ECHO_W = 1 / 3
VEC_FLOOR = 0.45
VEC_K = 8
ENTITY_PER_TERM = 12
FTS_PER_TERM = 6
ENTITY_RRF_WEIGHT = 1.0  # 专名路与其他路等权：只进池，不加分（旧 1.5）
VEC_SIM_BONUS = 0.1      # 语义分高于 0.5 的部分 × 0.1 加到 RRF（仍只影响池内排序，不绕过精排闸）
CONTEXT_TURNS = 4
SHORT_MSG = 12
# 泛词软专名：进池当 FTS cue 也容易把池子污染成「工具/摘要」故事（回放 #57/#91）。直通不受影响。
SOFT_BLOCK = frozenset({
    "工具", "摘要", "记账", "一下", "东西", "步骤", "调用", "思考", "模型",
    "内容", "记录", "消息", "窗口", "回复", "对话", "系统", "功能", "问题",
})
# 仅这些词做「前缀+词」合成 FTS cue（小模型/思考摘要/调用工具）；其它 SOFT_BLOCK 只屏蔽不合成
COMPOUND_STEMS = frozenset({"模型", "摘要", "工具", "记账"})


@dataclass
class Flow:
    text: str
    entities: list = field(default_factory=list)
    rewrites: list = field(default_factory=list)
    pool: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    need: bool | None = None
    how: str = ""
    reason: str = ""
    trace: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    judge_scores: list = field(default_factory=list)
    model_need: bool | None = None
    judged: bool = False            # 精排模型是否真的给了结果
    calls: dict = field(default_factory=dict)   # 本轮发生的模型调用次数 {"rewrite":0/1,"rerank":0/1}


def rewrite_on() -> bool:
    return flags.on("RECALL_REWRITE")


def _label(h: dict) -> str:
    out = ""
    src = h.get("echo_from")
    if src:
        try:
            import echo
            out = echo.label(src)
        except Exception:                                    # noqa: BLE001
            out = "〔转述〕"
    if h.get("self_note"):
        try:
            import persona
            out += f"〔{persona.N_CLAUDE}自记〕"
        except Exception:                                    # noqa: BLE001
            out += "〔自记〕"
    if h.get("assistant_say"):
        out += "〔我说的〕"
    return out


# ── 清洗：App 包装头不是她说的话 ───────────────────────────────────────────────
# 2026-11-03 实测：hook 把「【<应用名>当前联系人资料｜…】…」整段当检索句，真正的话（「我饿了」）埋在最后；
# 「【自主唤醒入口｜服务器任务，不是用户消息】」是定时任务，不是用户的话。
RE_SERVER_TASK = re.compile(
    r"^\s*【自主唤醒入口|^\s*【后台"
    r"|^\s*【[^】\n]{0,40}(?:服务器任务|不是[^】\n]{0,8}消息)[^】\n]{0,20}】"
    # 2026-11-12：「【定时·登录过期提醒】」「【某某测试】」这类给助手的任务头（行首方括号＋任务名词）
    r"|^\s*【(?![^】\n]*联系人资料)[^】\n]{0,12}(?:定时|测试|自检|体检|抽查)[^】\n]{0,24}】"
    r"|^\s*\[\[[a-z][a-z0-9_-]{1,24}\|[^\]\n]*\]\]\s*$")
# 聊天 App 转发时每行前面带「[hh:mm:ss] 」时间戳；认结构之前先剥掉，否则 [[bear-…]] 标记认不出来
RE_TS_PREFIX = re.compile(r"^[ \t]*\[\d{1,2}:\d{2}(?::\d{2})?\][ \t]*", re.M)


def strip_ts(text: str) -> str:
    return RE_TS_PREFIX.sub("", text or "")


RE_APP_HEADER = re.compile(r"^\s*【[^】｜\n]{0,12}当前联系人资料[^】]*】.*?(?:\n\s*\n|$)", re.S)


RE_PROBE_REPLY = re.compile(
    r"^(?:Reply|Respond|Return|Output|Print)\s+(?:with\s+)?(?:exactly|only)\s+"
    r"[\"'`]*[A-Z0-9]+(?:_[A-Z0-9]+)+\b")
RE_INFRA_CHECK = re.compile(
    r"^(?:(?:Infrastructure|System|Service)\s+(?:verification|self[- ]check|health[- ]check)"
    r"|(?:Self[- ]check|Health[- ]check|Smoke[- ]test))\s+(?:only|task|probe)\b", re.I)
RE_ACK = re.compile(r"^(?:嗯+|噢+|哦+|好(?:的|哒|吧|呀|啦)?|行(?:吧|的)?|对(?:的)?|是的|收到|知道了|ok|yes)$", re.I)


def synthetic_reason(text: str) -> str | None:
    """Remaining A6 only: anchored English probe syntax, not a sentence whitelist."""
    t = RE_APP_HEADER.sub("", text or "", count=1).strip()
    if not re.search(r"[一-鿿]", t) and (RE_PROBE_REPLY.match(t) or RE_INFRA_CHECK.match(t)):
        return "english_infrastructure_probe"
    return None


def frontgate_reason(text: str) -> str | None:
    # Reuse Claude's d65ac82 A3/A4/A5 classifier verbatim; do not duplicate it.
    clean = strip_ts(RE_APP_HEADER.sub("", text or "", count=1)).strip()
    if RE_SERVER_TASK.search(clean):
        return "app_control_marker" if clean.startswith("[[") else "synthetic_task_header"
    reason = synthetic_reason(text)
    if reason:
        return reason
    # Closed acknowledgement forms, NOT a length-only veto: 封号/地区/考研/120 survive.
    t = (text or "").strip()
    core = "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")
    if core and len(core) <= 3 and RE_ACK.fullmatch(core):
        return "short_ack"
    if t and not core and any(unicodedata.category(ch) == "So" for ch in t):
        if sum(unicodedata.category(ch) == "So" for ch in t) <= 3:
            return "short_emoji_ack"
    return None


def clean_message(text: str) -> str | None:
    """返回她真正说的话；服务器任务/系统包装返回 None（不是她的话，不查）。"""
    t = text or ""
    if RE_SERVER_TASK.search(strip_ts(t).strip()):
        return None
    t = strip_ts(RE_APP_HEADER.sub("", t, count=1)).strip()
    if synthetic_reason(t):
        return None
    return t or None


# ── 候选池 ──────────────────────────────────────────────────────────────────
def _vec_queries(text: str, context: list[str], rewrites: list[str]) -> list[str]:
    qs = [text]
    if len(re.sub(r"\s+", "", text)) < SHORT_MSG and context:        # 短句（「那个呢」「不值这个价」）补上最近一轮上文
        qs.append((context[-1][:120] + " " + text).strip())
    qs += rewrites
    seen, out = set(), []
    for q in qs:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _terms(con, texts: list[str], extra: list[str]) -> list[str]:
    import retrieve as R
    from router import keywords
    ts: list[str] = []
    for i, t in enumerate(texts):
        ts += keywords(t, k=4 if i == 0 else 3)
    ts = extra + ts
    seen, uniq = set(), []
    for t in ts:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return R.by_rarity(con, uniq)[:10]



def _blocked_stem_compounds(text: str) -> list[str]:
    """COMPOUND_STEMS 前接 1~2 个汉字 → FTS cue（小模型、思考摘要、调用工具）。
    前缀里若有「的/个/和…」则跳过；每个命中只取最长合法合成。"""
    t = text or ""
    out, seen = [], set()
    stop = set("的了吗呢个和是在有就都也与或及把被让给到从")

    def _han(ch: str) -> bool:
        return 0x4E00 <= ord(ch) <= 0x9FFF

    for stem in COMPOUND_STEMS:
        if stem not in t:
            continue
        for m in re.finditer(re.escape(stem), t):
            a, b = m.start(), m.end()
            picked = None
            for n in (2, 1):                                 # 先试更长
                if a < n or not all(_han(t[a - i]) for i in range(1, n + 1)):
                    continue
                prefix = t[a - n:a]
                if any(c in stop for c in prefix):
                    continue
                w = t[a - n:b]
                if w in SOFT_BLOCK or w in COMPOUND_STEMS:
                    continue
                picked = w
                break
            if picked and picked not in seen:
                seen.add(picked)
                out.append(picked)
    return out


# ── 新旧顶替（召回时算，不靠入库时打标） ──────────────────────────────────────
_RECENCY_STOP = set("the a an and or of to in on for with is are was were be been it its this that what how when where who which do does did i you he she we they my your his her our their me him us them have has had will would can could should may might just really very also then than so as at by from about into over after before currently recently now today yesterday tomorrow still already".split())


def _key_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _RECENCY_STOP}


def superseded_map(pool: list[dict], query: str) -> dict[str, str]:
    """同一件事的新旧两条分在一组，返回 {旧 ext_id: 新 ext_id}。

    分组信号：两条事实里含查询关键词的句子共享 ≥2 个「 distinctive 词」——
    在本池里出现频率低（≤30% 的事实）的实词，比如 Fitbit Charge 3、pre-1920。
    泛词（using、think、my 这类到处都有的词）不算数，否则毫不相关的事实也会被分到一组。
    只在组内按日期分新旧，不删不封旧的——旧的只是被标出来、排后一点。
    在随机抽的 30 对无关事实里误分组是 0 对（2026-11-21 实测）。"""
    if len(pool) < 2:
        return {}
    n = len(pool)
    df: dict[str, int] = {}
    contents = []
    for h in pool:
        ws = _key_words(h.get("content") or "")
        contents.append(ws)
        for w in ws:
            df[w] = df.get(w, 0) + 1

    def distinctive(w: str) -> bool:
        return len(w) > 2 and df.get(w, 0) <= max(2, int(0.5 * n))

    qterms = {w for w in re.findall(r"[a-z0-9']+", (query or "").lower()) if distinctive(w)}
    if not qterms:
        return {}
    # 锚词：问题里在本池中出现最少的词（最能代表“这件事”本身，如 Fitbit 之于健身追踪问题）。
    # 分组必须共享锚词，否则“currently/taking”这类泛词一共享就会把毫不相关的事实分到一组
    # （实测：不加锚词时，国会新闻会被分进拖鞋组、轮胎会被分进 Instagram 组）。
    # df 为 0 的词不在任何事实里出现，当不了锚词。
    cand = [w for w in qterms if df.get(w, 0) > 0]
    if not cand:
        return {}
    anchor = sorted(cand, key=lambda w: (df.get(w, 0), -len(w), w))[0]

    def key_sents(h: dict) -> list[set[str]]:
        out = []
        for s in re.split(r"(?<=[.!?])\s+", h.get("content") or ""):
            ws = _key_words(s)
            if ws & qterms:
                out.append({w for w in ws if distinctive(w)})
        return out

    keys = [key_sents(h) for h in pool]
    parent = list(range(len(pool)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            shared = 0
            anchored = False
            for sa in keys[i]:
                for sb in keys[j]:
                    inter = sa & sb
                    shared = max(shared, len(inter))
                    if anchor in inter:
                        anchored = True
            if shared >= 2 and anchored:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(len(pool)):
        groups.setdefault(find(i), []).append(i)

    def when(h: dict) -> str:
        return (h.get("valid_from") or h.get("occurred_at") or h.get("created_at") or "")

    out: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        newest = max(members, key=lambda i: when(pool[i]))
        for i in members:
            if i != newest and when(pool[i]) and when(pool[i]) < when(pool[newest]):
                out[pool[i]["ext_id"]] = pool[newest]["ext_id"]
    return out


def order_by_xscore(pool: list[dict], xs: list[float], *, eps: float = 0.1) -> list[int]:
    """按交叉编码器分排序；分差在 eps 以内的视作并列，日期新的排前面。

    精排分对新旧两条几乎一样（同一件事、只差一个数字），分差是噪声；
    这时该由日期定先后，而不是让旧值凭噪声占住第一。"""
    import math

    def when(h: dict) -> str:
        return (h.get("valid_from") or h.get("occurred_at") or h.get("created_at") or "")

    idx = sorted(range(len(pool)), key=lambda i: -(xs[i] or 0.0))           # 同日期（或都没日期）时仍按分
    idx = sorted(idx, key=lambda i: when(pool[i]), reverse=True)
    return sorted(idx, key=lambda i: -math.floor((xs[i] or 0.0) / eps))


def build_pool(con, text: str, context: list[str], ents: list[dict], rewrites: list[str], *,
               cap: int = POOL_CAP, trace: list | None = None, per_term: int = FTS_PER_TERM,
               vec_k: int = VEC_K, channel_cap: int | None = None, compat=None) -> list[dict]:
    """FTS + 向量 + 专名三路并联，加权 RRF 合并，专名加分，取前 cap 条。每路先封顶再融合（Hindsight fusion 的做法）。"""
    import hybrid
    from echo import echo_sources
    tr = trace if trace is not None else []
    ent_lists: dict[str, list[dict]] = {}
    ent_meta: dict[str, dict] = {}
    # 只有直通专名走专名路进池；软专名里的泛词（SOFT_BLOCK）连 FTS cue 都不当。
    for e in ents[:4]:
        if not e["direct"]:
            continue
        hits = ENT.facts_for(con, e["term"], limit=max(ENTITY_PER_TERM, cap) if channel_cap else ENTITY_PER_TERM)
        ent_lists["entity:" + e["term"]] = hits
        for h in hits:
            m = ent_meta.setdefault(h["ext_id"], {"direct": True, "weight": e["weight"], "terms": []})
            m["terms"].append(e["term"])
    fts, dist, seen_f, seen_d = [], [], set(), set()
    cue = [e["term"] for e in ents
           if e["term"] not in SOFT_BLOCK and (e["direct"] or e["kind"] != "config")][:3]
    cue = list(dict.fromkeys(_blocked_stem_compounds(text) + cue))  # 小模型等合成短语优先
    fts_terms = _terms(con, [text] + rewrites, cue)
    for rank, t in enumerate(fts_terms):
        for h in hybrid._fts_any(con, t, per_term):
            if h["ext_id"] not in seen_f:
                seen_f.add(h["ext_id"])
                fts.append(dict(h, term_rank=rank))
        for h in hybrid._fts_any(con, t, per_term, extracted_only=True):
            if h["ext_id"] not in seen_d:
                seen_d.add(h["ext_id"])
                dist.append(dict(h, term_rank=rank))
    fts.sort(key=lambda h: (h["term_rank"], -float(h.get("score") or 0)))
    dist.sort(key=lambda h: (h["term_rank"], -float(h.get("score") or 0)))
    vq = _vec_queries(text, context, rewrites)
    try:
        from vectors import search_many
        vec_lists = search_many(con, vq, k=vec_k, floor=VEC_FLOOR)
    except Exception:                      # noqa: BLE001  没装 numpy / 没配嵌入 → 向量路为空，照旧走全文+专名
        vec_lists = []
    lists = {"fts": fts, "distilled": dist}
    weights: dict[str, float] = {}
    for i, v in enumerate(vec_lists):
        lists[f"vector{i}"] = v
    for k, v in ent_lists.items():
        lists[k] = v
        weights[k] = ENTITY_RRF_WEIGHT
    if channel_cap:
        lists = {k: v[:channel_cap] for k, v in lists.items()}
    merged = hybrid.rrf_merge(lists, weights)
    vsim: dict[str, float] = {}
    for v in vec_lists:
        for h in v:
            vsim[h["ext_id"]] = max(vsim.get(h["ext_id"], 0.0), float(h.get("score") or 0))
    bad = hybrid._disowned(con, list(merged))
    merged = {k: v for k, v in merged.items() if k not in bad
              and (compat is None or compat(text, v.get("content") or ""))}
    for k, h in merged.items():
        # via 里的 vector0/vector1/entity:xx 统一成 vector / entity，下游展示用
        vs = {("vector" if v.startswith("vector") else "entity" if v.startswith("entity:") else v) for v in h["via"]}
        h["via"] = sorted(vs)
        if k in vsim:
            h["vec_sim"] = round(vsim[k], 3)
            h["rrf"] += VEC_SIM_BONUS * max(0.0, vsim[k] - 0.5)
        m = ent_meta.get(k)
        if m:
            h["entity_hit"] = True
            h["entity_direct"] = True
            h["entity_terms"] = m["terms"]
            # 不加 RRF 分：专名只负责进池，最终注入看精排分（Serein）
    try:
        echo = echo_sources(con, list(merged))
    except Exception:                                        # noqa: BLE001
        echo = {}
    try:
        import stale
        stale_marks = stale.marks_for(con, list(merged))
    except Exception:                                        # noqa: BLE001
        stale_marks = {}
    # Full-on keeps provenance/status, but a pin must not buy ranking slots.
    pin_flags = {k: h.get("user_pinned") for k, h in merged.items() if h.get("user_pinned")} if compat is not None else {}
    ranking = {k: dict(h, user_pinned=0) if k in pin_flags else h for k, h in merged.items()}
    pool = hybrid.finalize(ranking, cap=max(cap * 3, 60), echo=echo, stale=stale_marks)
    for h in pool:
        if h["ext_id"] in pin_flags:
            h["user_pinned"] = pin_flags[h["ext_id"]]
    # finalize 把 via 拼成字符串（专名不加分，只用 rrf+语义加成排序进池）
    # 同一件事的新旧两条：新的置顶，旧的排后并标出来（STALE_WEIGHT 同款处理，只排后、不删不封）
    try:
        sup = superseded_map(pool, text)
    except Exception:                                        # noqa: BLE001
        sup = {}
    if sup:
        from stale import STALE_WEIGHT
        by_id = {h["ext_id"]: h for h in pool}
        for old_id, new_id in sup.items():
            h = by_id.get(old_id)
            if h is not None:
                h["superseded_by"] = new_id
                h["stale_by"] = new_id
                h["hybrid_score"] = round(h.get("hybrid_score", 0) * STALE_WEIGHT, 5)
                h["score"] = h["hybrid_score"]
        pool.sort(key=lambda h: (-h["hybrid_score"], -float(h.get("act_score") or 0)))
    pool = pool[:cap]
    try:
        import echo as _echo
        _echo.annotate_origin(con, pool)
        _echo.annotate(con, pool)
        _echo.demote_say(pool)
    except Exception:                                        # noqa: BLE001
        pass
    tr.append(f"池：FTS {len(fts)}＋整理过的事 {len(dist)}＋向量 {sum(len(v) for v in vec_lists)}（{len(vq)} 句）"
              f"＋专名 {sum(len(v) for v in ent_lists.values())} → 合并取 {len(pool)} 条")
    return pool


def relevance_probe(con, text: str, pool: list[dict]) -> dict[str, float]:
    """Cheap query-specific signal, never activation/RRF/pin status as relevance.

    max(cached vector cosine, corpus-aware lexical coverage). This is a COST gate,
    not an injection score; surviving candidates still require model relevance.
    """
    import retrieve as R
    from router import keywords
    terms = R.by_rarity(con, keywords(text, k=10))
    weights = {t.lower(): 1.0 / math.log2(n + 2)
               for t in terms if 0 < (n := R.rarity(con, t)) <= R.TOO_COMMON}
    den = sum(weights.values())
    out = {}
    for h in pool:
        body = (h.get("content") or "").lower()
        def present(t):
            return bool(re.search(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])", body)) if t.isascii() else t in body
        lexical = sum(w for t, w in weights.items() if present(t)) / den if den else 0.0
        out[h["ext_id"]] = max(out.get(h["ext_id"], 0.0), float(h.get("vec_sim") or 0.0), lexical)
    return out


# ── 选择（注入） ──────────────────────────────────────────────────────────────
def select(pool: list[dict], text: str, ents: list[dict], j: dict | None, *, explicit: bool = False,
           compat=None, forced_fn=None, fallback_gate=None, max_hits: int = MAX_HITS, full_on: bool = False) -> tuple[list[dict], bool | None, str]:
    """从池里挑要注入的。返回 (selected, need, how)。

    · 专名直通：专名候选 ≥ENTITY_MIN(7)，其他 ≥9；不绕过精排分
    · 日常无专名：need=true 且 ≥NORMAL_MIN(8)；明确回忆口吻 ≥EXPLICIT_MIN(7)
    · 无专名时钉住条 ≥OTHER_WITH_ENTITY_MIN(9)，防 pinned 绕过
    · need=false 时 forced_fn / 明确回忆兜底也必须过对应分门槛（不硬塞）
    · 精排失败：不回退关键词闸；仅直通专名按池序
    """
    compat = compat or (lambda q, c: True)
    direct_ents = [e for e in ents if e["direct"]]
    ent_idx = [i for i, h in enumerate(pool) if h.get("entity_hit") and h.get("entity_direct")]
    gate = EXPLICIT_MIN if explicit else NORMAL_MIN
    if j is None:
        if full_on:
            return [], None, "精排不可用，不注入（full-on）"
        if direct_ents and ent_idx:
            return [pool[i] for i in ent_idx if not (pool[i].get("from_pin") or pool[i].get("user_pinned"))][:max_hits], True, "专名直通（精排不可用，钉住条不走兜底）"
        return [], None, "精排不可用，不注入"
    scores = list(j["scores"]) + [0] * max(0, len(pool) - len(j["scores"]))

    def adj(i):
        h = pool[i]
        return scores[i] * (ECHO_W if (h.get("echo_from") or h.get("assistant_say")) else 1.0)

    def thr(i):
        # 无专名路径里，钉住条更严，避免纠正卡/画像条靠置顶+改写混进来（回放 #12）
        if pool[i].get("from_pin") or pool[i].get("user_pinned"):
            return OTHER_WITH_ENTITY_MIN
        return gate

    if direct_ents and ent_idx:
        ranked = sorted((i for i in ent_idx), key=lambda i: (adj(i), scores[i], -i), reverse=True)
        chosen = [i for i in ranked if scores[i] >= (OTHER_WITH_ENTITY_MIN if (pool[i].get("from_pin") or pool[i].get("user_pinned")) else ENTITY_MIN)
                  and compat(text, pool[i].get("content") or "")]
        others = sorted((i for i in range(len(pool)) if i not in ent_idx and scores[i] >= OTHER_WITH_ENTITY_MIN
                         and compat(text, pool[i].get("content") or "")), key=lambda i: (adj(i), scores[i]), reverse=True)
        chosen = (chosen + others)[:max_hits]
        return [pool[i] for i in chosen], True, "专名直通+精排排序"

    if full_on:
        order = sorted(range(len(pool)), key=lambda i: (adj(i), scores[i], -i), reverse=True)
        chosen = [i for i in order if scores[i] >= thr(i) and compat(text, pool[i].get("content") or "")]
        return [pool[i] for i in chosen[:max_hits]], bool(chosen), "逐条相关性（need仅观测）"

    need = bool(j["need"])
    forced = [i for i, h in enumerate(pool)
              if forced_fn and forced_fn(text, h.get("content") or "")
              and scores[i] >= thr(i) and compat(text, h.get("content") or "")]
    how = "精排"
    if not need and not forced and explicit:
        how = "精排+明确回忆兜底"
        top = sorted(zip(scores, range(len(pool))), reverse=True)
        forced = [i for sc, i in top if sc >= thr(i) and compat(text, pool[i].get("content") or "")][:max_hits]
    if not need and not forced:
        return [], False, "精排判不用翻旧账"
    if not need and forced:
        return [pool[i] for i in forced][:max_hits], True, how

    # need=true：双重闸 —— 全局最高分也必须过门槛，再按条过滤
    if not scores or max(scores[i] for i in range(len(pool))) < gate:
        return [], False, "精排最高分不够"
    order = sorted(range(len(pool)), key=lambda i: (adj(i), scores[i], i), reverse=True)
    chosen = [i for i in order if scores[i] >= thr(i) and compat(text, pool[i].get("content") or "")]
    return [pool[i] for i in chosen][:max_hits], True, how


# ── 全开路径的交叉编码器精排（抄 Graphiti BGERerankerClient + reranker_min_score；mem0 search→rerank 同一套）──
# 阈值在 44 句按入库时间截断的回放上定。
XMIN = float(__import__("os").environ.get("ENGRAM_XRERANK_MIN", "0.45"))
XMIN_EXPLICIT = float(__import__("os").environ.get("ENGRAM_XRERANK_MIN_EXPLICIT", "0.25"))  # 句里有「之前/上次/还记得」时
XMIN_DERIVED = float(__import__("os").environ.get("ENGRAM_XRERANK_MIN_DERIVED", "0.8"))  # 〔归纳〕/钉住条更严
XSECOND_RATIO = 0.7      # 第二条要不低于第一条的 70%，防止「一条对、一条凑数」
XFALLBACK_CAP = 15       # 交叉编码器挂了才退回大模型精排，池截到 15 条免得超时


def _xscores(query, docs):
    import xrerank
    return xrerank.scores(query, docs)


def _xquery(text: str) -> str:
    """交叉编码器的查询句：去掉两位当事人的称呼（用户昵称/助手名/Claude…）。
    这些称呼几乎每条记忆里都有，对交叉编码器就是停用词——不去掉，「笨蛋＋助手名」「Claude 念法」这种闲聊
    会因为字面重合拿到 0.9 以上的分（2026-11-12 回放）。"""
    import persona
    names = [persona.get("ENGRAM_NAME_USER"), persona.get("ENGRAM_NAME_CLAUDE"), persona.get("ENGRAM_USER_FULLNAME")]
    names += (persona.get("ENGRAM_USER_ALIASES") or "").split(",") + (persona.get("ENGRAM_CLAUDE_ALIASES") or "").split(",")
    names += ["Claude", "claude"]
    q = strip_ts(text or "")
    for n in sorted({n.strip() for n in names if n and n.strip()}, key=len, reverse=True):
        q = q.replace(n, " ")
    return re.sub(r"\s+", " ", q).strip()


def _xdoc(h: dict) -> str:
    return (h.get("content") or "").strip()


def select_x(pool: list[dict], text: str, xs: list[float], *, compat=None, max_hits: int = MAX_HITS,
             explicit: bool = False):
    compat = compat or (lambda q, c: True)
    base = XMIN_EXPLICIT if explicit else XMIN
    def need(h):
        c = h.get("content") or ""
        strict = h.get("from_pin") or h.get("user_pinned") or c.startswith("〔归纳〕")
        return max(XMIN_DERIVED, base) if strict else base
    order = order_by_xscore(pool, xs)
    chosen = [i for i in order if xs[i] >= need(pool[i]) and compat(text, pool[i].get("content") or "")]
    if len(chosen) > 1 and xs[chosen[1]] < xs[chosen[0]] * XSECOND_RATIO:
        chosen = chosen[:1]
    return [pool[i] for i in chosen[:max_hits]], f"交叉编码器≥{base}（{len(chosen)} 条过线）"


def recall(con, text: str, context: list[str] | None = None, *, judge=None, rewriter=None, use_rewrite: bool | None = None,
           explicit: bool = False, compat=None, forced_fn=None, fallback_gate=None, max_hits: int = MAX_HITS,
           cap: int = POOL_CAP, extra: list[dict] | None = None, full_on: bool = False,
           pool_min: float | None = None, xscorer=None) -> Flow:
    """聊天注入用：专名直通 → 查询改写 → 池 → 精排 → 选 ≤2 条。judge/rewriter 可注入（测试、回放、成本估算）。"""
    started = time.perf_counter()
    ctx = [c for c in (context or []) if c][-CONTEXT_TURNS:]
    f = Flow(text=text, calls={"rewrite": 0, "rerank": 0})
    def finish():
        f.timings["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return f
    blocked = frontgate_reason(text)
    if blocked:
        f.reason, f.how = "frontgate:" + blocked, "前闸跳过"
        return finish()
    f.entities = ENT.match(con, text)
    direct = [e for e in f.entities if e["direct"]]
    if f.entities:
        f.trace.append("专名：" + "、".join(f"{e['term']}({'直通' if e['direct'] else '软'},{e['kind']},df{e['df']})"
                                          for e in f.entities[:5]))
    do_rw = rewrite_on() if use_rewrite is None else use_rewrite
    if full_on:
        do_rw = False  # no paid rewrite ahead of the cheap relevance/cost gate
    if do_rw:
        import query_rewrite as QR
        if QR.should_rewrite(text, has_direct_entity=bool(direct)):
            r = (rewriter or QR.rewrite)(text, ctx)
            f.calls["rewrite"] = 1
            f.rewrites = list(r.get("questions") or [])
            f.trace.append("改写：" + " / ".join(f.rewrites) if f.rewrites else "改写：无")
    pool_started = time.perf_counter()
    if full_on:
        cap = max(cap, FULL_POOL_CAP)
        check = compat or (lambda q, c: True)
        f.pool = build_pool(con, text, ctx, f.entities, f.rewrites, cap=cap, trace=f.trace,
                            per_term=12, vec_k=24, channel_cap=FULL_CHANNEL_CAP, compat=check)
        # Pins do not reserve front slots or bypass compatibility/relevance.
        extra = [dict(h, from_pin=True) for h in (extra or []) if check(text, h.get("content") or "")]
        probe = relevance_probe(con, text, f.pool + extra)
        for h in f.pool:
            if h.get("user_pinned"):
                h["from_pin"] = True
        existing = {h["ext_id"]: h for h in f.pool}
        for h in extra:
            if probe.get(h["ext_id"], 0.0) < (FULL_POOL_MIN if pool_min is None else pool_min):
                continue
            if h["ext_id"] in existing:
                existing[h["ext_id"]]["from_pin"] = True
            elif len(f.pool) < cap:
                f.pool.append(h)
                existing[h["ext_id"]] = h
        floor = FULL_POOL_MIN if pool_min is None else pool_min
        score = max(relevance_probe(con, text, f.pool).values(), default=0.0)
        f.trace.append(f"cheap_relevance={score:.3f},floor={floor:.3f}（仅观测；精排改交叉编码器后不再按它拦）")
    else:
        f.pool = build_pool(con, text, ctx, f.entities, f.rewrites, cap=cap, trace=f.trace)
        if extra:
            pinned = [dict(h, from_pin=True) for h in extra]
            seen = {h["ext_id"] for h in pinned}
            f.pool = (pinned + [h for h in f.pool if h["ext_id"] not in seen])[:cap]
            f.trace.append(f"钉住候选 {len(extra)} 条置顶")
    f.timings["pool_ms"] = round((time.perf_counter() - pool_started) * 1000, 3)
    if not f.pool:
        f.reason, f.how = "no_hits", "池为空"
        return finish()
    if full_on:
        rerank_started = time.perf_counter()
        xq = _xquery(text)
        xs = (xscorer or _xscores)(xq or text, [_xdoc(h) for h in f.pool])
        f.timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 3)
        if xs is not None:
            f.calls["xrerank"] = 1
            f.judged = True
            f.judge_scores = [round(x, 4) for x in xs]
            f.selected, f.how = select_x(f.pool, text, xs, compat=compat, max_hits=max_hits, explicit=explicit)
            f.need = bool(f.selected)
            f.reason = f"selected:{len(f.selected)}/{len(f.pool)}"
            return finish()
        f.trace.append("交叉编码器失败，退回大模型精排（前 %d 条）" % XFALLBACK_CAP)
        f.pool = f.pool[:XFALLBACK_CAP]
        cap = XFALLBACK_CAP
    if judge is None:
        from rerank import judge as _j
        judge = _j
    q = "\n".join(ctx + [text])
    names = [e["term"] for e in direct]
    rerank_started = time.perf_counter()
    options = {"relevance_only": True} if full_on else {}
    j = judge(q, [_label(h) + (h.get("content") or "") for h in f.pool], entities=names or None, max_candidates=cap, **options)
    f.timings["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 3)
    f.judge_scores = list(j.get("scores", [])) if j else []
    f.model_need = bool(j.get("need")) if j else None
    f.calls["rerank"] = 1
    f.judged = j is not None
    f.selected, f.need, f.how = select(f.pool, text, f.entities, j, explicit=explicit, compat=compat,
                                       forced_fn=forced_fn, fallback_gate=fallback_gate, max_hits=max_hits, full_on=full_on)
    f.reason = f"selected:{len(f.selected)}/{len(f.pool)}"
    return finish()


def topk(con, query: str, k: int = 10, context: list[str] | None = None, *, rewrite: bool = False, rewriter=None,
         autobuild_entities: bool = False) -> list[dict]:
    """评测/召回接口：返回池里前 k 条（默认 10），**不经**聊天的 2 条 / ≥7 分闸，也不调精排。
    评测库是临时库：autobuild_entities=True 时先建专名表（聊天库由维护脚本建，钩子只读）。"""
    if autobuild_entities and not ENT.table_exists(con):
        ENT.refresh(con, use_jieba=False)
    ctx = [c for c in (context or []) if c][-CONTEXT_TURNS:]
    ents = ENT.match(con, query)
    rws: list[str] = []
    if rewrite:
        import query_rewrite as QR
        if QR.should_rewrite(query, has_direct_entity=any(e["direct"] for e in ents)):
            rws = list((rewriter or QR.rewrite)(query, ctx).get("questions") or [])
    pool = build_pool(con, query, ctx, ents, rws, cap=max(k, POOL_CAP_BENCH), per_term=max(12, k), vec_k=max(12, k))
    return pool[:k]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="召回改造：试一句话（只读；不调模型，除非 --rewrite）")
    ap.add_argument("text")
    ap.add_argument("--db")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--rewrite", action="store_true")
    a = ap.parse_args()
    from store.store import connect
    con = connect(a.db) if a.db else connect()
    for i, h in enumerate(topk(con, a.text, k=a.k, rewrite=a.rewrite), 1):
        print(f"{i:2d} {h['hybrid_score']:.4f} [{h['via']}] {h['ext_id'][5:13]} {(h['content'] or '')[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
