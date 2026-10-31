#!/usr/bin/env python3
"""Lumen Engram · 抽取器第二版：原话 → 事（LangMem 那一块）

库里的 facts 大多是原话片段直接升上来的，一条就是一句聊天，召回时糊成一团。
这一版按时间把证据切块，交给模型抽「事」：谁、什么时候、做了/说了/定了什么。

几条硬规矩（照框架页和 Loci 的教训）：
  1. 每条事必须带一句 quote，而且 quote 必须**逐字**出现在块里某一条证据里。
     对不上就丢——模型编的东西进不了库。
  2. 事只挂在 quote 所在的那条证据上，证据回得到原始会话行号。
  3. 抽出来的一律 candidate，不许 confirmed；confidence 用模型给的，不用来决定存不存。
  4. subject 只能是固定名单里的，代词、外号不收。
  5. 不改、不删任何已有的 facts / evidence。

用法：
    python3 -m extract.from_evidence --since 2026-09-22 --until 2026-09-23 --dry-run
    python3 -m extract.from_evidence --since 2026-09-22 --until 2026-09-23
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store.store import (add_evidence, connect, promote, record_class, record_tier,  # noqa: E402
                         sha256)
from llm import chat  # noqa: E402
import echo  # noqa: E402
from extract import timeline  # noqa: E402
from extract.from_channel import WHO, beijing, harvest  # noqa: E402

import persona  # noqa: E402
ENV_FILE = persona.env_file()
EXTRACTOR = "extract-llm-v1"
SUBJECTS = {*persona.ACTORS, "us"}
TYPES = {"episodic", "semantic", "preference", "procedural"}
CHUNK_CHARS = 6000          # 一块喂给模型的原话总长
PER_MSG = 600               # 单条原话最长截多少
TIMEOUT_S = 90
MIN_QUOTE = 4

PROMPT = persona.fill("""下面是一段按时间排好的聊天原话。说话的人：@@ROSTER_USER@@、@@CLAUDE@@（我，AI）、@@GPT@@（@@DESC_GPT@@）、@@C@@（@@DESC_C@@）。
每条前面方括号里是编号。

从里面抽出**过一个月还可能想得起来的事**。只抽原话里明明白白有的，不推测、不补全。
宁多勿漏：拿不准值不值得记的，先记下来，confidence 给低一点（0.4～0.6）；后面有人定期检查，多了能删，漏了找不回来。

要抽（包括日常和感情里的事，不只是正事）：
- 两人之间发生的有来龙去脉的小事：谁逗了谁、谁吃了谁的醋、为什么、怎么收场（如「@@EX_PLAYFUL@@」）
- 她说出口的心意、承诺、在意的话，原话有分量的照抄进 quote
- 她的生活：买了什么、去了哪、身体怎样、在忙什么、用了哪些 App/AI
- 定下的做法、改过的设置、踩过的坑

不要抽：
- 只有打招呼、语气词、表情、单纯一句「嗯」「知道了」的
- 对眼前小事的一句指令（「去睡觉」「先吃饭」）
- @@CLAUDE@@自己的安慰、建议、推测——除非@@USER@@接受了、变成了约定
- 网上看到的别人的事、段子，除非@@USER@@明确表态了自己的看法（那就记她的看法）

@@GLOSSARY_BLOCK@@认不出是谁就写「某人」，**不要猜成同学、组员、朋友**。

注意说话人：谁说的就是谁的事。@@USER@@转述别人（比如「我让他教我」）时，「他」是谁看上下文，看不出就别抽。
亲密内容记「几点有过亲密互动」、她明确说出的偏好和有分量的话，不记动作和身体细节。

每条事给：
- subject：这件事主要关于谁。只能是 @@A_USER@@ / claude / gpt / @@A_C@@ / us（us = @@USER@@和@@CLAUDE@@两个人之间的事）。
  **硬规矩：subject 只能是 quote 所在那句话的说话人本人。** @@CLAUDE@@说「@@USER@@怎样怎样」只是转述，不要记成@@USER@@的事；
  @@USER@@说的才能记成@@USER@@的事。拿不准这句话是谁说的、或只是别人转述，就不要抽。
  **us 要双方的原话**：只有当@@USER@@和对方各有一句原话都能证明这件事（比如一方提出、另一方明确答应），才写 us，
  并且必须再给 quote2 / src2（另一方那句原话，逐字复制、编号对应）。只有一方的话（尤其只有@@CLAUDE@@一个人说「我们约定」「@@USER@@答应了」）
  不是 us——是@@CLAUDE@@说的就写 claude（内容必须以「@@CLAUDE@@」为主语），是@@USER@@说的就写 @@A_USER@@。
  content 里出现的年份、日期、数字，必须能在 quote 或那句原话里找到；原话没写年份就别补年份；消息编号、ID 不是日期。
- type：
    episodic   = 某个时间发生的一件具体的事（带上日期）
    semantic   = 稳定的事实（身份、关系、经历、客观情况）
    preference = 喜欢/讨厌/希望别人怎么做
    procedural = 定下的做法、规矩、踩过的坑、以后该怎么办
- content：一句中文，第三人称，写清楚是谁、什么事；有日期就写日期。不超过 80 字。不加引号不加评价。
- quote：从**某一条**原话里逐字复制的一小段（4～40 字），能证明这件事。必须一字不差，包括标点和 emoji。
- src：quote 所在那条原话的编号
- confidence：0～1。原话直接说了=0.9；需要结合上下文才看得出=0.6；有点拿不准=0.4

只输出 JSON：{{"facts": [{{"subject": "...", "type": "...", "content": "...", "quote": "...", "src": 编号, "quote2": "（仅 us 需要）", "src2": 编号, "confidence": 0.9}}, ...]}}
没有值得记的就输出 {{"facts": []}}。

原话：
{chunk}""")


from llm import load_env as _env  # noqa: E402,F401  共用配置读取（见 llm.py）


def build_prompt(chunk_text: str) -> str:
    """ENGRAM_VALUE_GATE 开：用封闭类别 + 「预计拒绝大多数」的提示词；关：原提示词一字不差。"""
    import value_gate as VG
    if VG.enabled():
        return VG.PROMPT_GATE.format(chunk=chunk_text, extra=VG.EXTRA_RULES)
    return PROMPT.format(chunk=chunk_text)


def prefilter_evs(evs: list[dict]) -> list[dict]:
    """确定性预过滤（零 LLM；只在 VALUE_GATE 开时有效）：心跳/应答/工具输出/JSON/Error/太短的证据不送模型。"""
    import value_gate as VG
    if not VG.enabled():
        return evs
    out = []
    for ev in evs:
        why = VG.prefilter(ev["content"])
        if why in VG.HARD_PRE:               # 应答/太短的留着当上下文（check() 里不许它当引文来源）
            VG.reject(why)
        else:
            out.append(ev)
    return out


def call_model(chunk_text: str) -> list[dict]:
    # 主路由（DIGEST_*）失败会自动换备用（FALLBACK_*），见 llm.py
    text = chat([
        {"role": "system", "content": "你只输出合法 JSON。"},
        {"role": "user", "content": build_prompt(chunk_text)},
    ], purpose="extract", max_tokens=2500, timeout=TIMEOUT_S)
    m = re.search(r"\{.*\}", text, re.S)
    got = json.loads(m.group(0) if m else text)
    facts = got.get("facts", []) or []
    return facts + call_model_old(chunk_text)


def union_on() -> bool:
    """召回补偿（ENGRAM_VALUE_GATE_UNION，默认关）：闸开时，旧抽取器提示词在同一片段上也跑一遍，候选并进来过同一套后检。"""
    try:
        import flags
        import value_gate as VG
        return VG.enabled() and flags.on("VALUE_GATE_UNION")
    except Exception:                                        # noqa: BLE001
        return False


def call_model_old(chunk_text: str) -> list[dict]:
    """召回补偿的第二路：旧提示词（PROMPT）的输出，每条打上 _old=True。没开/出错＝空（新提示词的结果照用）。"""
    if not union_on():
        return []
    try:
        text = chat([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": PROMPT.format(chunk=chunk_text)},
        ], purpose="extract", max_tokens=2500, timeout=TIMEOUT_S)
        m = re.search(r"\{.*\}", text, re.S)
        got = json.loads(m.group(0) if m else text)
        return [{**f, "_old": True} for f in (got.get("facts", []) or []) if isinstance(f, dict)]
    except Exception as e:                                   # noqa: BLE001
        print(f"旧抽取器补偿路失败（忽略）：{type(e).__name__}", file=sys.stderr)
        return []


def load_evidence(con, since: str, until: str) -> list[dict]:
    """取时间窗里的聊天证据，按发生时间排。只要她、我、GPT、GLM 助手说的原话。"""
    rows = con.execute(
        """SELECT id, occurred_at, source_actor, content FROM evidence
           WHERE source_kind='chat' AND occurred_at >= ? AND occurred_at < ?
             AND source_actor IN (?,?,?,?)
           ORDER BY occurred_at, id""", (since, until, *persona.ACTORS)).fetchall()
    return [dict(r) for r in rows]


def load_jsonl(path: Path, since: str, until: str) -> list[dict]:
    """直接读会话记录：她的 channel 消息 + 我的 reply。id 为 None，命中时才落 evidence。"""
    stamps = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            m = re.search(r'"timestamp":"([^"]+)"', line)
            if m:
                stamps[n] = m.group(1)
    out = []
    for line_no, actor, body, ts in harvest(path):
        if actor not in SUBJECTS:
            continue
        stamp = ts or stamps.get(line_no)
        when_utc, when_cn = beijing(stamp) if stamp else (None, "")
        if not when_utc or not (since <= when_utc < until):
            continue
        who = WHO.get(actor, actor)
        out.append({"id": None, "occurred_at": when_utc, "source_actor": actor,
                    "path": str(path), "line": line_no,
                    "content": f"{who}（{when_cn}）：" + body.replace("\n", " ")})
    return out


def load_app(path: Path, names: list[str]) -> list[dict]:
    """claude.ai 导出的 conversations.json。只读名字在 names 里的窗口（只读指定名字的窗口）。"""
    data = json.load(open(path, encoding="utf-8"))
    out = []
    for conv in data:
        if conv.get("name") not in names:
            continue
        for msg in conv.get("chat_messages", []):
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            actor = persona.USER if msg.get("sender") == "human" else "claude"
            when_utc, when_cn = beijing(msg["created_at"])
            who = WHO.get(actor, actor)
            out.append({"id": None, "occurred_at": when_utc, "source_actor": actor,
                        "path": str(path),
                        "line": f"conv:{conv['uuid']}#msg:{msg['uuid']}",
                        "content": f"{who}（{when_cn}）：" + text.replace("\n", " ")})
    return out


def chunks(evs: list[dict]):
    buf, size = [], 0
    for ev in evs:
        text = ev["content"][:PER_MSG]
        if buf and size + len(text) > CHUNK_CHARS:
            yield buf
            buf, size = [], 0
        buf.append(ev)
        size += len(text)
    if buf:
        yield buf


def render(chunk: list[dict]) -> str:
    return "\n".join(f"[{i}] {ev['content'][:PER_MSG]}" for i, ev in enumerate(chunk))


_CN_DIGIT = {"1": "一", "2": "二两", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def _runs(text: str) -> set[str]:
    out = set()
    for r in re.findall(r"\d+", text or ""):
        out.add(r)
        out.add(r.lstrip("0") or "0")
    return out


def numbers_unsupported(content: str, *sources: str) -> list[str]:
    """task2-09 数字校验：content 里每一个数字串（年、日期、数量、编号）都必须作为完整数字串出现在
    quote / 原话里；个位数允许原话写成中文数字（6 ↔ 六）。返回找不到出处的数字串（空 = 通过）。"""
    src = " ".join(s for s in sources if s)
    have = _runs(src)
    bad = []
    for tok in re.findall(r"\d+", content or ""):
        t = tok.lstrip("0") or "0"
        if tok in have or t in have:
            continue
        if len(tok) == 1 and any(c in src for c in _CN_DIGIT.get(tok, "")):
            continue
        bad.append(tok)
    return bad


def _second_hit(fact: dict, chunk: list[dict], hit: dict):
    """us 的另一方原话：quote2 必须逐字在某条证据里，而且说话人和第一条不同。"""
    q2 = str(fact.get("quote2", "") or "").strip()
    if len(q2) < MIN_QUOTE:
        return None
    try:
        s2 = int(fact.get("src2"))
    except (TypeError, ValueError):
        s2 = -1
    cands = []
    if 0 <= s2 < len(chunk):
        cands.append(chunk[s2])
    cands += [ev for ev in chunk if ev is not hit]
    import value_gate as _V
    for ev in cands:
        if _V.enabled():
            # 值得记闸：quote2 必须出自另一个说话人、另一条（另一行）原话；同一条证据的几段、或和第一句同一行的不算
            if ev is hit or (ev.get("id") is not None and ev.get("id") == hit.get("id")):
                continue
            if _V.quote_in(q2, ev["content"]) and ev.get("source_actor") and ev.get("source_actor") != hit.get("source_actor"):
                if _V.quote_in_thinking(ev["content"], q2):            # quote2 也不许是英文思考文本
                    continue
                return ev
        elif q2 in ev["content"] and ev.get("source_actor") and ev.get("source_actor") != hit.get("source_actor"):
            return ev
    return None


def check(fact: dict, chunk: list[dict]) -> tuple[dict | None, str]:
    """过硬规矩。返回 (清洗后的事, 丢弃原因)。"""
    subj = str(fact.get("subject", "")).strip()
    typ = str(fact.get("type", "")).strip()
    content = str(fact.get("content", "")).strip()
    quote = str(fact.get("quote", "")).strip()
    import value_gate as _VG
    if not typ and _VG.enabled():                     # 值得记闸的提示词没有 type 字段：按 category 补（否则全被「type 不对」挡掉）
        typ = _VG.TYPE_FOR_CATEGORY.get(str(fact.get("category") or "").strip(), "")
    if subj not in SUBJECTS:
        return None, f"subject 不在名单：{subj}"
    if typ not in TYPES:
        return None, f"type 不对：{typ}"
    if not content or len(content) > 160:
        return None, "content 空或太长"
    if len(quote) < MIN_QUOTE:
        return None, "quote 太短"
    try:
        src = int(fact.get("src"))
    except (TypeError, ValueError):
        src = -1
    # quote 必须逐字在证据里。先看它说的那条，对不上再全块找一遍（模型编号偶尔错位）
    def _has(q, text):                                # 闸开：忽略空白差异、弯/直引号、全/半角标点、大小写（模型常改这些）
        return q in text or (_VG.enabled() and _VG.quote_in(q, text))
    hit = None
    if 0 <= src < len(chunk) and _has(quote, chunk[src]["content"]):
        hit = chunk[src]
    else:
        hit = next((ev for ev in chunk if _has(quote, ev["content"])), None)
    if hit is None:
        return None, f"quote 在原话里找不到：{quote[:30]}"
    if _VG.enabled():
        if len(_VG.norm_quote(quote)) > _VG.QUOTE_REJECT:                       # 整段/跨行照抄的不当引文
            _VG.reject("post:quote-too-long")
            return None, "post:quote-too-long：值得记闸"
        if _VG.quote_in_thinking(hit["content"], quote):                         # 英文思考文本只当上下文，不当引文来源
            _VG.reject("pre:thinking-text")
            return None, "pre:thinking-text：值得记闸"
        quote = _VG.cap_quote(quote)                                             # 超长引文截到 QUOTE_MAX 字（仍是原话子串）
        if len(quote) < MIN_QUOTE:
            return None, "quote 太短"
    try:
        conf = float(fact.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(max(conf, 0.0), 1.0)
    # 防回声（task2-08/09，没有开关）：证据说话人是 X，subject 只能是 X；us 要双方原话
    actor = hit.get("source_actor")
    hit2 = None
    reattributed = False
    if _VG.enabled() and subj == persona.USER and actor == "claude" and fact.get("category") in ("decision", "preference", "correction"):
        # 12495 型：模型把助手的建议写成了她的决定。她随后在窗口里明确同意（紧随其后的她的话）→ 改记 us 的约定（a）；没有她的同意就照旧被回声挡。
        ag = _VG.find_user_agreement(chunk, hit, content)
        if ag is not None and echo.us_ok({actor, ag.get("source_actor")}):
            subj, hit2, reattributed = "us", ag, True
    if subj == "us" and not reattributed:
        hit2 = _second_hit(fact, chunk, hit)
        if hit2 is None or not echo.us_ok({actor, hit2.get("source_actor")}):
            hit2 = None
            new = echo.resolve_us(actor, content)           # 单方证据：落回说话人本人，或丢弃
            if new is None:
                return None, f"subject-us：只有 {actor} 一方的原话，不能记成 us 的事"
            subj = new
    if hit2 is None and not echo.owner_ok(actor, subj):
        return None, f"subject-echo：{actor} 的话不能记成 {subj} 的事"
    # 数字校验（task2-09）：年、日期、数字必须出自 quote 或原话，不许模型补
    import value_gate as VG
    category = "agreement" if reattributed else fact.get("category")
    if VG.enabled():
        # 值不值得记闸（只在开关开时）：预过滤 → 角色规则 → 后检；置信度分档。每次拒绝按原因计数。
        actors = {actor} | ({hit2.get("source_actor")} if hit2 else set())
        if fact.get("_old") and not category:                  # 旧抽取器（召回补偿路径）没有 category：按关键词推；推不出来的不要
            category = VG.infer_category_union(content, actors, subj, typ)
            if category in (None, "other"):
                VG.reject("union:old-uncategorized")
                return None, "union:old-uncategorized：值得记闸"
        confirmed, us_reject = True, None
        if hit2 is not None:
            first, later = (hit, hit2) if chunk.index(hit) < chunk.index(hit2) else (hit2, hit)
            confirmed = later.get("source_actor") == persona.USER and VG.user_consents(later["content"], content)   # (a) 助手提、她随后确认（共用同意判定）
            if not confirmed and first.get("source_actor") == persona.USER and later.get("source_actor") == "claude":
                # (b) 她先提要求、助手随后明确答应：她的话得是请求，助手的话得是明确答应，事实实质得出自她的话
                us_reject = ("role:us-too-far" if chunk.index(later) - chunk.index(first) > 8      # 答应要紧跟着她的请求
                             else VG.us_request_then_accept(first["content"], later["content"], content))
                confirmed = us_reject is None
        why = (VG.prefilter(hit["content"], actor)
               or VG.role_check(subj, category if category in VG.CATEGORIES else None, actors,
                                confirmed_by_user_after=confirmed, us_reject=us_reject)
               or VG.postcheck({**fact, "subject": subj, "content": content, "category": category, "quote": quote},
                               hit["content"], actor=None if subj == "us" else actor,
                               context_text="\n".join(e["content"] for e in chunk),
                               user_text="\n".join(e["content"] for e in chunk if e.get("source_actor") == persona.USER)))
        if why:
            VG.reject(why)
            return None, f"{why}：值得记闸"
        conf = VG.confidence_for(category, subj, actor, model_conf=conf)
        if fact.get("_old") and not VG.is_priority(subj, category, content, actor, quote=quote, strict=True):
            VG.reject("union:old-not-priority")                 # 旧抽取器的候选只补「大事」，其余不要（否则回到每片 26 条）
            return None, "union:old-not-priority：值得记闸"
    srcs = [quote, hit["content"]] + ([str(fact.get("quote2", "")), hit2["content"]] if hit2 else [])
    bad = numbers_unsupported(content, *srcs)
    if bad:
        return None, f"number-unsupported：{'/'.join(bad[:3])} 在原话里找不到"
    out = {"subject": subj, "type": typ, "content": content, "quote": quote,
           "evidence_id": hit["id"], "hit": hit, "hit2": hit2,
           "evidence_id2": hit2["id"] if hit2 else None,
           "occurred_at": hit["occurred_at"], "confidence": conf, "category": category}
    if VG.enabled():
        out["priority"] = VG.is_priority(subj, category, content, actor, quote=quote)
        out["old"] = bool(fact.get("_old"))
        if reattributed:
            VG.REATTRIBUTED["us:reattributed"] += 1
            out["reattributed"] = True
    return out, ""


def gate_pass(facts: list[dict], chunk: list[dict]) -> list[tuple[dict | None, str]]:
    """一个片段的全部候选 → 等长的 [(清洗后的事 或 None, 丢弃原因)]。
    闸关：就是逐条 check，一字不差。闸开：逐条 check（旧抽取器的候选 _old=True 也走同一套）→ 去重 → 每片段 3 条封顶，
    高置信的大事（priority）可超到 5 条（value_gate.select_with_cap）。新提示词的候选排在旧抽取器的前面。"""
    import value_gate as VG
    res = [check(f, chunk) for f in facts]
    if not VG.enabled():
        return res
    idx = [i for i, (ok, _) in enumerate(res) if ok]
    order = sorted(idx, key=lambda i: (bool(facts[i].get("_old")), i))          # 新的在前、旧的在后
    sel = VG.select_with_cap([res[i][0] for i in order])
    out = list(res)
    for i, (ok, why) in zip(order, sel):
        out[i] = (ok, why) if ok else (None, why)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="UTC 日期或时间，含")
    ap.add_argument("--until", required=True, help="UTC 日期或时间，不含")
    ap.add_argument("--db")
    ap.add_argument("--jsonl", nargs="*", type=Path, help="直接读会话记录，不走 evidence 表")
    ap.add_argument("--app", type=Path, help="claude.ai 导出的 conversations.json")
    ap.add_argument("--names", nargs="*", default=[], help="--app 时只读这些窗口名")
    ap.add_argument("--no-timeline", action="store_true", help="不判新事顶不顶掉旧事")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-chunks", type=int, default=0)
    args = ap.parse_args()

    con = connect(args.db) if args.db else connect()
    if args.app:
        evs = load_app(args.app, args.names)
        evs = [e for e in evs if args.since <= e["occurred_at"] < args.until]
        evs.sort(key=lambda m: m["occurred_at"])
    elif args.jsonl:
        evs = [m for p in args.jsonl for m in load_jsonl(p, args.since, args.until)]
        evs.sort(key=lambda m: m["occurred_at"])
    else:
        evs = load_evidence(con, args.since, args.until)
    evs = prefilter_evs(evs)
    print(f"证据 {len(evs)} 条", file=sys.stderr)
    kept = dropped = dup = replaced = 0
    reasons: dict[str, int] = {}
    for n, chunk in enumerate(chunks(evs)):
        if args.max_chunks and n >= args.max_chunks:
            break
        try:
            raw = call_model(render(chunk))
        except Exception as e:                                   # noqa: BLE001
            print(f"块 {n} 调用失败：{e}", file=sys.stderr)
            continue
        for ok, why in gate_pass(raw, chunk):
            if not ok:
                dropped += 1
                key = why.split("：")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            hit = ok.pop("hit")
            hit2 = ok.pop("hit2", None)
            ok.pop("category", None)
            if args.dry_run:
                print(json.dumps(ok, ensure_ascii=False))
                kept += 1
                continue
            for h_, key_ in ((hit, "evidence_id"), (hit2, "evidence_id2")):
                if h_ is not None and ok.get(key_) is None:
                    loc = h_["line"] if isinstance(h_["line"], str) else f"L{h_['line']}"
                    ok[key_] = add_evidence(
                        con, source_kind="chat", source_actor=h_["source_actor"],
                        source_path=h_["path"], source_locator=loc,
                        content=h_["content"], occurred_at=h_["occurred_at"],
                        verification_status="verified")
                    record_tier(con, ok[key_], "primary", f"{h_['path']}:{loc}")
                    h_["id"] = ok[key_]
            ev_ids = [ok["evidence_id"]] + ([ok["evidence_id2"]] if ok.get("evidence_id2") else [])
            live_dup = con.execute(
                "SELECT 1 FROM facts WHERE subject=? AND content_sha256=?"
                " AND status IN ('candidate','confirmed') AND valid_to IS NULL",
                (ok["subject"], sha256(ok["content"]))).fetchone()
            target = None
            if not live_dup and not args.no_timeline and ok["subject"] == persona.USER:   # 只有她的事才会被标「可能过时」，别的不必问模型
                olds = timeline.candidates(con, ok["subject"], ok["content"], ok["occurred_at"])
                idx, why = timeline.judge(ok["content"], ok["occurred_at"], olds)
                if idx is not None:
                    target = (olds[idx], why.split("｜", 1)[0])
            if live_dup:
                fid = None
            else:
                fid = promote(con, subject=ok["subject"], fact_type=ok["type"],
                              content=ok["content"], evidence_ids=ev_ids,
                              valid_from=ok["occurred_at"], confidence=ok["confidence"])
            if fid is None:
                dup += 1
                continue
            if target:                               # 2026-10-29 降级：只标「可能过时」，不封旧事（见 stale.py）
                import stale
                new_ext = con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]
                if stale.mark(con, target[0]["ext_id"], new_ext, slot=target[1], judged_by="llm"):
                    replaced += 1
                    print(f"  可能过时：{target[0]['content'][:40]} ← {ok['content'][:40]}（{why}）",
                          file=sys.stderr)
            record_class(con, fid, ok["type"], ok["confidence"], EXTRACTOR)
            kept += 1
        if not args.dry_run:
            con.commit()
        time.sleep(0.3)
    print(f"留下 {kept}（其中标了「可能过时」{replaced}），重复 {dup}，丢弃 {dropped} {reasons}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
