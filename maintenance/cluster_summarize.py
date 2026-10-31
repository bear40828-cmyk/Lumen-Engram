#!/usr/bin/env python3
"""Lumen Engram · 夜间向量聚类归纳（任务四·步骤3，2026-10-29，开关 ENGRAM_CLUSTER_SUMMARY，默认关）

取代「按 fact_key 巩固」那条路（9485 条活事实只有 179 条有 fact_key，巩固只产出 3 条）。
做法（用户和助手定的设计）：

  1. 取所有活事实的向量（fact_vec，1024 维），余弦相似度 ≥ 0.75、average linkage 聚簇。
     （numpy 实现：先按 ≥0.75 的相似对分连通块，再在块内做精确的 average linkage；不用 scipy。）
  2. 够格的簇：≥3 条、其中她本人原话 ≥2 条、本人原话来自 ≥2 个不同日子。
     「本人原话」＝主体是她、挂的支持证据全是她自己说的（走说话人修正层）。助理/别的 AI 的话不进提示词（防回声）。
  3. 让模型对她的原话写一句话归纳（没有共同结论就写 NONE）。
  4. 代码后检（任何一条不过就不写，记进 cluster_summary_log）：
       · 簇里有矛盾（fact_key 同槽异值/同值反向、「可能过时」标记过的一对、同话题一肯定一否定/一改口）→ 不写
       · 说了「多次/经常/总是…」但引用的原话撑不起（不同事件 < 2 或不同日子 < 2；强频率词要 ≥3 个事件）→ 拒绝
       · 引用的编号必须是她的原话（≥2 条）；归纳里的数字必须出现在被引用的原话里；长度 6–80 字
  5. 通过的写成一条**新的派生事实**：正文「〔归纳〕…」，subject=她，candidate，importance 0.5，
     用 derived_from 连到被引用的原话。**原事实一行不改**（confirmed/钉住的也只读不写）。
  6. 同一批成员只处理一次（cluster_summary_log.members_hash）；已有归纳覆盖了 ≥50% 成员的簇不重复写（outcome=covered）。

用法（默认只看，不写库、不调模型除非 --llm）：
    .venv/bin/python maintenance/cluster_summarize.py                 # 干看：有多少簇、多少够格（不调模型）
    .venv/bin/python maintenance/cluster_summarize.py --apply         # 调模型并写库（开关要开）
    选项：--db 路径  --max-calls 40（每晚模型调用上限）  --show（打印归纳正文；默认只打印数量）
定时（不在这里装，crontab 由助手加，见 TALK/部署清单）：maintenance/nightly-cluster.sh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

THRESHOLD = 0.75
MIN_SIZE = 3
MIN_OWN = 2
MIN_DAYS = 2
MAX_COMPONENT = 12000         # 单个连通块超过这个就跳过（矩阵 float32，12000 条≈580MB；防内存吃光）
MAX_CALLS = 40
BLOCK = 1024
LABEL = "〔归纳〕"
SUMMARY_IMPORTANCE = 0.5
SUMMARY_CONFIDENCE = 0.6
MIN_LEN, MAX_LEN = 6, 80
COVER_RATIO = 0.5

SECRET = re.compile(r"1[3-9]\d{9}|密码|password|passwd|token|密钥|api[_-]?key", re.I)
FREQ_ANY = re.compile(r"(多次|经常|常常|总是|一直|每次|每天|天天|屡次|反复|老是|频繁|不断|时常|总会|常年|一再)")
FREQ_STRONG = re.compile(r"(经常|常常|总是|一直|每次|每天|天天|频繁|老是|不断|时常|总会|常年)")
NEG = re.compile(r"(不|没|未|别|无|取消|放弃|算了)")
NEG_FALSE = re.compile(r"(不错|不过|不仅|不只|不管|不如|不好意思|没事|无聊|无论|无所谓)")
CHANGE = re.compile(r"(改成|改为|改考|换成|换了|改用|变成|更正|其实是|不是.{1,8}而是)")


def _np():
    try:
        import numpy as np
    except ImportError as e:                                 # pragma: no cover
        raise SystemExit("缺 numpy：请用 venv 的 python 跑（.venv/bin/python），见部署清单") from e
    return np


# ── 聚类（numpy） ───────────────────────────────────────────────────────────
def components(V, thr: float = THRESHOLD, block: int = BLOCK) -> list[list[int]]:
    """余弦（V 已单位化）≥ thr 的相似对的连通块。分块算，不一次性造 n×n。"""
    np = _np()
    n = len(V)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i0 in range(0, n, block):
        sims = V[i0:i0 + block] @ V.T
        ii, jj = np.nonzero(sims >= thr)
        for a, b in zip(ii.tolist(), jj.tolist()):
            a += i0
            if b > a:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values()]


def average_linkage(V, idx: list[int], thr: float = THRESHOLD) -> list[list[int]]:
    """在 idx 这一块里做精确的 average linkage（簇间两两余弦的平均 ≥ thr 才合并）。

    Lance-Williams 更新 + 每行最大值缓存：每步只在「最大值指向被合并的两个簇」的行上重算，
    其余行 O(1) 更新，所以到 thr 就停的话成本接近 O(m²)（m=块大小）。矩阵是 float32：m=9500 时约 360MB。"""
    np = _np()
    m = len(idx)
    if m == 1:
        return [list(idx)]
    X = V[idx].astype(np.float32)
    S = X @ X.T
    del X
    ninf = np.float32(-np.inf)
    np.fill_diagonal(S, ninf)
    size = np.ones(m, np.float32)
    active = np.ones(m, dtype=bool)
    members = [[i] for i in idx]
    rowmax = S.max(axis=1)
    rowarg = S.argmax(axis=1)
    while True:
        a = int(np.argmax(rowmax))
        if rowmax[a] < thr:
            break
        b = int(rowarg[a])
        na, nb = size[a], size[b]
        new = (na * S[a] + nb * S[b]) / (na + nb)             # inactive 行/列是 -inf，算出来仍是 -inf
        S[a, :] = new
        S[:, a] = new
        S[a, a] = ninf
        S[b, :] = ninf
        S[:, b] = ninf
        size[a] = na + nb
        active[b] = False
        rowmax[b] = ninf
        members[a] += members[b]
        members[b] = []
        rowmax[a] = S[a].max()
        rowarg[a] = int(S[a].argmax())
        stale_rows = np.nonzero(active & ((rowarg == a) | (rowarg == b)))[0]
        for i in stale_rows.tolist():
            if i != a:
                rowmax[i] = S[i].max()
                rowarg[i] = int(S[i].argmax())
        better = active & (new > rowmax)
        better[a] = False
        better[stale_rows] = False
        rowmax[better] = new[better]
        rowarg[better] = a
    return [mm for mm in members if mm]


def cluster(V, thr: float = THRESHOLD) -> list[list[int]]:
    """所有簇（含单条）。返回 V 的行下标列表。"""
    out: list[list[int]] = []
    for comp in components(V, thr):
        if len(comp) == 1:
            out.append(comp)
        elif len(comp) > MAX_COMPONENT:
            print(f"跳过一个过大的连通块（{len(comp)} 条 > {MAX_COMPONENT}）", file=sys.stderr)
        else:
            out.extend(average_linkage(V, comp, thr))
    return out


# ── 读库 ────────────────────────────────────────────────────────────────────
def load_vectors(con):
    """活事实（candidate/confirmed、valid_to 为空）的向量，排除归纳本身。返回 (fact_ids, 单位化矩阵)。"""
    np = _np()
    rows = con.execute(
        """SELECT v.fact_id, v.vec FROM fact_vec v JOIN facts f ON f.id = v.fact_id
           WHERE f.status IN ('candidate','confirmed') AND f.valid_to IS NULL
             AND f.content NOT LIKE ? ORDER BY v.fact_id""", (LABEL + "%",)).fetchall()
    if not rows:
        return [], np.zeros((0, 1), np.float32)
    ids = [r[0] for r in rows]
    mat = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1).copy()
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return ids, mat


def load_members(con, fact_ids: list[int]) -> list[dict]:
    """簇成员：正文、日期、说话人、事件（同源证据算一个事件）。按时间排。"""
    import echo
    out = []
    for fid in fact_ids:
        f = con.execute("SELECT id, ext_id, subject, content, valid_from, created_at, status FROM facts WHERE id=?",
                        (fid,)).fetchone()
        if f is None:
            continue
        ev = [r[0] for r in con.execute(
            "SELECT evidence_id FROM fact_evidence WHERE fact_id=? AND role='support'", (fid,))]
        actors = echo.evidence_actors(con, ev) if ev else set()
        roots = set()
        for e in ev:
            r = con.execute("SELECT origin_root FROM evidence_tier WHERE evidence_id=?", (e,)).fetchone()
            roots.add(r[0] if r and r[0] else f"ev{e}")
        out.append({"id": f["id"], "ext_id": f["ext_id"], "subject": f["subject"], "content": f["content"] or "",
                    "day": (f["valid_from"] or f["created_at"] or "")[:10], "status": f["status"],
                    "actors": actors, "events": roots or {f"fact{f['id']}"}})
    out.sort(key=lambda m: (m["day"], m["id"]))
    return out


def is_own(m: dict) -> bool:
    """她本人的原话：主体是她，支持证据的说话人只有她（防回声）；含密钥/手机号的不进提示词。"""
    return (m["subject"] == persona.USER and m["actors"] == {persona.USER}
            and not SECRET.search(m["content"]))


def eligible(members: list[dict]) -> tuple[bool, str]:
    own = [m for m in members if is_own(m)]
    if len(members) < MIN_SIZE:
        return False, "簇太小"
    if len(own) < MIN_OWN:
        return False, "本人原话不够"
    if len({m["day"] for m in own}) < MIN_DAYS:
        return False, "本人原话不在不同日子"
    return True, ""


def members_hash(own: list[dict]) -> str:
    return hashlib.sha256(",".join(str(i) for i in sorted(m["id"] for m in own)).encode()).hexdigest()


# ── 代码后检 ────────────────────────────────────────────────────────────────
def _bg(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _nums(s: str) -> set[str]:
    return set(re.findall(r"\d+", s or ""))


def find_contradiction(con, members: list[dict]) -> str | None:
    """簇里有没有互相矛盾的成员。有就返回原因（不含正文），没有 None。只含结构性的硬检查（fact_key 同槽异值／同值反向、已有「可能过时」标记）；词面类的见 lexical_hint。"""
    from keys import by_name
    ids = [m["id"] for m in members]
    mark = ",".join("?" * len(ids))
    # 1) fact_key：single 的槽两个值 / 同值一肯定一否定
    rows = con.execute(f"SELECT fact_id, fact_key, value, COALESCE(polarity,'pos') p FROM fact_key_map"
                       f" WHERE fact_id IN ({mark}) AND value IS NOT NULL", ids).fetchall()
    by_key: dict[str, set] = defaultdict(set)
    pols: dict[tuple, set] = defaultdict(set)
    for r in rows:
        spec = by_name(r["fact_key"])
        if spec is not None and spec.cardinality == "single":
            by_key[r["fact_key"]].add(r["value"])
        pols[(r["fact_key"], r["value"])].add(r["p"])
    if any(len(v) > 1 for v in by_key.values()):
        return "contradiction:fact_key 同槽异值"
    if any(len(p) > 1 for p in pols.values()):
        return "contradiction:fact_key 同值反向"
    # 2) 已被标过「可能过时」的一对
    try:
        hit = con.execute(f"SELECT 1 FROM fact_stale_marks WHERE old_fact_id IN ({mark}) AND new_fact_id IN ({mark})",
                          ids + ids).fetchone()
        if hit:
            return "contradiction:簇里有「可能过时」标记的一对"
    except Exception:                                        # noqa: BLE001
        pass
    return None


def lexical_hint(members: list[dict]) -> str | None:
    """词面上看着像「同话题一肯定一否定／一改口」的提示（**只是提示，不是裁决**）。

    2026-10-29 真模型实测：这条规则原来是硬拒绝（不问模型直接丢簇），43 个被拒的簇里抽查 20 个，
    约 18 个是误杀——「不/没」只是话里顺嘴带的字（「不知道吃什么」「没睡好」），并不是在否定同一件事。
    借鉴：MemoryScope 的 contra_repeat 和 Graphiti 的 resolve_edge 都是让模型判「矛盾还是重复」，
    词面只负责把可疑的挑出来；mem0 的 UPDATE/DELETE 也是模型决定。所以这里降级成给模型的提示，
    真矛盾由模型的 conflict 字段拦；fact_key 同槽异值、已有「可能过时」标记这种结构性的才留作硬拒绝。
    """
    own = [m for m in members if is_own(m)]
    for i, a in enumerate(own):
        for b in own[i + 1:]:
            ga, gb = _bg(a["content"]), _bg(b["content"])
            if not ga or not gb:
                continue
            jac = len(ga & gb) / len(ga | gb)
            if jac < 0.25:
                continue
            na = bool(NEG.search(NEG_FALSE.sub("", a["content"])))
            nb = bool(NEG.search(NEG_FALSE.sub("", b["content"])))
            if na != nb:
                return f"记录 [{a['id']}] 和 [{b['id']}] 说的像同一个话题，但一条带否定、一条没有，请先判断它们是不是真的在说反话"
            ca, cb = bool(CHANGE.search(a["content"])), bool(CHANGE.search(b["content"]))
            if (ca or cb) and _nums(a["content"]) != _nums(b["content"]):
                return f"记录 [{a['id']}] 和 [{b['id']}] 像同一个话题但数字不同、带改口的词，请先判断是不是前后改口了"
    return None


def check_frequency(summary: str, cited: list[dict]) -> str | None:
    """「多次/经常…」要有多个事件撑着；只靠一个事件（同源证据/同一天）的拒绝。"""
    if not FREQ_ANY.search(summary):
        return None
    events = set().union(*(m["events"] for m in cited)) if cited else set()
    days = {m["day"] for m in cited}
    need_events = 3 if FREQ_STRONG.search(summary) else 2
    if len(events) < need_events or len(days) < 2:
        return f"frequency:频率词要 ≥{need_events} 个事件、≥2 个日子，实际 {len(events)} 个事件、{len(days)} 天"
    return None


def check_summary(con, summary: str, cited_ids: list, members: list[dict]) -> tuple[bool, str]:
    """模型的归纳过不过代码后检。members 是整个簇（含非本人原话）。"""
    s = (summary or "").strip()
    if not s or s.upper() == "NONE":
        return False, "none"
    s = s[len(LABEL):] if s.startswith(LABEL) else s
    if "\n" in s or not (MIN_LEN <= len(s) <= MAX_LEN):
        return False, f"rejected:长度/格式（{len(s)} 字）"
    if SECRET.search(s):
        return False, "rejected:含密钥/手机号"
    if re.match(r"^[我咱]", s):
        return False, "rejected:第一人称（归纳要用第三人称写她的事）"
    own = {m["id"]: m for m in members if is_own(m)}
    try:
        cited = [own[int(i)] for i in dict.fromkeys(cited_ids or [])]
    except (KeyError, ValueError, TypeError):
        return False, "rejected:引用了不是她本人原话的编号"
    if len(cited) < MIN_OWN:
        return False, "rejected:引用的本人原话不足 2 条"
    if len({m["day"] for m in cited}) < MIN_DAYS:
        return False, "rejected:引用的原话不在不同日子"
    contradiction = find_contradiction(con, members)
    if contradiction:
        return False, "rejected:" + contradiction
    freq = check_frequency(s, cited)
    if freq:
        return False, "rejected:" + freq
    src = " ".join(m["content"] for m in cited)
    if not _nums(s) <= _nums(src):
        return False, "rejected:归纳里有原话没有的数字"
    return True, s


# ── 模型 ────────────────────────────────────────────────────────────────────
def build_prompt(own: list[dict], hint: str | None = None) -> str:
    who = persona.N_USER
    lines = "\n".join(f"[{m['id']}] {m['day']} {m['content'][:160]}" for m in own)
    return (f"下面是{who}本人自己说的话或她自己的事的几条记录，按时间排列。\n"
            "如果它们共同指向一个稳定的事实（偏好、状态、计划、关系、习惯），用一句话概括（60 字以内，"
            "只能用记录里出现过的信息，不许推测、不许加记录里没有的数字和日期；说「多次/经常」要真有多次）。\n"
            "写法：第三人称，用她的名字，不要用「我」；只写记录里明说的，不要补因果（不要写「因为…所以…」）；"
            "只出现过一次的事不要写成「习惯/总是」。\n"
            "如果只是同一个话题、但没有共同的结论，或者每条说的是不同的事，或者记录之间有矛盾，输出 NONE。\n"
            + (f"提示（词面规则看到的，不一定对）：{hint}。如果确实矛盾，conflict 填 true，summary 填 NONE。\n" if hint else "")
            + "只输出 JSON：{\"summary\":\"一句话或NONE\",\"ids\":[用到的记录编号],\"conflict\":是否发现记录互相矛盾(true/false)}\n\n"
            + lines)


def ask(prompt: str) -> str:
    """调模型（测试里 mock 这一个函数）。失败抛异常，由调用方当「这一簇跳过」。"""
    from llm import chat
    return chat([{"role": "system", "content": "你只输出合法 JSON。"}, {"role": "user", "content": prompt}],
                purpose="consolidate", max_tokens=300, timeout=60)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        return d if isinstance(d, dict) else {}
    except (ValueError, TypeError):
        return {}


# ── 主流程 ──────────────────────────────────────────────────────────────────
def _log(con, h, size, own_n, outcome, reason=None, fid=None, model=None):
    from store.store import now
    con.execute("INSERT OR IGNORE INTO cluster_summary_log (members_hash, size, own_count, outcome, reason,"
                " summary_fact_id, model, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (h, size, own_n, outcome, reason, fid, model, now()))


def _covered(con, own: list[dict]) -> bool:
    """已有归纳连着这批原话里 ≥50% → 不再写（旧归纳不删不改，也不重复）。"""
    ids = [m["id"] for m in own]
    mark = ",".join("?" * len(ids))
    rows = con.execute(
        f"""SELECT parent_fact_id, COUNT(*) c FROM fact_relations
            WHERE relation='derived_from' AND child_fact_id IN ({mark})
              AND parent_fact_id IN (SELECT summary_fact_id FROM cluster_summary_log WHERE outcome='summary')
            GROUP BY parent_fact_id""", ids).fetchall()
    return any(r[1] / len(ids) >= COVER_RATIO for r in rows)


def run(con, *, apply: bool = False, max_calls: int = MAX_CALLS, show: bool = False, thr: float = THRESHOLD) -> dict:
    """返回统计。apply=False：只聚类、数够格的，不调模型不写库。"""
    from store.store import promote, relate
    ids, V = load_vectors(con)
    stat = {"facts": len(ids), "clusters": 0, "size_ge3": 0, "eligible": 0, "calls": 0, "summary": 0,
            "none": 0, "rejected": 0, "covered": 0, "skipped_done": 0}
    if not ids:
        return stat
    clusters = cluster(V, thr)
    stat["clusters"] = len(clusters)
    todo = []
    for cl in clusters:
        if len(cl) < MIN_SIZE:
            continue
        stat["size_ge3"] += 1
        members = load_members(con, [ids[i] for i in cl])
        ok, _why = eligible(members)
        if not ok:
            continue
        own = [m for m in members if is_own(m)]
        h = members_hash(own)
        if con.execute("SELECT 1 FROM cluster_summary_log WHERE members_hash=?", (h,)).fetchone():
            stat["skipped_done"] += 1
            continue
        stat["eligible"] += 1
        todo.append((h, members, own))
        contradiction = find_contradiction(con, members)      # 簇里有矛盾：不用问模型了
        if contradiction:
            todo.pop()
            stat["eligible"] -= 1
            stat["rejected"] += 1
            if apply:
                _log(con, h, len(members), len(own), "rejected", contradiction.split(":", 1)[-1][:120])
                con.commit()
    if not apply:
        return stat

    for h, members, own in todo:
        if _covered(con, own):
            _log(con, h, len(members), len(own), "covered")
            con.commit()
            stat["covered"] += 1
            continue
        if stat["calls"] >= max_calls:
            break                                   # 超过每晚上限：剩下的明晚接着做（没记 log，不会丢）
        stat["calls"] += 1
        try:
            got = parse(ask(build_prompt(own, lexical_hint(members))))
        except Exception:                           # noqa: BLE001  模型失败＝这一簇今晚跳过，不记 log，明晚再试
            continue
        if got.get("conflict") is True:
            _log(con, h, len(members), len(own), "rejected", "model-flagged-conflict")
            con.commit()
            stat["rejected"] += 1
            continue
        ok, res = check_summary(con, str(got.get("summary") or ""), got.get("ids") or [], members)
        if not ok:
            outcome = "none" if res == "none" else "rejected"
            _log(con, h, len(members), len(own), outcome, None if outcome == "none" else res.split(":", 1)[-1][:120])
            con.commit()
            stat[outcome] += 1
            continue
        cited = [m for m in own if m["id"] in {int(i) for i in got["ids"]}]
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        try:
            fid = promote(con, subject=persona.USER, fact_type="semantic", content=LABEL + res,
                          valid_from=min(m["day"] for m in cited), importance=SUMMARY_IMPORTANCE,
                          confidence=SUMMARY_CONFIDENCE)
            if fid is None:
                _log(con, h, len(members), len(own), "rejected", "duplicate")
                stat["rejected"] += 1
            else:
                for m in cited:
                    relate(con, fid, m["id"], "derived_from")
                _log(con, h, len(members), len(own), "summary", None, fid)
                stat["summary"] += 1
                if show:
                    print(f"  {LABEL}{res}  ← {len(cited)} 条")
            con.commit()
        except BaseException:
            con.rollback()
            raise
    return stat


def main(argv=None) -> int:
    import flags
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--allow-inherit-model", action="store_true",
                    help="没配 CLUSTER_MODEL 也照跑（会退回 EXTRACT_MODEL，通常是更贵的强模型）")
    args = ap.parse_args(argv)
    if not flags.on("CLUSTER_SUMMARY"):
        print("聚类归纳已关闭（ENGRAM_CLUSTER_SUMMARY=1 才运行；见 flags.py），未做任何事。")
        return 0
    if args.apply and not args.allow_inherit_model:
        import llm
        if not llm._get(llm.load_env(), "CLUSTER_MODEL"):
            # 归纳写进事实库；试用里 Flash 的错误更少、也便宜得多。没单独配就会悄悄继承 EXTRACT_MODEL（Pro），所以拒绝。
            print("没配 CLUSTER_MODEL（会继承 EXTRACT_MODEL）；夜间归纳要显式指定模型（建议 Flash 档）。"
                  "在 .env 加 CLUSTER_MODEL=…，或明知故犯加 --allow-inherit-model。未做任何事。")
            return 2
    from store.store import connect
    con = connect(args.db) if args.db else connect()
    st = run(con, apply=args.apply, max_calls=args.max_calls, show=args.show)
    print(("写库：" if args.apply else "干看（没调模型、没写库）：") + json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
