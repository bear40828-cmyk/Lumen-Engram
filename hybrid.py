#!/usr/bin/env python3
"""Lumen Engram · 混合召回（任务二·步骤3，2026-10-28）

替换旧的「六类分层路由 + 逐层升级」（retrieve.py 里的 L1/CUES/order/升级循环）。
旧办法先猜这句话属于哪类问题、再一层层往下翻——猜错了就翻错屋子，
翻对了还要靠 metamem 判够不够（步骤2已摘）。现在改成：

    FTS 关键词候选（所有类型一起查；抽取器整理过的「事」另算一路）
  + 向量候选（vectors.py，按意思找）
  → 倒数排名融合（RRF）合并
  → 交给 rerank（hook 里那道模型精排）挑。

「要不要查」仍然是 router 管（含软话豁免、稀有词唤醒、工作记忆唤醒，都在 retrieve.retrieve 上半段，没动）。
这里只管「查哪些」。

保留的旧规矩：
  · 只要 candidate + confirmed、valid_to 为空（store.search / vectors 已过滤；这里对向量路再补一道 disown 过滤）
  · confirmed / 钉住的优先：RRF 上加固定加分（钉 +PIN_BONUS，confirmed +CONFIRM_BONUS）
  · 防回声（task2-08）：主体是人、证据全来自别人（只有转述）、且未 confirmed/钉住的，最终分 × ECHO_WEIGHT(1/3)，
    并在 hit 上带 echo_from，下游（hook / MCP）据此标〔转述·来自某某〕；confirmed / 钉住的豁免
  · 冷层降权：最终分 × salience.COLD_WEIGHT（补丁02）
  · 最稀有的词找回来的排前面（FTS 路按 term_rank 排，沿用 2026-08-18 的教训）
  · 工作记忆「手上的事」相关的小幅加分，只动顺序不动存在
  · 向量失败/没配 → 空，照旧走 FTS（vectors.search 本来就吞所有异常）
  · 注入上限（hook 里 MAX_HITS=2）不在这里管，在 hook

旧路径还在：ENGRAM_LAYERED_RETRIEVE=1 回滚（见 flags.py）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from echo import ECHO_WEIGHT, echo_sources  # noqa: E402
from salience import COLD_WEIGHT  # noqa: E402
from stale import STALE_WEIGHT  # noqa: E402
from store.store import search    # noqa: E402

RRF_K = 60
PIN_BONUS = 0.02          # ≈ 把一条排在第 1 的分（1/61≈0.0164）再抬高一截：钉住的几乎总能进前几
CONFIRM_BONUS = 0.012
GOAL_BONUS = 0.03         # × 工作记忆相关度(0~1)
FTS_PER_TERM = 8
FTS_PER_TERM_LIGHT = 3
VEC_K = 8
VEC_K_LIGHT = 3
VEC_FLOOR = 0.55
SUBJECT_BONUS = 0.01      # 问的是谁的事，那个人的事（和 us）小幅加分；比钉住（0.02）轻，只动顺序
SUMMARY_LABEL = "〔归纳〕"   # 同 maintenance/cluster_summarize.LABEL
SUMMARY_WEIGHT = 0.7
POOL_CAP = 12
POOL_CAP_LIGHT = 6


def _vsearch(con, text: str, k: int, floor: float) -> list[dict]:   # 单独一层，测试时 mock
    try:
        from vectors import search as vs
        return vs(con, text, k=k, floor=floor)
    except Exception:                                        # noqa: BLE001
        return []


def _fts_any(con, term: str, limit: int, *, extracted_only: bool = False) -> list[dict]:
    """查一个词（不分记忆类型）；长词整块没中就拆窗口再试，跟旧 probe 一样。"""
    import retrieve as R
    kw = {"extracted_only": True} if extracted_only else {}
    hits = search(con, term, limit=limit, **kw)
    if hits or len(term) <= 3:
        return hits
    out, seen = [], set()
    for t, _n in R.usable_forms(con, term):
        for h in search(con, t, limit=limit, **kw):
            if h["ext_id"] not in seen:
                seen.add(h["ext_id"])
                out.append(h)
    return out[: limit * 2]


def _disowned(con, ext_ids: list[str]) -> set[str]:
    if not ext_ids:
        return set()
    q = ",".join("?" * len(ext_ids))
    try:
        return {r[0] for r in con.execute(
            f"""SELECT f.ext_id FROM facts f JOIN current_stances cs ON cs.fact_id = f.id
                WHERE f.ext_id IN ({q}) AND cs.actor = 'claude' AND cs.stance = 'disown'""",
            ext_ids)}
    except Exception:                                        # noqa: BLE001
        return set()


def rrf_merge(lists: dict[str, list[dict]]) -> dict[str, dict]:
    """多路排好序的候选 → {ext_id: hit（带 rrf、via）}。"""
    merged: dict[str, dict] = {}
    for via, hits in lists.items():
        for rank, h in enumerate(hits, 1):
            m = merged.setdefault(h["ext_id"], dict(h) | {"rrf": 0.0, "via": []})
            m["rrf"] += 1.0 / (RRF_K + rank)
            m["via"].append(via)
            # 优先保留带激活度分数的那份记录（FTS 路的 score 是激活度，向量路的是余弦）
            if via != "vector" and "act_score" not in m:
                m["act_score"] = float(h.get("score") or 0)
    return merged


def finalize(merged: dict[str, dict], *, goal_rel: float = 0.0, goal_overlap=None,
             cap: int = POOL_CAP, echo: dict | None = None, stale: dict | None = None,
             who_hint: str | None = None) -> list[dict]:
    out = []
    for h in merged.values():
        s = h["rrf"]
        if h.get("user_pinned"):
            s += PIN_BONUS
        elif h.get("status") == "confirmed":
            s += CONFIRM_BONUS
        if goal_overlap and goal_rel:
            gb = goal_overlap(h.get("content") or "")
            h["goal_boost"] = round(GOAL_BONUS * gb, 4)
            s += h["goal_boost"]
        if who_hint and h.get("subject") in (who_hint, "us"):
            s += SUBJECT_BONUS                       # 按上下文推出这句话问的是谁的事（CONTEXT_ROUTE）：软加分，不过滤
            h["who_match"] = True
        src = (echo or {}).get(h["ext_id"])
        h["echo_from"] = list(src) if src else []
        if src:
            s *= ECHO_WEIGHT                       # 只有转述：降权（confirmed/钉住已在 echo_sources 里豁免）
        if h.get("layer") == "cold":
            s *= COLD_WEIGHT
        newer = (stale or {}).get(h["ext_id"])
        h["stale_by"] = newer["ext_id"] if newer else None      # 「可能过时」软标记（stale.py，开关 INVALIDATE_CHAIN）
        h["stale_new"] = newer
        if (h.get("content") or "").startswith(SUMMARY_LABEL):
            s *= SUMMARY_WEIGHT                      # 夜间聚类归纳（〔归纳〕）排在原话后面
        if newer:
            s *= STALE_WEIGHT                        # 只排后一点，不丢；confirmed/钉住读标记时已排除
        h["hybrid_score"] = round(s, 5)
        h["score"] = h["hybrid_score"]          # 下游（遥测/日志）读 score
        h["via"] = "+".join(h["via"])
        out.append(h)
    out.sort(key=lambda h: (-h["hybrid_score"], -float(h.get("act_score") or 0)))
    return out[:cap]


def run(con, text: str, res, *, terms: list[str], light: bool, goal_rel: float,
        goal_why: str, who_hint: str | None = None):
    """retrieve.retrieve 的混合分支。res 是上半段已经填好 router 信息的 Result。"""
    import retrieve as R
    import working

    per = FTS_PER_TERM_LIGHT if light else FTS_PER_TERM
    vk = VEC_K_LIGHT if light else VEC_K
    cap = POOL_CAP_LIGHT if light else POOL_CAP

    fts, dist, seen_f, seen_d = [], [], set(), set()
    for rank, t in enumerate(terms):
        for h in _fts_any(con, t, per):
            if h["ext_id"] not in seen_f:
                seen_f.add(h["ext_id"])
                fts.append(dict(h, term_rank=rank))
        for h in _fts_any(con, t, per, extracted_only=True):
            if h["ext_id"] not in seen_d:
                seen_d.add(h["ext_id"])
                dist.append(dict(h, term_rank=rank))
    fts.sort(key=lambda h: (h["term_rank"], -float(h.get("score") or 0)))
    dist.sort(key=lambda h: (h["term_rank"], -float(h.get("score") or 0)))

    vec = [dict(h) for h in _vsearch(con, text, vk, VEC_FLOOR)]
    bad = _disowned(con, [h["ext_id"] for h in vec])
    vec = [h for h in vec if h["ext_id"] not in bad]

    merged = rrf_merge({"fts": fts, "distilled": dist, "vector": vec})
    anchor = goal_why.split("「", 1)[-1].rstrip("」") if goal_why else ""
    overlap = (lambda c: working._overlap(c, anchor)) if anchor else None
    try:
        echo = echo_sources(con, list(merged))     # 转述信号：失败就当没有，不影响召回
    except Exception:                              # noqa: BLE001
        echo = {}
    try:
        import stale
        stale_marks = stale.marks_for(con, list(merged))     # 开关没开/没表 → 空，不影响召回
    except Exception:                              # noqa: BLE001
        stale_marks = {}
    res.hits = finalize(merged, goal_rel=goal_rel if goal_rel >= R.GOAL_STRONG else 0.0,
                        goal_overlap=overlap, cap=cap, echo=echo, stale=stale_marks, who_hint=who_hint)
    try:                                           # 自记的标记（渲染时加〔助手自记〕；不影响排序）
        __import__("echo").annotate_origin(con, res.hits)
    except Exception:                              # noqa: BLE001
        pass
    res.mode = "hybrid"
    res.steps.append(R.Step("hybrid", len(res.hits), 0.0, False, False,
                            f"FTS {len(fts)} 条＋整理过的事 {len(dist)} 条＋向量 {len(vec)} 条，"
                            f"合并后取 {len(res.hits)} 条"
                            + (f"（其中只有转述的 {sum(1 for h in res.hits if h.get('echo_from'))} 条已降权）"
                               if any(h.get("echo_from") for h in res.hits) else "")))
    return res
