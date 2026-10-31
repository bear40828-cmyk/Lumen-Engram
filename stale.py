#!/usr/bin/env python3
"""Lumen Engram · 「可能过时」软标记（任务四·步骤2，2026-10-29，开关 ENGRAM_INVALIDATE_CHAIN，默认关）

背景：入库时让模型判「哪条新事取代哪条旧事」，同槽取代的准确率只有 29–59%（试验记录见 TALK），
错一次就封掉一条真事实。助手审后改成降级方案：**只标，不封，不删**。

做了什么：较新的一条（必须是她本人的原话）和较旧的一条是同一个槽 → 在旁表 fact_stale_marks 记一行。
没做什么：不改 facts 任何一行（不封 valid_to、不改 status）、不删、不碰 confirmed / 钉住。
效果只有两个：召回排序 × STALE_WEIGHT（排后一点，仍在候选里）；注入时标「可能已过时」并把较新的那条附在后面。

代码硬卡（不靠模型自觉，mark() 里全部检查，任何一条不过就不写）：
  · 新旧同一个主体，而且主体是她（persona.USER）
  · 较新的这条必须有她本人说话的支持证据（echo.evidence_actors 里有她；只有助理/别的 AI 的话不行）
  · 旧的必须是还活着的 candidate、没钉住；新的是还活着的 candidate / confirmed
  · 旧的发生时间必须早于新的（valid_from，缺失用 created_at）
  · 同槽：两条都有 fact_key 时 key 必须相同且值或肯定否定不同；否则调用方必须给出非空的 slot 描述
    （模型回答的「新旧共同回答的是什么问题」）。两条 key 不同 → 拒绝。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona  # noqa: E402

STALE_WEIGHT = 0.5      # 召回排序倍率：只排后，不丢


def enabled() -> bool:
    import flags
    return flags.on("INVALIDATE_CHAIN")


def _row(con, ext_id: str):
    return con.execute(
        "SELECT f.*, COALESCE(f.valid_from, f.created_at) AS _when, m.fact_key AS _key, m.value AS _val,"
        " m.polarity AS _pol FROM facts f LEFT JOIN fact_key_map m ON m.fact_id = f.id WHERE f.ext_id=?",
        (ext_id,)).fetchone()


def check(con, old_ext: str, new_ext: str, slot: str | None = None) -> tuple[bool, str]:
    """代码闸：能不能把 old 标成「可能被 new 顶替而过时」。只读。"""
    old, new = _row(con, old_ext), _row(con, new_ext)
    if old is None or new is None:
        return False, "找不到事实"
    if old["id"] == new["id"]:
        return False, "新旧是同一条"
    if old["subject"] != new["subject"]:
        return False, "新旧主体不同"
    if new["subject"] != persona.USER:
        return False, "只对她的事实标过时"
    if old["status"] != "candidate" or old["user_pinned"] or old["valid_to"]:
        return False, "旧的不是活着的 candidate（confirmed / 钉住 / 已作废的一律不碰）"
    if new["status"] not in ("candidate", "confirmed") or new["valid_to"]:
        return False, "新的不是活着的事实"
    if not (old["_when"] and new["_when"] and old["_when"] < new["_when"]):
        return False, "新的不比旧的晚"
    import echo
    ids = [r[0] for r in con.execute(
        "SELECT evidence_id FROM fact_evidence WHERE fact_id=? AND role='support'", (new["id"],))]
    if persona.USER not in echo.evidence_actors(con, ids):
        return False, "较新的这条不是她本人的原话（没有她说话的支持证据）"
    if old["_key"] and new["_key"]:
        if old["_key"] != new["_key"]:
            return False, "fact_key 不同，不是同一个槽"
        if (old["_val"], old["_pol"]) == (new["_val"], new["_pol"]):
            return False, "同槽但值一样，没变"
    elif not (slot or "").strip():
        return False, "说不出同一个槽（没有 fact_key 也没有共同问题）"
    return True, ""


def mark(con, old_ext: str, new_ext: str, *, slot: str | None = None, reason: str = "",
         judged_by: str = "llm") -> int | None:
    """写一行「可能过时」。开关没开、闸不过都返回 None（什么都不写）。不 commit（跟着调用方的事务）。"""
    if not enabled():
        return None
    ok, _why = check(con, old_ext, new_ext, slot)
    if not ok:
        return None
    import store.store as S
    old, new = _row(con, old_ext), _row(con, new_ext)
    cur = con.execute(
        "INSERT OR IGNORE INTO fact_stale_marks (old_fact_id, new_fact_id, slot, reason, judged_by, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (old["id"], new["id"], (slot or old["_key"] or "")[:120], (reason or "")[:200], judged_by, S.now()))
    return cur.lastrowid if cur.rowcount else None


def marks_for(con, ext_ids) -> dict[str, dict]:
    """{旧事 ext_id: {ext_id, content, valid_from, slot}}（较新那条；一条旧事被多条标过时只取最新的一条）。
    开关没开 / 没有表 / 出错都当没有。读时再验一遍：旧的若后来被 confirm / 钉住，标记失效。"""
    ids = [i for i in ext_ids or () if i]
    if not ids or not enabled():
        return {}
    try:
        q = ",".join("?" * len(ids))
        rows = con.execute(
            f"""SELECT o.ext_id AS old_ext, n.ext_id AS ext_id, n.content AS content,
                       n.valid_from AS valid_from, s.slot AS slot, COALESCE(n.valid_from, n.created_at) AS w
                FROM fact_stale_marks s
                JOIN facts o ON o.id = s.old_fact_id
                JOIN facts n ON n.id = s.new_fact_id
                WHERE o.ext_id IN ({q}) AND o.status = 'candidate' AND o.user_pinned = 0
                  AND o.valid_to IS NULL AND n.valid_to IS NULL
                  AND n.status IN ('candidate','confirmed')
                ORDER BY w""", ids).fetchall()
    except Exception:                                        # noqa: BLE001
        return {}
    out: dict[str, dict] = {}
    for r in rows:                                           # 按时间升序，后面的覆盖前面的 → 留最新
        out[r["old_ext"]] = {"ext_id": r["ext_id"], "content": r["content"],
                             "valid_from": r["valid_from"], "slot": r["slot"]}
    return out


def annotate(con, hits: list[dict]) -> dict[str, dict]:
    """给召回命中就地写 stale_by（较新那条的 ext_id）和 stale_new；返回 {旧 ext_id: 较新记录}。"""
    m = marks_for(con, [h.get("ext_id") for h in hits])
    for h in hits:
        n = m.get(h.get("ext_id"))
        h["stale_by"] = n["ext_id"] if n else None
        h["stale_new"] = n
    return m
