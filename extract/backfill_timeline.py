#!/usr/bin/env python3
"""把已经抽好的事按时间顺序过一遍时间线：后来的事让先前的同槽事「可能过时」的，给先前那条打软标记。

2026-10-29 降级：以前这里会把旧的标 superseded 封掉；自动判新旧的准确率只有 29–59%，现在只在旁表
fact_stale_marks 记一行（stale.py），**不封旧事、不改状态、不删**。需要 ENGRAM_TIMELINE=1（负责判）
和 ENGRAM_INVALIDATE_CHAIN=1（负责写标记）两个开关都开，否则什么都不做。
只处理她的事实（主体是她）；新的必须是她本人原话（stale.check 硬卡）。

用法：
    python3 -m extract.backfill_timeline --dry-run [--db 路径] [--limit N]
    python3 -m extract.backfill_timeline          [--db 路径] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import persona  # noqa: E402
from store.store import connect  # noqa: E402
from extract import timeline  # noqa: E402


def run(con, *, dry_run: bool = False, limit: int = 0) -> dict:
    import flags
    import stale
    st = {"todo": 0, "judged": 0, "found": 0, "marked": 0}
    if not (flags.on("TIMELINE") and flags.on("INVALIDATE_CHAIN")):
        return st
    rows = con.execute(
        """SELECT f.id, f.ext_id, f.subject, f.content, f.valid_from FROM facts f
           WHERE f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')
             AND f.subject = ? AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL
           ORDER BY f.valid_from, f.id""", (persona.USER,)).fetchall()
    if limit:
        rows = rows[:limit]
    st["todo"] = len(rows)
    for r in rows:
        olds = [o for o in timeline.candidates(con, r["subject"], r["content"], r["valid_from"])
                if o["id"] != r["id"]]
        if not olds:
            continue
        st["judged"] += 1
        idx, why = timeline.judge(r["content"], r["valid_from"] or "", olds)
        if idx is None:
            continue
        st["found"] += 1
        if dry_run:
            continue
        if stale.mark(con, olds[idx]["ext_id"], r["ext_id"], slot=why.split("｜", 1)[0], reason=why[:200],
                      judged_by="llm"):
            st["marked"] += 1
            con.commit()
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    con = connect(args.db) if args.db else connect()
    st = run(con, dry_run=args.dry_run, limit=args.limit)
    print(f"待过 {st['todo']} 条，判了 {st['judged']}，判中 {st['found']}，写标记 {st['marked']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
