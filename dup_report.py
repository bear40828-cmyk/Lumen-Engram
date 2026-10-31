#!/usr/bin/env python3
"""Lumen Engram · 近重复报告（只读，每周跑一次看看；任务三，2026-10-28）

只出报告，**不写库、不收起、不合并**（用户 7A：先看两周再决定要不要真合并）。
对蒸馏 / 自记出来的 candidate，同 subject 内 bigram Jaccard ≥ 0.7 聚簇，只列 id，不抄原文。
    python3 dup_report.py [--db PATH] [--days 14]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store.store import connect                                    # noqa: E402


def _b(s):
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def clusters(con, days=14, thr=0.7):
    rows = con.execute(
        "SELECT f.id, f.subject, f.content, f.status FROM facts f JOIN fact_origin o ON o.fact_id=f.id"
        " WHERE f.status='candidate' AND f.user_pinned=0 AND f.valid_to IS NULL"
        " AND f.created_at >= datetime('now', ?) ORDER BY f.id", (f"-{days} days",)).fetchall()
    seen, out = set(), []
    bs = {r["id"]: _b(r["content"]) for r in rows}
    for i, a in enumerate(rows):
        if a["id"] in seen:
            continue
        grp = [a["id"]]
        for b in rows[i + 1:]:
            if b["id"] in seen or b["subject"] != a["subject"]:
                continue
            x, y = bs[a["id"]], bs[b["id"]]
            if x and y and len(x & y) / len(x | y) >= thr:
                grp.append(b["id"])
        if len(grp) > 1:
            seen.update(grp)
            out.append((a["subject"], grp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    ap.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    con = connect(a.db) if a.db else connect()
    cl = clusters(con, a.days)
    print(f"近重复报告（只读）：{len(cl)} 簇，共 {sum(len(g) for _, g in cl)} 条")
    for subj, g in cl:
        print(f"  [{subj}] {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
