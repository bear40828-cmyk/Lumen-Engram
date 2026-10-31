#!/usr/bin/env python3
"""Lumen Engram · 派生层补齐 + 待补计数（任务三，2026-10-28）

新事实（切片蒸馏 / 自记）入库时只写 facts + fact_evidence + fact_origin；FTS 由触发器同步，
**分类（fact_class）、槽位（fact_key_map）、卡片（fact_cards）、向量（fact_vec）都是存完之后再补的**，
坏了可以重建。重要：这几层没补上之前，卡片/向量那两路搜不到新事实（FTS 那路立刻就能搜到）。
所以要有「修理工」和「待补计数」：

    python3 -m derive_repair --count     # 只数：每层还有多少没补（监控用）
    python3 -m derive_repair --apply     # 补 class / key / cards（本地，不连网）；向量由 vectors.py build 补

只补缺的、可重复跑；只动派生表，不碰 facts / evidence。fact_class.reason：蒸馏出来的是 'slice-v1'、
自记的是 'self-note-v1'——**都不以 extract-llm- 开头**，所以暂时不进「整理过的事」那一层的优先召回。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cards as C                                                  # noqa: E402
import keys as K                                                   # noqa: E402
from common import bigrams                                         # noqa: E402
from store.store import connect, record_class, record_key         # noqa: E402

REASON = {"distill": "slice-v1", "distill-cap": "slice-v1", "self-note": "self-note-v1"}
LIVE = "f.status IN ('candidate','confirmed') AND f.valid_to IS NULL"


def _cards_ready(con) -> bool:
    return bool(con.execute("SELECT 1 FROM sqlite_master WHERE name='fact_cards'").fetchone())


def _vec_ready(con) -> bool:
    return bool(con.execute("SELECT 1 FROM sqlite_master WHERE name='fact_vec'").fetchone())


def pending(con) -> dict:
    """每层还没补的条数。只数 fact_origin 登记过的（蒸馏 / 自记）事实的 class/key；卡片/向量数全库活着的。"""
    q = lambda s: con.execute(s).fetchone()[0]                      # noqa: E731
    out = {
        "class": q("SELECT COUNT(*) FROM fact_origin o JOIN facts f ON f.id=o.fact_id"
                   " WHERE f.id NOT IN (SELECT fact_id FROM fact_class)"),
        "cards": (q(f"SELECT COUNT(*) FROM facts f WHERE {LIVE} AND f.id NOT IN (SELECT fact_id FROM fact_cards)")
                  if _cards_ready(con) else None),
        "vec": (q(f"SELECT COUNT(*) FROM facts f WHERE {LIVE} AND f.id NOT IN (SELECT fact_id FROM fact_vec)")
                if _vec_ready(con) else None),
    }
    return out


def repair(con) -> dict:
    n_class = n_key = n_card = 0
    rows = con.execute(
        "SELECT f.id, f.subject, f.fact_type, f.content, f.confidence, o.origin FROM fact_origin o"
        " JOIN facts f ON f.id=o.fact_id WHERE f.id NOT IN (SELECT fact_id FROM fact_class)").fetchall()
    for r in rows:
        record_class(con, r["id"], r["fact_type"], r["confidence"] if r["confidence"] is not None else 0.5,
                     REASON.get(r["origin"], "slice-v1"))
        n_class += 1
        try:
            key, value, pol, elig = K.match(r["content"])
        except Exception:                                          # noqa: BLE001
            key = None
        if key:
            record_key(con, r["id"], key, value, pol, elig)
            n_key += 1
    if _cards_ready(con):
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        for r in con.execute(
                f"SELECT f.id, f.subject, f.content FROM facts f WHERE {LIVE}"
                " AND f.id NOT IN (SELECT fact_id FROM fact_cards)").fetchall():
            card = C.make_card(r["content"])
            con.execute("INSERT INTO fact_cards (fact_id, subject, card, card_grams, full_len, built_at)"
                        " VALUES (?,?,?,?,?,?)", (r["id"], r["subject"], card, bigrams(card), len(r["content"]), now))
            con.execute("INSERT INTO cards_fts (rowid, card, subject) VALUES (?,?,?)", (r["id"], card, r["subject"]))
            con.execute("INSERT INTO cards_gram_fts (rowid, card_grams) VALUES (?,?)", (r["id"], bigrams(card)))
            n_card += 1
    con.commit()
    return {"class": n_class, "key": n_key, "cards": n_card}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--count", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--db")
    args = ap.parse_args()
    con = connect(args.db) if args.db else connect()
    if args.apply:
        print("补齐：", repair(con))
    print("待补：", pending(con))
    return 0


if __name__ == "__main__":
    sys.exit(main())
