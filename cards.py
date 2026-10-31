#!/usr/bin/env python3
"""Lumen Engram · 薄发现层：先看卡片，命中才读全文

为什么要这层：库里有的事实一条就六百字（我自己那些长回复）。直接搜全文库，
一次召回能吐几千字进上下文——省下来的时间全赔在 token 上了。

所以照 Table-GitHub-Capability-Router 的双层结构做：
  1. discover()  在薄表里找候选，每条只有 40 字的卡片
  2. expand()    只对真正要用的那几条读全文

fact_cards 是**派生层**，删了重建即可，schema.sql v0.1 一个字没改。

    python3 cards.py --build        # 建/重建卡片
    python3 cards.py 同事           # 只看发现结果（不展开）
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from common import bigrams          # noqa: E402
from salience import COLD_WEIGHT     # noqa: E402
from store.store import connect     # noqa: E402

CARD_LEN = 40
RE_HEAD = re.compile(persona.fill(r"^(@@alt:USER,CLAUDE@@|我|@@alt:GPT,C@@)(（[^）]*）)?[：:]\s*"))
RE_SPACE = re.compile(r"\s+")


def make_card(content: str) -> str:
    """把一条事实压成一行。留住谁说的和头一句，其余截掉。"""
    body = RE_HEAD.sub("", content)
    body = RE_SPACE.sub(" ", body).strip()
    # 优先在第一个句读处断开，读起来是一句完整的话
    for sep in ("。", "！", "？", "；", "\n"):
        idx = body.find(sep, 8)
        if 8 <= idx <= CARD_LEN:
            body = body[:idx]
            break
    if len(body) > CARD_LEN:
        body = body[:CARD_LEN] + "…"
    who = RE_HEAD.match(content)
    return f"{who.group(1)}：{body}" if who else body


def build(con, *, quiet: bool = False) -> dict:
    con.executescript((ROOT / "cards.sql").read_text(encoding="utf-8"))
    con.execute("DELETE FROM fact_cards")
    con.execute("INSERT INTO cards_fts (cards_fts) VALUES ('delete-all')")
    con.execute("INSERT INTO cards_gram_fts (cards_gram_fts) VALUES ('delete-all')")

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = con.execute(
        "SELECT id, subject, content FROM facts WHERE status IN ('candidate','confirmed')"
    ).fetchall()

    total_full = total_card = 0
    for r in rows:
        card = make_card(r["content"])
        total_full += len(r["content"])
        total_card += len(card)
        con.execute(
            "INSERT INTO fact_cards (fact_id, subject, card, card_grams, full_len, built_at)"
            " VALUES (?,?,?,?,?,?)",
            (r["id"], r["subject"], card, bigrams(card), len(r["content"]), now))
        con.execute("INSERT INTO cards_fts (rowid, card, subject) VALUES (?,?,?)",
                    (r["id"], card, r["subject"]))
        con.execute("INSERT INTO cards_gram_fts (rowid, card_grams) VALUES (?,?)",
                    (r["id"], bigrams(card)))
    con.commit()

    out = {"卡片": len(rows), "全文总字数": total_full, "卡片总字数": total_card,
           "压缩到": f"{total_card / max(total_full, 1):.1%}"}
    if not quiet:
        print("、".join(f"{k} {v}" for k, v in out.items()))
    return out


HIT_WIDTH = 60          # 单个命中窗口的宽度
MAX_WINDOWS = 2         # 一条事实最多取两个命中窗口
MERGE_GAP = 24          # 两个命中挨得比这还近就并成一个窗口
HIT_BUDGET = 160        # 命中片段总字数上限


def _windows(body: str, positions: list[int], qlen: int) -> list[tuple[int, int]]:
    """把命中位置变成窗口区间，近的合并。"""
    half = max(HIT_WIDTH // 2 - qlen // 2, 8)
    spans = [(max(0, p - half), min(len(body), p + qlen + half)) for p in positions]
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def snippet_around(content: str, query: str, width: int = HIT_WIDTH) -> str | None:
    """按查询词在全文里开窗，取命中的那些段。

    GPT 第三轮第四颗螺丝：不要永远只取第一处。长文本里同一个词可能前后各命中一次，
    而两处讲的是不同的事。所以最多取两个窗口，近的合并、远的用 … 拼接，总长封顶。

    为什么自己写而不用 FTS5 的 snippet()：实测 trigram 索引上它返回空串，
    bigram 那张表返回的是切好的 gram 串（「粉 粉色 色狗」），人读不了。
    """
    body = RE_SPACE.sub(" ", content)
    needle = query
    if body.find(needle) < 0:                     # 两字词可能跨空白，退一格再找
        for i in range(len(query) - 1, 1, -1):
            if body.find(query[:i]) >= 0:
                needle = query[:i]
                break
    positions: list[int] = []
    start = body.find(needle)
    while start >= 0 and len(positions) < MAX_WINDOWS:
        positions.append(start)
        start = body.find(needle, start + len(needle) + HIT_WIDTH // 2)
    if not positions:
        return None

    parts, used = [], 0
    spans = _windows(body, positions, len(needle))
    for i, (a, b) in enumerate(spans):
        frag = body[a:b].strip()
        if used + len(frag) > HIT_BUDGET:
            frag = frag[:max(0, HIT_BUDGET - used)]
        if not frag:
            break
        used += len(frag)
        lead = "…" if a > 0 else ""
        tail = "…" if b < len(body) else ""
        parts.append(f"{lead}{frag}{tail}")
        if used >= HIT_BUDGET:
            break
    return " ".join(parts)


def discover(con, query: str, *, subject=None, limit=5):
    """第一层：只在薄表里找。卡片是静态摘要，hit 是按这次查询开窗的命中片段。"""
    q = query.strip()
    if len(q) < 2:
        return []
    if len(q) == 2:
        table, match = "cards_gram_fts", f'"{q}"'
    else:
        table, match = "cards_fts", q

    sql = f"""
      SELECT c.fact_id, f.ext_id, c.subject, c.card, c.full_len, f.layer, f.fact_type,
             engram_activation(f.importance, f.confidence, f.emotion, f.access_count,
                               julianday('now') - julianday(COALESCE(f.valid_from, f.created_at)),
                               f.user_pinned, f.fact_type)
               * CASE WHEN f.layer = 'cold' THEN {COLD_WEIGHT} ELSE 1.0 END AS score
      FROM {table} t
      JOIN fact_cards c ON c.fact_id = t.rowid
      JOIN facts f ON f.id = c.fact_id
      LEFT JOIN current_stances cs ON cs.fact_id = f.id AND cs.actor = 'claude'
      WHERE {table} MATCH ?
        AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL
        AND COALESCE(cs.stance,'none') <> 'disown'
    """
    args: list = [match]
    if subject:
        sql += " AND c.subject = ?"
        args.append(subject)
    sql += " ORDER BY score DESC LIMIT ?"
    args.append(limit * 3)          # 多取一些：只有转述的要降权重排，再切回 limit

    rows = [dict(r) for r in con.execute(sql, args)]
    try:                            # 防回声（task2-08）：只有转述的 ×1/3，并带 echo_from 给渲染层标注
        import echo
        src = echo.echo_sources(con, [r["ext_id"] for r in rows])
        for row in rows:
            row["echo_from"] = src.get(row["ext_id"], [])
            if row["echo_from"]:
                row["score"] *= echo.ECHO_WEIGHT
        rows.sort(key=lambda r: -r["score"])
    except Exception:               # noqa: BLE001
        for row in rows:
            row.setdefault("echo_from", [])
    try:                            # 自记的标记：旁表里查，显示时才加〔助手自记〕
        _sn = __import__("echo").self_note_ids(con, [r["ext_id"] for r in rows])
        for row in rows:
            row["self_note"] = row["ext_id"] in _sn
    except Exception:               # noqa: BLE001
        pass
    out = []
    for row in rows[:limit]:
        full = con.execute("SELECT content FROM facts WHERE id=?",
                           (row["fact_id"],)).fetchone()["content"]
        row["hit"] = snippet_around(full, q)
        row["show"] = row["hit"] or row["card"]     # 有命中片段就用它，否则退回静态卡片
        out.append(row)
    return out


DIRECT_READ_BUDGET = 400        # 命中全文加起来还不到这么多字，卡片层反而是累赘


def cheap_enough(hits, budget: int = DIRECT_READ_BUDGET) -> bool:
    """命中的全文本来就短，就别绕卡片层了——那样反而多花字。"""
    return sum(h["full_len"] for h in hits) <= budget


def expand(con, fact_ids):
    """第二层：只对选中的这几条读全文。"""
    if not fact_ids:
        return []
    marks = ",".join("?" * len(fact_ids))
    rows = [dict(r) for r in con.execute(
        f"SELECT ext_id, subject, content, status, user_pinned, layer"
        f" FROM facts WHERE id IN ({marks})", list(fact_ids))]
    try:                            # 防回声：读全文时也带上「只有转述」的标注来源
        import echo
        src = echo.echo_sources(con, [r["ext_id"] for r in rows])
        for r in rows:
            r["echo_from"] = src.get(r["ext_id"], [])
    except Exception:               # noqa: BLE001
        for r in rows:
            r.setdefault("echo_from", [])
    try:
        _sn = __import__("echo").self_note_ids(con, [r["ext_id"] for r in rows])
        for r in rows:
            r["self_note"] = r["ext_id"] in _sn
    except Exception:               # noqa: BLE001
        pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-n", "--limit", type=int, default=5)
    args = ap.parse_args()

    con = connect()
    if args.build:
        build(con)
        return 0
    if not args.query:
        ap.print_help()
        return 1

    hits = discover(con, args.query, limit=args.limit)
    if not hits:
        print("薄表里没找到。")
        return 0
    for i, h in enumerate(hits, 1):
        tag = "命中" if h["hit"] else "卡片"
        et = __import__("echo").tag(h)
        print(f"{i}. [{h['score']:.3f}]（{tag}）{et}{h['show']}    （全文 {h['full_len']} 字）")
    print(f"\n发现层一共给了 {sum(len(h['show']) for h in hits)} 字，"
          f"要是直接读全文是 {sum(h['full_len'] for h in hits)} 字。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
