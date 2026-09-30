from __future__ import annotations
import argparse
import datetime as dt
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import bigrams
from store.store import connect
CARD_LEN = 40
RE_HEAD = re.compile('^(用户|助手|我|助手B|助手C)(（[^）]*）)?[：:]\\s*')
RE_SPACE = re.compile('\\s+')

def make_card(content: str) -> str:
    body = RE_HEAD.sub('', content)
    body = RE_SPACE.sub(' ', body).strip()
    for sep in ('。', '！', '？', '；', '\n'):
        idx = body.find(sep, 8)
        if 8 <= idx <= CARD_LEN:
            body = body[:idx]
            break
    if len(body) > CARD_LEN:
        body = body[:CARD_LEN] + '…'
    who = RE_HEAD.match(content)
    return f'{who.group(1)}：{body}' if who else body

def build(con, *, quiet: bool=False) -> dict:
    con.executescript((ROOT / 'cards.sql').read_text(encoding='utf-8'))
    con.execute('DELETE FROM fact_cards')
    con.execute("INSERT INTO cards_fts (cards_fts) VALUES ('delete-all')")
    con.execute("INSERT INTO cards_gram_fts (cards_gram_fts) VALUES ('delete-all')")
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = con.execute("SELECT id, subject, content FROM facts WHERE status IN ('candidate','confirmed')").fetchall()
    total_full = total_card = 0
    for r in rows:
        card = make_card(r['content'])
        total_full += len(r['content'])
        total_card += len(card)
        con.execute('INSERT INTO fact_cards (fact_id, subject, card, card_grams, full_len, built_at) VALUES (?,?,?,?,?,?)', (r['id'], r['subject'], card, bigrams(card), len(r['content']), now))
        con.execute('INSERT INTO cards_fts (rowid, card, subject) VALUES (?,?,?)', (r['id'], card, r['subject']))
        con.execute('INSERT INTO cards_gram_fts (rowid, card_grams) VALUES (?,?)', (r['id'], bigrams(card)))
    con.commit()
    out = {'卡片': len(rows), '全文总字数': total_full, '卡片总字数': total_card, '压缩到': f'{total_card / max(total_full, 1):.1%}'}
    if not quiet:
        print('、'.join((f'{k} {v}' for k, v in out.items())))
    return out
HIT_WIDTH = 60
MAX_WINDOWS = 2
MERGE_GAP = 24
HIT_BUDGET = 160

def _windows(body: str, positions: list[int], qlen: int) -> list[tuple[int, int]]:
    half = max(HIT_WIDTH // 2 - qlen // 2, 8)
    spans = [(max(0, p - half), min(len(body), p + qlen + half)) for p in positions]
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]

def snippet_around(content: str, query: str, width: int=HIT_WIDTH) -> str | None:
    body = RE_SPACE.sub(' ', content)
    needle = query
    if body.find(needle) < 0:
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
    parts, used = ([], 0)
    spans = _windows(body, positions, len(needle))
    for i, (a, b) in enumerate(spans):
        frag = body[a:b].strip()
        if used + len(frag) > HIT_BUDGET:
            frag = frag[:max(0, HIT_BUDGET - used)]
        if not frag:
            break
        used += len(frag)
        lead = '…' if a > 0 else ''
        tail = '…' if b < len(body) else ''
        parts.append(f'{lead}{frag}{tail}')
        if used >= HIT_BUDGET:
            break
    return ' '.join(parts)

def discover(con, query: str, *, subject=None, limit=5):
    q = query.strip()
    if len(q) < 2:
        return []
    if len(q) == 2:
        table, match = ('cards_gram_fts', f'"{q}"')
    else:
        table, match = ('cards_fts', q)
    sql = f"\n      SELECT c.fact_id, f.ext_id, c.subject, c.card, c.full_len, f.layer, f.fact_type,\n             engram_activation(f.importance, f.confidence, f.emotion, f.access_count,\n                               julianday('now') - julianday(f.valid_from),\n                               f.user_pinned, f.fact_type) AS score\n      FROM {table} t\n      JOIN fact_cards c ON c.fact_id = t.rowid\n      JOIN facts f ON f.id = c.fact_id\n      LEFT JOIN current_stances cs ON cs.fact_id = f.id AND cs.actor = 'assistant'\n      WHERE {table} MATCH ?\n        AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL\n        AND COALESCE(cs.stance,'none') <> 'disown'\n    "
    args: list = [match]
    if subject:
        sql += ' AND c.subject = ?'
        args.append(subject)
    sql += ' ORDER BY score DESC LIMIT ?'
    args.append(limit)
    out = []
    for r in con.execute(sql, args):
        row = dict(r)
        full = con.execute('SELECT content FROM facts WHERE id=?', (row['fact_id'],)).fetchone()['content']
        row['hit'] = snippet_around(full, q)
        row['show'] = row['hit'] or row['card']
        out.append(row)
    return out
DIRECT_READ_BUDGET = 400

def cheap_enough(hits, budget: int=DIRECT_READ_BUDGET) -> bool:
    return sum((h['full_len'] for h in hits)) <= budget

def expand(con, fact_ids):
    if not fact_ids:
        return []
    marks = ','.join('?' * len(fact_ids))
    return [dict(r) for r in con.execute(f'SELECT ext_id, subject, content, status, user_pinned, layer FROM facts WHERE id IN ({marks})', list(fact_ids))]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='?')
    ap.add_argument('--build', action='store_true')
    ap.add_argument('-n', '--limit', type=int, default=5)
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
        print('薄表里没找到。')
        return 0
    for i, h in enumerate(hits, 1):
        tag = '命中' if h['hit'] else '卡片'
        print(f'{i}. [{h['score']:.3f}]（{tag}）{h['show']}    （全文 {h['full_len']} 字）')
    print(f'\n发现层一共给了 {sum((len(h['show']) for h in hits))} 字，要是直接读全文是 {sum((h['full_len'] for h in hits))} 字。')
    return 0
if __name__ == '__main__':
    sys.exit(main())
