from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store.store import _bump, connect, relate
from extract import timeline

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    con = connect()
    rows = con.execute("SELECT f.id, f.ext_id, f.subject, f.content, f.valid_from FROM facts f\n           WHERE f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')\n             AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL\n           ORDER BY f.valid_from, f.id").fetchall()
    print(f'待过 {len(rows)} 条', file=sys.stderr)
    n = 0
    for r in rows:
        still = con.execute('SELECT valid_to FROM facts WHERE id=?', (r['id'],)).fetchone()
        if still['valid_to'] is not None:
            continue
        olds = [o for o in timeline.candidates(con, r['subject'], r['content'], r['valid_from']) if o['id'] != r['id']]
        idx, why = timeline.judge(r['content'], r['valid_from'] or '', olds)
        if idx is None:
            continue
        old = olds[idx]
        n += 1
        print(f'{(old.get('valid_from') or '')[:10]} {old['content'][:50]}\n  → {(r['valid_from'] or '')[:10]} {r['content'][:50]}\n  （{why}）')
        if not args.dry_run:
            _bump(con, old['id'], None, status='superseded', valid_to=r['valid_from'])
            relate(con, r['id'], old['id'], 'contradicts')
            con.commit()
    print(f'顶替 {n} 条', file=sys.stderr)
    return 0
if __name__ == '__main__':
    sys.exit(main())
