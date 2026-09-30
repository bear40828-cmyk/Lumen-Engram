from __future__ import annotations
import argparse
import datetime as dt
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import message_text, sha256
RE_TG = 'plugin:telegram:telegram'

def line_text(raw: str) -> str:
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError:
        return ''
    _, text = message_text(rec)
    return text

def verify(db: Path, limit: int | None, quiet: bool) -> int:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = con.execute("SELECT id, ext_id, source_path, source_locator, content, content_sha256,\n                  verification_status\n           FROM evidence WHERE source_kind='chat' ORDER BY source_path, id" + (f' LIMIT {int(limit)}' if limit else '')).fetchall()
    todo: dict[str, dict[int, list[sqlite3.Row]]] = defaultdict(lambda: defaultdict(list))
    unsupported = 0
    for r in rows:
        loc = r['source_locator'] or ''
        if not (r['source_path'] and loc.startswith('L') and loc[1:].isdigit()):
            unsupported += 1
            continue
        todo[r['source_path']][int(loc[1:])].append(r)
    tally = {'verified': 0, 'uncertain': 0, 'broken': 0}
    changed: list[tuple] = []
    for path, by_line in todo.items():
        try:
            fh = open(path, encoding='utf-8', errors='ignore')
        except OSError:
            for rs in by_line.values():
                for r in rs:
                    tally['broken'] += 1
                    if r['verification_status'] != 'broken':
                        changed.append((r['ext_id'], r['verification_status'], 'broken', Path(path).name, r['source_locator']))
                        con.execute("UPDATE evidence SET verification_status='broken', verified_at=? WHERE id=?", (now, r['id']))
            continue
        with fh:
            wanted = set(by_line)
            for idx, raw in enumerate(fh, start=1):
                if idx not in wanted:
                    continue
                whole = line_text(raw)
                for r in by_line[idx]:
                    stored = r['content']
                    ok = sha256(whole) == r['content_sha256'] or stored in whole
                    status = 'verified' if ok else 'uncertain'
                    tally[status] += 1
                    if status != r['verification_status']:
                        changed.append((r['ext_id'], r['verification_status'], status, Path(path).name, r['source_locator']))
                        con.execute('UPDATE evidence SET verification_status=?, verified_at=? WHERE id=?', (status, now, r['id']))
                wanted.discard(idx)
                if not wanted:
                    break
            for idx in wanted:
                for r in by_line[idx]:
                    tally['broken'] += 1
                    if r['verification_status'] != 'broken':
                        changed.append((r['ext_id'], r['verification_status'], 'broken', Path(path).name, r['source_locator']))
                        con.execute("UPDATE evidence SET verification_status='broken', verified_at=? WHERE id=?", (now, r['id']))
    con.commit()
    print(f'核对 {len(rows)} 条证据：对上 {tally['verified']}，内容变了 {tally['uncertain']}，读不到 {tally['broken']}，格式不支持 {unsupported}')
    if changed:
        print(f'\n状态变了 {len(changed)} 条：')
        for ext_id, old, new, fname, loc in changed[:20]:
            print(f'  {ext_id}  {old} → {new}   {fname}:{loc}')
        if len(changed) > 20:
            print(f'  …… 还有 {len(changed) - 20} 条')
    bad = con.execute("SELECT f.ext_id, f.status, e.verification_status, substr(f.content,1,46) AS head\n           FROM facts f\n           JOIN fact_evidence fe ON fe.fact_id = f.id\n           JOIN evidence e ON e.id = fe.evidence_id\n           WHERE e.verification_status IN ('uncertain','broken')").fetchall()
    if bad and (not quiet):
        print(f'\n⚠ 有 {len(bad)} 条事实挂着已降级的证据（只提醒，不自动改状态）：')
        for r in bad[:20]:
            print(f'  {r['ext_id']} [{r['status']}] 证据 {r['verification_status']}：{r['head']}')
        if len(bad) > 20:
            print(f'  …… 还有 {len(bad) - 20} 条')
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', type=Path, default=Path(__file__).resolve().parent.parent / 'engram.db')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    return verify(args.db, args.limit, args.quiet)
if __name__ == '__main__':
    sys.exit(main())
