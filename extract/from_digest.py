from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
CN = dt.timezone(dt.timedelta(hours=8))
SPEAKER_ROLE = {'用户': 'user', '助手': 'assistant'}
SPEAKER_ACTOR = {'用户': 'user', '助手': 'assistant'}
RE_SOURCE = re.compile('出处是\\s*`([^`]+)`')
RE_SECTION = re.compile('^##\\s+\\d{2}:\\d{2}[–-]\\d{2}:\\d{2}.*?第\\s*(\\d+)[–-](\\d+)\\s*行')
RE_BULLET = re.compile('^-\\s+(\\d{2}):(\\d{2})\\s+([^：:]{1,12})[：:]\\s*(.+)$')
RE_QUOTED = re.compile('[「『\\"“]([^」』\\"”]{2,})[」』\\"”]')

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def bigrams(text: str) -> str:
    s = ''.join((ch for ch in text if not ch.isspace()))
    return ' '.join((s[i:i + 2] for i in range(len(s) - 1))) or s

@dataclass
class Bullet:
    hhmm: str
    speaker: str
    text: str
    line_lo: int
    line_hi: int
    quoted: list[str] = field(default_factory=list)

def parse_digest(path: Path) -> tuple[str, list[Bullet]]:
    jsonl_path, bullets = ('', [])
    lo = hi = 0
    for raw in path.read_text(encoding='utf-8').splitlines():
        if not jsonl_path and (m := RE_SOURCE.search(raw)):
            jsonl_path = os.path.expanduser(m.group(1))
            continue
        if (m := RE_SECTION.match(raw)):
            lo, hi = (int(m.group(1)), int(m.group(2)))
            continue
        if lo and (m := RE_BULLET.match(raw)):
            hh, mm, speaker, text = m.groups()
            if speaker not in SPEAKER_ROLE:
                continue
            bullets.append(Bullet(f'{hh}:{mm}', speaker, text.strip(), lo, hi, RE_QUOTED.findall(text)))
    return (jsonl_path, bullets)

def load_lines(jsonl_path: str, lo: int, hi: int) -> list[tuple[int, str, str]]:
    out = []
    with open(jsonl_path, encoding='utf-8', errors='ignore') as fh:
        for idx, raw in enumerate(fh, start=1):
            if idx < lo:
                continue
            if idx > hi:
                break
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = rec.get('message')
            if not isinstance(msg, dict):
                continue
            role, content = (msg.get('role'), msg.get('content'))
            if role not in ('user', 'assistant'):
                continue
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for b in content:
                    if not isinstance(b, dict):
                        continue
                    if b.get('type') == 'text':
                        parts.append(b.get('text', ''))
                    elif b.get('type') == 'tool_use':
                        inp = b.get('input')
                        if isinstance(inp, dict) and isinstance(inp.get('text'), str):
                            parts.append(inp['text'])
                text = '\n'.join((p for p in parts if p))
            else:
                continue
            if text.strip():
                out.append((idx, role, text))
    return out

def locate(bullet: Bullet, lines: list[tuple[int, str, str]]) -> tuple[int, str] | None:
    want_role = SPEAKER_ROLE[bullet.speaker]
    cands = [(n, t) for n, role, t in lines if role == want_role]
    for quote in bullet.quoted:
        needle = quote.strip()
        if len(needle) < 2:
            continue
        for n, text in cands:
            if needle in text:
                return (n, text)
    return None

def ensure_db(db_path: Path) -> sqlite3.Connection:
    schema = (Path(__file__).resolve().parent.parent / 'schema.sql').read_text(encoding='utf-8')
    con = sqlite3.connect(db_path)
    con.executescript(schema)
    return con

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('digest', type=Path)
    ap.add_argument('--db', type=Path, default=Path(__file__).resolve().parent.parent / 'engram.db')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    jsonl_path, bullets = parse_digest(args.digest)
    if not jsonl_path:
        print('摘要里没写出处 jsonl，拒绝抽取', file=sys.stderr)
        return 1
    if not os.path.exists(jsonl_path):
        print(f'出处文件不存在：{jsonl_path}', file=sys.stderr)
        return 1
    day = args.digest.stem
    con = ensure_db(args.db)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    stats = {'verified': 0, 'uncertain': 0, 'skipped': 0}
    cache: dict[tuple[int, int], list] = {}
    for b in bullets:
        key = (b.line_lo, b.line_hi)
        if key not in cache:
            cache[key] = load_lines(jsonl_path, *key)
        hit = locate(b, cache[key])
        occurred = dt.datetime.strptime(f'{day} {b.hhmm}', '%Y-%m-%d %H:%M').replace(tzinfo=CN).astimezone(dt.timezone.utc).isoformat()
        if not hit:
            stats['uncertain' if b.quoted else 'skipped'] += 1
            if not b.quoted:
                continue
            ev_id = None
            status = 'uncertain'
        else:
            line_no, real_text = hit
            locator = f'L{line_no}'
            ev_id = None
            if not args.dry_run:
                try:
                    cur = con.execute('INSERT INTO evidence (ext_id, captured_at, occurred_at, source_kind,\n                             source_actor, source_path, source_locator, content, content_sha256,\n                             verification_status, verified_at)\n                           VALUES (?,?,?,?,?,?,?,?,?,?,?)', (f'ev_{uuid.uuid4().hex[:12]}', now, occurred, 'chat', SPEAKER_ACTOR[b.speaker], jsonl_path, locator, real_text, sha256(real_text), 'verified', now))
                    ev_id = cur.lastrowid
                except sqlite3.IntegrityError:
                    ev_id = con.execute('SELECT id FROM evidence WHERE source_locator=? AND content_sha256=?', (locator, sha256(real_text))).fetchone()[0]
            stats['verified'] += 1
            status = 'candidate'
        if args.dry_run:
            mark = '✓' if hit else '?'
            print(f'  {mark} {b.hhmm} {b.speaker}: {b.text[:42]}' + (f'  → {jsonl_path.split('/')[-1]}:{hit[0]}' if hit else '  → 没对上'))
            continue
        fact_text = f'{b.speaker}（{day} {b.hhmm}）：{b.text}'
        try:
            cur = con.execute('INSERT INTO facts (ext_id, subject, fact_type, content, grams, content_sha256,\n                     status, valid_from, created_at, updated_at)\n                   VALUES (?,?,?,?,?,?,?,?,?,?)', (f'fact_{uuid.uuid4().hex[:12]}', SPEAKER_ACTOR[b.speaker], 'episodic', fact_text, bigrams(fact_text), sha256(fact_text), status, occurred, now, now))
            fact_id = cur.lastrowid
        except sqlite3.IntegrityError:
            continue
        if ev_id:
            con.execute('INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id) VALUES (?,?)', (fact_id, ev_id))
    if not args.dry_run:
        con.commit()
    print(f'\n{args.digest.name}：带原话且核对上 {stats['verified']} 条，有原话但没对上 {stats['uncertain']} 条，无原话跳过 {stats['skipped']} 条')
    if not args.dry_run:
        print(f'库：{args.db}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
