from __future__ import annotations
import os
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from classify import classify
from gate.gate import judge
from keys import match as match_key
from store.store import add_evidence, connect, promote, record_class, record_key, record_tier, stats
WHO = {'user': '用户', 'assistant': '我', 'agent_b': '助手B', 'agent_c': '助手C'}
CN = dt.timezone(dt.timedelta(hours=8))
PROJ = Path(os.environ.get('ENGRAM_TRANSCRIPT_DIR', str(Path.home() / '.claude/projects')))
RE_CHANNEL = re.compile('<channel\\s+(?P<attrs>[^>]*?source="[^"]*telegram[^"]*"[^>]*?)>(?P<body>.*?)</channel>', re.S)
RE_ATTR = re.compile('(\\w+)="([^"]*)"')
RE_TAGGY = re.compile('^\\s*<[^>]+>\\s*$')
# 身份映射从环境变量读，格式 id:role,id:role / name:role,name:role
def _pairs(v):
    return dict(x.split(':', 1) for x in os.environ.get(v, '').split(',') if ':' in x)
BY_ID = _pairs('ENGRAM_CHANNEL_IDS')
BY_NAME = _pairs('ENGRAM_CHANNEL_NAMES')

def whose(attrs: str) -> str:
    a = dict(RE_ATTR.findall(attrs))
    uid, uname = (a.get('user_id', ''), a.get('user', ''))
    return BY_ID.get(uid) or BY_NAME.get(uname) or f'unknown:{uname or uid or '?'}'

def beijing(iso_utc: str) -> tuple[str, str]:
    t = dt.datetime.fromisoformat(iso_utc.replace('Z', '+00:00'))
    return (t.isoformat(), t.astimezone(CN).strftime('%Y-%m-%d %H:%M'))

def harvest(path: Path):
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for line_no, raw in enumerate(fh, start=1):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = rec.get('message')
            if not isinstance(msg, dict):
                continue
            role, content = (msg.get('role'), msg.get('content'))
            if role == 'user':
                text = content if isinstance(content, str) else '\n'.join((b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text')) if isinstance(content, list) else ''
                for m in RE_CHANNEL.finditer(text or ''):
                    body = m.group('body').strip()
                    if not body or RE_TAGGY.match(body):
                        continue
                    attrs = dict(RE_ATTR.findall(m.group('attrs')))
                    yield (line_no, whose(m.group('attrs')), body, attrs.get('ts'))
            elif role == 'assistant' and isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    kind = blk.get('type')
                    if kind == 'tool_use':
                        if 'reply' not in (blk.get('name') or ''):
                            continue
                        inp = blk.get('input')
                        if isinstance(inp, dict) and isinstance(inp.get('text'), str):
                            body = inp['text'].strip()
                            if body:
                                yield (line_no, 'assistant', body, None)
                        continue
                    if kind == 'text':
                        body = (blk.get('text') or '').strip()
                        if len(body) >= 8:
                            yield (line_no, 'assistant', body, None)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--since', help='只要这天及以后的（按 channel 时间戳，YYYY-MM-DD）')
    args = ap.parse_args()
    files = args.files or sorted(PROJ.glob('*.jsonl'))
    con = connect()
    tally: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for path in files:
        for line_no, actor, body, ts in harvest(path):
            when_utc, when_cn = beijing(ts) if ts else (None, '')
            if args.since and when_utc and (when_utc[:10] < args.since):
                continue
            who = WHO.get(actor, actor)
            head = f'{who}（{when_cn}）：' if when_cn else f'{who}：'
            content = head + body.replace('\n', ' ')
            if args.dry_run:
                tally[actor] = tally.get(actor, 0) + 1
                continue
            ev_id = add_evidence(con, source_kind='chat', source_actor=actor, source_path=str(path), source_locator=f'L{line_no}', content=body, occurred_at=when_utc, verification_status='verified')
            record_tier(con, ev_id, 'primary', f'{path}:L{line_no}')
            v = judge(con, subject=actor, content=content, evidence_rows=[{'verification_status': 'verified'}])
            if not v.promote:
                tally['_gate_no'] = tally.get('_gate_no', 0) + 1
                reasons[v.reason] = reasons.get(v.reason, 0) + 1
                continue
            kind, kconf, kwhy = classify(content, actor)
            fact_id = promote(con, subject=actor, fact_type=kind, content=content, evidence_ids=[ev_id], valid_from=when_utc, importance=v.importance, confidence=v.confidence)
            if fact_id is not None:
                record_class(con, fact_id, kind, kconf, kwhy)
                fkey, fval, fpol, fok = match_key(content)
                if fkey:
                    record_key(con, fact_id, fkey, fval, fpol, fok)
            key = '_dup' if fact_id is None else actor
            tally[key] = tally.get(key, 0) + 1
    if not args.dry_run:
        con.commit()
    print('入库：')
    for k in sorted(tally, key=lambda k: -tally[k]):
        if k.startswith('_'):
            continue
        print(f'  {WHO.get(k, k):<10} {tally[k]:>6} 条')
    print(f'  门控拒 {tally.get('_gate_no', 0)}，重复 {tally.get('_dup', 0)}')
    if reasons:
        print('门控拒的理由：')
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f'  {n:>5}  {r}')
    if not args.dry_run:
        print('库：', stats(con))
    return 0
if __name__ == '__main__':
    sys.exit(main())
