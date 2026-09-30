from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import read_line, sha256
from extract.from_digest import CN, SPEAKER_ACTOR, load_lines, locate, parse_digest
from classify import classify
from gate.gate import judge
from keys import match as match_key
from store.store import add_evidence, connect, promote, record_class, record_key, record_tier, stats
DIGESTS = Path.home() / '.claude/projects/-home-ubuntu-workspace-claude/memory/digests'

def ingest_one(con, digest: Path) -> dict:
    jsonl_path, bullets = parse_digest(digest)
    if not jsonl_path or not Path(jsonl_path).exists():
        return {'skipped_file': True}
    day = digest.stem
    cache: dict[tuple[int, int], list] = {}
    tally = {'promoted': 0, 'gate_no': 0, 'no_source': 0, 'dup': 0}
    for b in bullets:
        key = (b.line_lo, b.line_hi)
        cache.setdefault(key, load_lines(jsonl_path, *key))
        hit = locate(b, cache[key])
        if not hit:
            tally['no_source'] += 1
            continue
        line_no, real_text = hit
        occurred = dt.datetime.strptime(f'{day} {b.hhmm}', '%Y-%m-%d %H:%M').replace(tzinfo=CN).astimezone(dt.timezone.utc).isoformat()
        got = read_line(jsonl_path, line_no)
        status = 'verified' if got and sha256(got[1]) == sha256(real_text) else 'uncertain'
        ev_id = add_evidence(con, source_kind='chat', source_actor=SPEAKER_ACTOR[b.speaker], source_path=jsonl_path, source_locator=f'L{line_no}', content=real_text, occurred_at=occurred, verification_status=status)
        record_tier(con, ev_id, 'primary', f'{jsonl_path}:L{line_no}')
        subject = SPEAKER_ACTOR[b.speaker]
        content = f'{b.speaker}（{day} {b.hhmm}）：{b.text}'
        verdict = judge(con, subject=subject, content=content, evidence_rows=[{'verification_status': status}])
        if not verdict.promote:
            tally['gate_no'] += 1
            continue
        kind, kconf, kwhy = classify(content, subject)
        fact_id = promote(con, subject=subject, fact_type=kind, content=content, evidence_ids=[ev_id], valid_from=occurred, importance=verdict.importance, confidence=verdict.confidence)
        if fact_id is not None:
            record_class(con, fact_id, kind, kconf, kwhy)
            fkey, fval, fpol, fok = match_key(content)
            if fkey:
                record_key(con, fact_id, fkey, fval, fpol, fok)
        tally['dup' if fact_id is None else 'promoted'] += 1
    con.commit()
    return tally

def main() -> int:
    con = connect()
    want = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(DIGESTS.glob('*.md'))
    if want:
        files = [f for f in files if f.stem == want]
    if not files:
        print('没有匹配的摘要文件', file=sys.stderr)
        return 1
    total = {'promoted': 0, 'gate_no': 0, 'no_source': 0, 'dup': 0}
    for f in files:
        t = ingest_one(con, f)
        if t.get('skipped_file'):
            print(f'{f.stem}  出处 jsonl 不在了，跳过')
            continue
        for k in total:
            total[k] += t.get(k, 0)
        print(f'{f.stem}  晋升 {t['promoted']:>3}   门控拒 {t['gate_no']:>3}   无原话 {t['no_source']:>3}   重复 {t['dup']:>3}')
    print(f'\n合计：晋升 {total['promoted']}，门控拒 {total['gate_no']}，无原话 {total['no_source']}，重复 {total['dup']}')
    s = stats(con)
    print(f'库：证据 {s['evidence']}（已核对 {s['evidence_verified']}）｜事实 {s['facts_by_status']}｜当下有效 {s['live']}｜钉住 {s['pinned']}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
