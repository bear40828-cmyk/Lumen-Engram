from __future__ import annotations
import os
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rerank import judge
from retrieve import retrieve
from store.store import connect
_spec = importlib.util.spec_from_file_location('hook', Path(os.environ.get('ENGRAM_HOOK_PATH', str(Path(__file__).resolve().parent / 'hooks_engram_recall.py'))))
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

def picked(con, text: str) -> tuple[list[str], str]:
    if hook.skip_input(text):
        return ([], '跳过输入')
    marks = hook.distinctive_terms(text)
    if not marks and (not hook.EXPLICIT_RECALL.search(text)) and (not hook.QUESTION.search(text)):
        return ([], '无指向词')
    r = retrieve(con, text, who=None)
    pinned = hook.pinned_candidates(con, text)
    if not r.should_recall and (not pinned or '软话' in (r.reason or '')):
        return ([], f'路由不查:{r.reason}')
    seen = {p['ext_id'] for p in pinned}
    hits = pinned + [h for h in (r.hits if r.should_recall else []) if h.get('ext_id') not in seen]
    if not hits:
        return ([], '检索无结果')
    j = judge(text, [h.get('content') or '' for h in hits])
    if j is None:
        return ([], '精排失败')
    if not j['need']:
        return ([], '精排判不用查')
    top = [hits[i] for sc, i in sorted(zip(j['scores'], range(len(hits))), reverse=True) if sc >= 7][:hook.MAX_HITS]
    if not top:
        return ([], '精排不够分' + ('' if pinned else '（纠正层没捞到）'))
    return ([h.get('content') or '' for h in top], '塞了别的' + ('' if pinned else '（纠正层没捞到）'))

def negative(n: int) -> None:
    import glob
    sess = Path(os.environ.get('ENGRAM_TRANSCRIPT_DIR', str(Path.home() / '.claude/projects')))
    pool = []
    for f in sorted(glob.glob(str(sess / '*.jsonl')), key=lambda p: Path(p).stat().st_mtime)[-6:]:
        for line in open(f, encoding='utf-8', errors='ignore'):
            if '"type":"user"' in line and '<channel' in line:
                from corrections import _text
                for m in hook.RE_CHANNEL.finditer(_text(json.loads(line)['message']['content'])):
                    b = m.group('body').strip()
                    if 4 <= len(b) <= 200:
                        pool.append(b)
    random.seed(1)
    qs = random.sample(pool, min(n, len(pool)))
    con = connect()
    fp = any_inject = 0
    for q in qs:
        got, _ = picked(con, q)
        any_inject += bool(got)
        bad = [g for g in got if g.startswith('【用户纠正过我')]
        if bad:
            fp += 1
            print(f'误报：{q[:40]} → {bad[0][:60]}')
    print(f'\n{len(qs)} 句日常话：塞了纠正 {fp} 句，塞了任何东西 {any_inject} 句')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=0)
    ap.add_argument('--hard', action='store_true', help='用换了说法的题（ask_hard）')
    ap.add_argument('--neg', type=int, default=0, help='反面卷：随机抽用户 N 句日常话，看会不会乱塞纠正')
    args = ap.parse_args()
    if args.neg:
        return negative(args.neg)
    cases = []
    for x in open(ROOT / 'corrections.jsonl', encoding='utf-8'):
        c = json.loads(x)
        if not c.get('keep'):
            continue
        key = c['right']
        for q in (c.get('ask_hard' if args.hard else 'ask'),):
            if q and len(q) >= 4:
                cases.append((q, key, c['topic']))
    random.seed(0)
    if args.n:
        cases = random.sample(cases, min(args.n, len(cases)))
    con = connect()
    ok, why = (0, {})
    for q, key, topic in cases:
        got, reason = picked(con, q)
        hit = any((key in g for g in got))
        ok += hit
        if not hit:
            why[reason] = why.get(reason, 0) + 1
            print(f'✗ [{topic}] {q[:40]} → {reason}')
    total = len(cases)
    print(f'\n命中 {ok}/{total} = {ok / max(total, 1):.0%}')
    print('没命中卡在：', dict(sorted(why.items(), key=lambda x: -x[1])))
if __name__ == '__main__':
    main()
