from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from gate.gate import find_slots
from store.store import connect
WHO = {'user': '用户', 'assistant': '我', 'agent_b': '助手B', 'agent_c': '助手C'}

def bucket(n: int) -> str:
    for edge, name in ((12, '≤12'), (20, '13-20'), (40, '21-40'), (100, '41-100')):
        if n <= edge:
            return name
    return '>100'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0, help='随机抽几条晋升事实人工看')
    args = ap.parse_args()
    con = connect()
    ev = con.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
    fa = con.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
    print(f'证据 {ev}｜事实 {fa}｜晋升率 {fa / max(ev, 1):.1%}')
    print('\n按人：')
    for actor, n in con.execute('SELECT subject, COUNT(*) FROM facts GROUP BY subject ORDER BY 2 DESC'):
        evn = con.execute('SELECT COUNT(*) FROM evidence WHERE source_actor=?', (actor,)).fetchone()[0]
        print(f'  {WHO.get(actor, actor):<8} 事实 {n:>5}   该人证据 {evn:>5}   {n / max(evn, 1):>6.1%}')
    print('\n按长度：')
    by_len: Counter = Counter()
    for c, in con.execute('SELECT content FROM facts'):
        by_len[bucket(len(c.split('：', 1)[-1]))] += 1
    for b in ('≤12', '13-20', '21-40', '41-100', '>100'):
        print(f'  {b:<8} {by_len.get(b, 0):>5}')
    print('\n事实槽位命中分布（一条可以命中多个）：')
    slot_count: Counter = Counter()
    none_slot = []
    for c, in con.execute('SELECT content FROM facts'):
        body = c.split('：', 1)[-1]
        s = find_slots(body)
        if not s:
            none_slot.append(c)
        for name in s:
            slot_count[name] += 1
    for name, n in slot_count.most_common():
        print(f'  {name:<6} {n:>5}')
    print(f'  没有任何槽位 {len(none_slot):>5}   ← 这些是门控该拦下的')
    for c in none_slot[:8]:
        print(f'      · {c[:52]}')
    if args.sample:
        print(f'\n随机抽 {args.sample} 条晋升事实：')
        for c, imp in con.execute('SELECT content, importance FROM facts ORDER BY random() LIMIT ?', (args.sample,)):
            body = c.split('：', 1)[-1]
            print(f'  [{(imp if imp is not None else '-')}] {'＋'.join(find_slots(body)) or '无槽位':<12} {c[:56]}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
