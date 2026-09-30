from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cards import cheap_enough, discover, expand
from router import explain, route
from telemetry import Run
from store.store import connect, search, stats, touch
WHO = {'user': '用户', 'assistant': '我', 'agent_b': '助手B', 'agent_c': '助手C'}

def show(con, rows, why: bool) -> None:
    if not rows:
        print('没查到。库里没有，或者查询词太短（一个字不做召回）。')
        return
    for i, r in enumerate(rows, 1):
        pin = ' 📌' if r['user_pinned'] else ''
        conf = f' 置信{r['confidence']:.2f}' if r['confidence'] is not None else ''
        print(f'{i}. {r['content']}{pin}')
        print(f'   {WHO.get(r['subject'], r['subject'])}｜{r['status']}{conf}｜权重 {r['score']:.2f}')
        if why:
            ev = con.execute("SELECT e.source_path, e.source_locator, e.verification_status,\n                          substr(replace(e.content, char(10), ' '), 1, 90) AS head\n                   FROM fact_evidence fe JOIN evidence e ON e.id = fe.evidence_id\n                   JOIN facts f ON f.id = fe.fact_id\n                   WHERE f.ext_id = ?", (r['ext_id'],)).fetchall()
            if not ev:
                print('   出处：没有挂证据 ← 这条只能当线索，不能当依据')
            for e in ev:
                mark = {'verified': '✓', 'uncertain': '?', 'broken': '✗'}.get(e['verification_status'], '·')
                print(f'   出处{mark} {Path(e['source_path']).name}:{e['source_locator']}')
                print(f'        原文：{e['head']}')
        touch(con, r['ext_id'])
        print()
    con.commit()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('query', nargs='?')
    ap.add_argument('--who', choices=list(WHO), help='只看谁的事实')
    ap.add_argument('-n', '--limit', type=int, default=8)
    ap.add_argument('--why', action='store_true', help='带出处，追到原文哪一行')
    ap.add_argument('--all', action='store_true', help='连我标过「不认」的也列出来')
    ap.add_argument('--pinned', action='store_true', help='列出用户钉住的')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--auto', metavar='消息原文', help='交给 Router 决定该不该查、查什么、读多少')
    ap.add_argument('--expand', type=int, default=2, help='发现层命中后展开几条全文，默认 2；0 表示只看卡片')
    args = ap.parse_args()
    con = connect()
    if args.stats:
        s = stats(con)
        print(f'证据 {s['evidence']} 条（核对通过 {s['evidence_verified']}）')
        print(f'事实 {s['facts_by_status']}')
        print(f'当下有效 {s['live']}｜钉住 {s['pinned']}')
        return 0
    if args.pinned:
        rows = [dict(r) | {'score': 99.0} for r in con.execute('SELECT ext_id, subject, content, status, confidence, user_pinned\n               FROM live_facts WHERE user_pinned = 1 ORDER BY valid_from DESC')]
        show(con, rows, args.why)
        return 0
    if args.auto:
        d = route(args.auto, who=args.who)
        print(f'Router：{explain(d)}\n')
        if not d.should_recall:
            with Run(con, query=args.auto, decision='skip', reason=d.reason):
                pass
            return 0
        for q in d.queries:
            with Run(con, query=args.auto, decision='recall', reason=d.reason, terms=[q]) as run:
                hits = discover(con, q, subject=d.who, limit=d.limit)
                run.mode = 'direct' if cheap_enough(hits) else 'card'
                for i, h in enumerate(hits, 1):
                    picked = run.mode == 'direct' or i <= max(args.expand, 0)
                    run.add_hit(fact_id=h['fact_id'], rank=i, final_score=h['score'], salience_score=h['score'], layer=h.get('layer'), memory_type=h.get('fact_type'), snippet_chars=h['full_len'] if run.mode == 'direct' else len(h['show']), selected=picked)
            print(f'── 「{q}」发现层 ──')
            if not hits:
                print('  薄表里没有\n')
                continue
            full_chars = sum((h['full_len'] for h in hits))
            if cheap_enough(hits):
                print(f'  （全文合计才 {full_chars} 字，直接读，不绕卡片层）')
                for row in expand(con, [h['fact_id'] for h in hits]):
                    print(f'  · {row['content']}')
                    touch(con, row['ext_id'])
                con.commit()
                print()
                continue
            for i, h in enumerate(hits, 1):
                tag = '命中' if h['hit'] else '卡片'
                print(f'  {i}. [{h['score']:.2f}]（{tag}）{h['show']}  （全文 {h['full_len']} 字）')
            card_chars = sum((len(h['show']) for h in hits))
            if args.expand > 0:
                chosen = [h['fact_id'] for h in hits[:args.expand]]
                print(f'\n  展开前 {len(chosen)} 条：')
                for row in expand(con, chosen):
                    print(f'  · {row['content']}')
                    touch(con, row['ext_id'])
                con.commit()
                used = card_chars + sum((len(r['content']) for r in expand(con, chosen)))
            else:
                used = card_chars
            print(f'\n  这一轮花了 {used} 字；全读是 {full_chars} 字\n')
        return 0
    if not args.query:
        ap.print_help()
        return 1
    show(con, search(con, args.query, subject=args.who, limit=args.limit, include_disowned=args.all), args.why)
    return 0
if __name__ == '__main__':
    sys.exit(main())
