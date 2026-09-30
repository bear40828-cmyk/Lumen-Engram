from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from keys import by_name
from store.store import connect, promote, record_class, relate
MIN_SUPPORT = 3.0
MIN_DAYS = 2
ECHO_WEIGHT = 1 / 3
MIN_OWNER_VOTES = 1
WHO = {'user': '用户', 'assistant': '助手', 'agent_b': '助手B', 'agent_c': '助手C'}

def candidates(con) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in con.execute("\n            SELECT f.id, f.subject, f.content, f.valid_from, f.fact_type,\n                   m.fact_key, m.value, m.polarity, t.origin_root,\n                   e.source_actor\n            FROM fact_key_map m\n            JOIN facts f ON f.id = m.fact_id\n            JOIN fact_evidence fe ON fe.fact_id = f.id\n            JOIN evidence e ON e.id = fe.evidence_id\n            JOIN evidence_tier t ON t.evidence_id = e.id\n            WHERE f.status IN ('candidate','confirmed') AND f.valid_to IS NULL\n              AND m.value IS NOT NULL AND m.eligible = 1\n              AND t.tier = 'primary'\n              "):
        r = dict(r)
        spec = by_name(r['fact_key'])
        if spec is not None and (not spec.consolidatable):
            continue
        owner = (spec.owner if spec is not None else None) or r['subject']
        r['weight'] = 1.0 if r['source_actor'] == owner else ECHO_WEIGHT
        r['subject'] = owner
        groups[owner, r['fact_key'], r['value'], r['polarity']].append(r)
    out = []
    for (subject, key, value, polarity), rows in groups.items():
        by_root: dict[str, dict] = {}
        for r in rows:
            cur = by_root.get(r['origin_root'])
            if cur is None or r['weight'] > cur['weight']:
                by_root[r['origin_root']] = r
        rows = list(by_root.values())
        votes = sum((r['weight'] for r in rows))
        owner_votes = sum((1 for r in rows if r['weight'] == 1.0))
        days = {(r['valid_from'] or '')[:10] for r in rows if r['valid_from']}
        if votes < MIN_SUPPORT or len(days) < MIN_DAYS or owner_votes < MIN_OWNER_VOTES:
            continue
        already = con.execute('\n            SELECT 1 FROM fact_relations r JOIN fact_key_map m ON m.fact_id = r.parent_fact_id\n            WHERE m.fact_key=? AND m.value=? LIMIT 1', (key, value)).fetchone()
        if already:
            continue
        out.append({'subject': subject, 'key': key, 'value': value, 'polarity': polarity, 'rows': rows, 'days': sorted(days), 'votes': votes, 'owner_votes': owner_votes})
    return out

def phrase(subject: str, key: str, value: str, n: int, days: list[str], polarity: str='pos') -> str:
    who = WHO.get(subject, subject)
    span = f'{days[0]} 到 {days[-1]}' if len(days) > 1 else days[0]
    topic = {'preference.color': '在颜色上稳定偏向', 'preference.food': '口味上反复提到', 'education.target_school': '考研目标院校是', 'education.target_major': '考研目标专业是', 'education.exam_subject': '备考科目里反复出现', 'education.graduation': '毕业时间是', 'person.hometown': '老家是', 'person.pet': '养的是', 'comm.format_rule': '沟通格式上的固定要求是', 'project.status': '项目状态反复被记为'}.get(key, f'在 {key} 上反复提到')
    neg = '反向（明确否定）' if polarity == 'neg' else ''
    return f'{who}：{topic}「{value}」{neg}——{span} 之间有 {n} 次独立支撑，由巩固得出'

def run(con, *, apply: bool) -> int:
    cands = candidates(con)
    if not cands:
        print(f'没有够条件的（要求：≥{MIN_SUPPORT} 条独立支撑、来自 ≥{MIN_DAYS} 个不同日子）')
        return 0
    for c in cands:
        spec = by_name(c['key'])
        content = phrase(c['subject'], c['key'], c['value'], len(c['rows']), c['days'], c['polarity'])
        content += f'（其中本人原话 {c['owner_votes']} 次）'
        print(f'\n{content}')
        print(f'  key={c['key']}  cardinality={(spec.cardinality if spec else '?')}  conflict={(spec.conflict_policy if spec else '?')}')
        for r in c['rows'][:4]:
            print(f'    ← {(r['valid_from'] or '')[:10]}  {r['content'][:52]}')
        if len(c['rows']) > 4:
            print(f'    …… 还有 {len(c['rows']) - 4} 条支撑')
        if not apply:
            continue
        fid = promote(con, subject=c['subject'], fact_type='semantic', content=content, valid_from=c['days'][0], importance=0.8, confidence=0.85)
        if fid is None:
            print('    （库里已有同样的抽象，跳过）')
            continue
        record_class(con, fid, 'semantic', 0.85, f'由 {len(c['rows'])} 条支撑巩固而来，跨 {len(c['days'])} 天')
        con.execute("INSERT OR REPLACE INTO fact_key_map (fact_id, fact_key, value, built_at) VALUES (?,?,?,datetime('now'))", (fid, c['key'], c['value']))
        for r in c['rows']:
            relate(con, fid, r['id'], 'derived_from')
        con.commit()
        print(f'    ✓ 写入 {len(c['rows'])} 条 derived_from，原始那几条一个字没动')
    print(f'\n共 {len(cands)} 组够条件' + ('' if apply else '（干看，没写库）'))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    return run(connect(), apply=args.apply)
if __name__ == '__main__':
    sys.exit(main())
