from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import gramset
FOK_ESCALATE = 0.4
TOT_TOP_MAX = 0.55
TOT_MIN_HITS = 3
DEPTH = 5
ASK = {'year': re.compile('(哪年|哪一年|那年|几几年|哪届)'), 'date': re.compile('(哪天|哪一天|什么时候|几号|几月|多久|哪个月|上次是.*时候|日期)'), 'number': re.compile('(多少|几分|几个|几点|几次|几条|占比|百分|多大|几岁|多少钱|多少分)'), 'person': re.compile('(是谁|谁说|谁的|谁做|哪个人|谁给)'), 'place': re.compile('(在哪|哪里|哪儿|什么地方|去了哪)')}
HAS = {'year': re.compile('(\\d{4}\\s*年|\\d{4}-\\d{2}|去年|前年|今年|大前年|[初高][一二三]那年|\\d{4}\\s*届)'), 'date': re.compile('(\\d{4}\\s*[-年/]|\\d{1,2}\\s*[月日号]|\\d{4}-\\d{2}-\\d{2}|[今昨前明后]天|上周|上个月|去年|前年|凌晨|早上|下午|晚上|半夜)'), 'number': re.compile('(\\d|[一二三四五六七八九十百千万两半]+(?=[个条分点次成％%元块])|%)'), 'person': re.compile('(用户|助手|室友|老师|用户|他|我)'), 'place': re.compile('(北京|上海|酒店|车站|宿舍|教室|家里|学校|集训营)')}
RE_STAMP = re.compile('[（(]\\s*\\d{4}-\\d{2}-\\d{2}[^）)]*[）)]')
RE_SPEAKER = re.compile('^[^：:\\n]{1,12}[：:]')

def body(content: str) -> str:
    t = RE_STAMP.sub('', content or '')
    return RE_SPEAKER.sub('', t, count=1)

@dataclass(frozen=True)
class Judgment:
    fok: float
    tot: bool
    enough: bool
    reason: str
    expects: tuple[str, ...] = ()
    answered: tuple[str, ...] = ()
    top: float = 0.0
    spread: float = 0.0

def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum(((x - m) ** 2 for x in xs)) / len(xs)

def expects(text: str) -> tuple[str, ...]:
    return tuple((k for k, p in ASK.items() if p.search(text or '')))

def satisfied(hits: list[dict], kinds: tuple[str, ...], depth: int=DEPTH) -> tuple[tuple[str, ...], float]:
    if not kinds or not hits:
        return ((), 0.0)
    head = [body(h.get('content') or '') for h in hits[:depth]]
    got, hit_rows = ([], 0)
    for k in kinds:
        pat = HAS[k]
        if any((pat.search(c) for c in head)):
            got.append(k)
    for c in head:
        if all((HAS[k].search(c) for k in kinds)):
            hit_rows += 1
    return (tuple(got), hit_rows / len(head))

def coverage(query: str, content: str) -> float:
    qg = gramset(query)
    if not qg:
        q = (query or '').strip()
        return 1.0 if q and q in (content or '') else 0.0
    return len(qg & gramset(content or '')) / len(qg)

def judge(text: str, hits: list[dict], *, score_key: str='score') -> Judgment:
    if not hits:
        return Judgment(0.0, False, False, '一条都没查到', expects(text))
    want = expects(text)
    if want:
        got, ratio = satisfied(hits, want)
        if not got:
            return Judgment(0.0, False, False, '问的是' + '、'.join(want) + '，命中里没有这类东西', want, got, 0.0)
        base = len(got) / len(want)
        fok = min(1.0, base * 0.7 + ratio * 0.3)
        tot = fok < TOT_TOP_MAX and len(hits) >= TOT_MIN_HITS
        if tot:
            return Judgment(fok, True, False, '话到嘴边：沾边的不少，但' + '、'.join((k for k in want if k not in got)) + '答不上', want, got, fok)
        if fok < FOK_ESCALATE:
            return Judgment(fok, False, False, f'信心不够（{fok:.2f}）', want, got, fok)
        return Judgment(fok, False, True, f'够了（信心 {fok:.2f}）', want, got, fok)
    scores = [min(1.0, float(h.get(score_key) or 0.0)) for h in hits[:DEPTH]]
    top, var = (max(scores), _variance(scores))
    fok = min(0.8, top * 0.7 + (1.0 - min(var, 1.0)) * 0.3)
    tot = len(hits) >= TOT_MIN_HITS and top < TOT_TOP_MAX
    if tot:
        return Judgment(fok, True, False, '话到嘴边：一堆沾边的，没一条够强', (), (), top, var)
    if fok < FOK_ESCALATE:
        return Judgment(fok, False, False, f'信心不够（{fok:.2f}）', (), (), top, var)
    return Judgment(fok, False, True, f'够了（信心 {fok:.2f}，无明确期待）', (), (), top, var)
if __name__ == '__main__':
    from store.store import connect, search
    con = connect()
    cases = sys.argv[1:] or ['上次去海边是哪年的事', '专业课数据库占多少分', '室友弄丢钥匙那次换锁花了多少钱', '我喜欢什么颜色', '那场比赛在哪打的']
    for text in cases:
        from router import keywords
        terms = keywords(text)
        hits: list[dict] = []
        for t in terms:
            hits.extend(search(con, t, limit=6))
        j = judge(text, hits)
        print(f'{text}\n  词{terms} 命中{len(hits)}  期待{j.expects or '—'} 答上{j.answered or '—'}  信心{j.fok:.2f} → {j.reason}\n')
TIER_WEIGHT = {'primary': 1.0, 'secondary': 0.6, 'derived': 0.4}
STALE_DAYS = 180
RE_NOT_YET = re.compile('(还没|没有打|没打|尚未|还欠着|没写|未完成|还剩)')
RE_DONE = re.compile('(已经|打过|做完|写完了|搞定|完工|办完|通了|上线)')
NEAR = 14

def _says(content: str, topic: str, pat: re.Pattern) -> bool:
    text = content or ''
    for m in re.finditer(re.escape(topic), text):
        window = text[max(0, m.start() - NEAR):m.end() + NEAR]
        if pat.search(window):
            return True
    return False

@dataclass(frozen=True)
class Evidence:
    checked: int = 0
    expired: int = 0
    secondhand: int = 0
    unverified: int = 0
    conflicts: tuple[str, ...] = ()
    weight: float = 1.0
    stale: bool = False

    @property
    def clean(self) -> bool:
        return not self.conflicts and (not self.expired)

def check_evidence(con, hits: list[dict], *, depth: int=DEPTH, terms: list[str] | None=None) -> Evidence:
    head = hits[:depth]
    if not head:
        return Evidence()
    ids = [h.get('ext_id') for h in head if h.get('ext_id')]
    if not ids:
        return Evidence(checked=len(head))
    marks = ','.join('?' * len(ids))
    rows = con.execute(f"\n        SELECT f.ext_id, f.content, f.fact_type, f.valid_to, f.valid_from,\n               julianday('now') - julianday(f.valid_from) AS age,\n               MIN(COALESCE(t.tier,'primary')) AS tier,\n               MIN(COALESCE(e.verification_status,'unverified')) AS ver,\n               MAX(m.fact_key) AS fact_key, MAX(m.value) AS value,\n               MAX(COALESCE(m.polarity,'pos')) AS polarity\n        FROM facts f\n        LEFT JOIN fact_evidence fe ON fe.fact_id = f.id\n        LEFT JOIN evidence e ON e.id = fe.evidence_id\n        LEFT JOIN evidence_tier t ON t.evidence_id = e.id\n        LEFT JOIN fact_key_map m ON m.fact_id = f.id\n        WHERE f.ext_id IN ({marks})\n        GROUP BY f.ext_id\n    ", ids).fetchall()
    expired = sum((1 for r in rows if r['valid_to']))
    secondhand = sum((1 for r in rows if (r['tier'] or 'primary') != 'primary'))
    unverified = sum((1 for r in rows if (r['ver'] or '') != 'verified'))
    weights = [TIER_WEIGHT.get(r['tier'] or 'primary', 0.5) for r in rows]
    weight = sum(weights) / len(weights) if weights else 1.0
    stale = any(((r['age'] or 0) > STALE_DAYS and r['fact_type'] in ('semantic', 'procedural') for r in rows))
    conflicts: list[str] = []
    by_key: dict[str, set[str]] = {}
    pol_by_key: dict[str, set[str]] = {}
    for r in rows:
        if r['fact_key'] and r['value']:
            by_key.setdefault(r['fact_key'], set()).add(r['value'])
            pol_by_key.setdefault(r['fact_key'], set()).add(r['polarity'] or 'pos')
    for k, vals in by_key.items():
        if len(vals) > 1:
            conflicts.append(f'{k} 有两个值：{'／'.join(sorted(vals))}')
    for k, pols in pol_by_key.items():
        if len(pols) > 1:
            conflicts.append(f'{k} 一条肯定一条否定')
    for t in terms or []:
        if len(t) < 2:
            continue
        not_yet = [r for r in rows if _says(r['content'], t, RE_NOT_YET)]
        done = [r for r in rows if _says(r['content'], t, RE_DONE)]
        if not_yet and done:
            conflicts.append(f'「{t}」上一边说还没做、一边说已经做了（按时间新的那条为准，别挑顺嘴的）')
            break
    return Evidence(len(head), expired, secondhand, unverified, tuple(conflicts), round(weight, 3), stale)

def judge_with_evidence(con, text: str, hits: list[dict], *, score_key: str='score', terms: list[str] | None=None) -> tuple[Judgment, Evidence]:
    j = judge(text, hits, score_key=score_key)
    if not hits:
        return (j, Evidence())
    ev = check_evidence(con, hits, terms=terms)
    if ev.conflicts:
        return (Judgment(min(j.fok, 0.35), j.tot, False, '查到的东西自己打架：' + '；'.join(ev.conflicts), j.expects, j.answered, j.top, j.spread), ev)
    if ev.checked and ev.expired == ev.checked:
        return (Judgment(0.0, j.tot, False, '答上的那几条全是已作废的', j.expects, j.answered, j.top, j.spread), ev)
    if ev.weight < 1.0:
        fok = round(j.fok * ev.weight, 3)
        enough = fok >= FOK_ESCALATE and j.enough
        tail = f'（来源不是一手，信心打到 {ev.weight} 折）'
        return (Judgment(fok, j.tot, enough, j.reason + tail, j.expects, j.answered, j.top, j.spread), ev)
    return (j, ev)
