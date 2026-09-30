from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import metamem
import working
from router import Decision, keywords, route
from store.store import connect, search
L1: list[tuple[str, str, tuple[str, ...]]] = [('过去的具体事', '问某件事什么时候、在哪、怎么发生的', ('episodic', 'semantic')), ('偏好口味', '问喜欢什么、不喜欢什么、爱吃什么', ('preference', 'semantic', 'episodic')), ('怎么做', '问操作步骤、命令、流程、以前是怎么弄的', ('procedural', 'semantic', 'episodic')), ('稳定事实', '问数字、名称、身份、结论这类应该已经定下来的', ('semantic', 'episodic')), ('规矩要求', '问用户定过的规矩、格式要求、说过不许怎样', ('procedural', 'semantic')), ('不查', '软话、撒娇、单纯叫人——这时候要的是接住，不是资料', ())]
import re
CUES: dict[str, re.Pattern] = {'偏好口味': re.compile('(喜欢|爱吃|讨厌|不喜欢|受不了|口味|偏好|最想|爱喝)'), '怎么做': re.compile('(怎么弄|怎么做|怎么搞|步骤|流程|命令|脚本|装|配置|部署|怎么办|上次是怎么|做法)'), '规矩要求': re.compile('(规矩|要求|不许|不准|别再|说过要|讲过要|以后都|一律)'), '稳定事实': re.compile('(是谁|叫什么|多少|几个|几点|哪个学校|专业|多大|生日|\\d{2,}|%|占比)'), '过去的具体事': re.compile('(上次|上回|之前|以前|当时|那天|昨天|前天|哪年|哪天|什么时候|多久|后来|结果)')}
DEFAULT_ORDER = ('episodic', 'semantic', 'preference', 'procedural')
RARE_MIN, RARE_MAX = (1, 60)
CHATTER = re.compile('^(好?无聊|好累|好困|好烦|真的吗|怎么办|随便|不知道|没什么|还行|挺好|要不|算了|行吧|好吧|再说|然后呢)$')
GOAL_STRONG = 0.25
GOAL_WEIGHT = 0.5
MAX_EMPTY = 2
TYPE_CN = {'episodic': '情景', 'semantic': '语义', 'preference': '偏好', 'procedural': '程序'}

@dataclass
class Step:
    layer: str
    hits: int
    fok: float
    tot: bool
    enough: bool
    reason: str
    expects: tuple[str, ...] = ()
    answered: tuple[str, ...] = ()

@dataclass
class Result:
    should_recall: bool
    depth: str = 'skip'
    goal_rel: float = 0.0
    goal_why: str = ''
    conflicts: tuple[str, ...] = ()
    terms: list[str] = field(default_factory=list)
    task_type: str = ''
    order: tuple[str, ...] = ()
    hits: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    reason: str = ''
    escalations: int = 0

    @property
    def layers_read(self) -> list[str]:
        return [s.layer for s in self.steps]

def classify_task(text: str) -> tuple[str, tuple[str, ...]]:
    for name, _boundary, order in L1:
        if not order:
            continue
        pat = CUES.get(name)
        if pat and pat.search(text):
            return (name, order)
    return ('泛问', DEFAULT_ORDER)

def rarity(con, term: str) -> int:
    try:
        if len(term) == 2:
            return con.execute('SELECT COUNT(*) FROM facts_gram_fts WHERE facts_gram_fts MATCH ?', (f'"{term}"',)).fetchone()[0]
        return con.execute('SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?', (term,)).fetchone()[0]
    except Exception:
        return 10 ** 6

def windows(term: str) -> list[str]:
    out = [term]
    for n in (4, 3, 2):
        if len(term) > n:
            out += [term[i:i + n] for i in range(len(term) - n + 1)]
    seen, uniq = (set(), [])
    for t in out:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq
MAX_WINDOWS = 3
TOO_COMMON = 300

def usable_forms(con, term: str) -> list[tuple[str, int]]:
    out = []
    for t in windows(term):
        n = rarity(con, t)
        if n and n <= TOO_COMMON:
            out.append((t, n))
    out.sort(key=lambda x: (x[1], -len(x[0])))
    return out[:MAX_WINDOWS]

def best_form(con, term: str) -> tuple[str, int]:
    forms = usable_forms(con, term)
    return forms[0] if forms else (term, 0)

def by_rarity(con, terms: list[str]) -> list[str]:
    terms = [t for t in terms if best_form(con, t)[1] <= TOO_COMMON] or terms
    forms = [best_form(con, t) for t in terms]
    return [t for t, n in sorted(forms, key=lambda x: (x[1] == 0, x[1], -len(x[0])))]
DISTILLED = 'distilled'

def probe(con, term: str, *, layer: str, limit: int) -> list[dict]:
    if layer == DISTILLED:
        return search(con, term, limit=limit, extracted_only=True)
    hits = search(con, term, limit=limit, fact_types=[layer])
    if hits or len(term) <= 3:
        return hits
    out, seen = ([], set())
    for t, _n in usable_forms(con, term):
        for h in search(con, t, limit=limit, fact_types=[layer]):
            if h['ext_id'] not in seen:
                seen.add(h['ext_id'])
                out.append(h)
    return out[:limit * 2]

def retrieve(con, text: str, *, who: str | None=None, per_layer: int=6, hard_cap: int=12) -> Result:
    d: Decision = route(text, who=who)
    goal_rel, goal_why = working.relevance(con, text)
    if not d.should_recall:
        vetoed = d.reason.startswith('软话') or d.reason.startswith('这个我本来')
        vote = None
        if not vetoed:
            rare = [t for t in keywords(text) if not CHATTER.search(t) and RARE_MIN <= best_form(con, t)[1] <= RARE_MAX]
            if rare:
                vote = Decision(True, keywords(text), who, limit=3, depth='light', reason=f'聊到了具体的旧事（{'、'.join(rare)}），自己翻一眼')
        if vote is None and (not vetoed) and (goal_rel >= GOAL_STRONG):
            vote = Decision(True, keywords(text), who, limit=4, depth='deep', reason=f'跟手上的事有关（{goal_why}），工作记忆要求查')
        if vote is None:
            return Result(False, depth='skip', goal_rel=goal_rel, goal_why=goal_why, reason=d.reason, task_type='不查')
        d = vote
    terms = by_rarity(con, d.queries or keywords(text))
    task, order = classify_task(text)
    if goal_rel >= GOAL_STRONG and goal_why:
        lead = 'procedural' if goal_why.startswith('在办') else 'semantic'
        order = (lead,) + tuple((x for x in order if x != lead))
    order = (DISTILLED,) + tuple(order)
    if d.depth == 'light':
        order = order[:2]
        per_layer = min(per_layer, 3)
    res = Result(True, depth=d.depth, goal_rel=goal_rel, goal_why=goal_why, terms=terms, task_type=task, order=order, reason=d.reason)
    seen: dict[str, dict] = {}
    empty_streak = 0
    for layer in order:
        got: list[dict] = []
        for rank, t in enumerate(terms):
            for h in probe(con, t, layer=layer, limit=per_layer):
                if h['ext_id'] not in seen:
                    h['term_rank'] = rank
                    seen[h['ext_id']] = h
                    got.append(h)
        j, ev = metamem.judge_with_evidence(con, text, got, terms=terms)
        if ev.conflicts:
            res.conflicts = tuple(dict.fromkeys(res.conflicts + ev.conflicts))
        res.steps.append(Step(layer, len(got), j.fok, j.tot, j.enough, j.reason, j.expects, j.answered))
        res.hits.extend(got)
        if j.enough:
            break
        res.escalations += 1
        empty_streak = empty_streak + 1 if not got else 0
        if empty_streak >= MAX_EMPTY:
            res.steps.append(Step('停手', 0, 0.0, False, False, f'连着 {MAX_EMPTY} 层空手，不再往下翻'))
            break
        if len(res.hits) >= hard_cap:
            break
    if d.depth == 'light':
        res.escalations = 0
    if goal_rel >= GOAL_STRONG and goal_why:
        anchor = goal_why.split('「', 1)[-1].rstrip('」')
        for h in res.hits:
            h['goal_boost'] = round(GOAL_WEIGHT * working._overlap(h.get('content') or '', anchor), 4)
    res.hits = sorted(res.hits, key=lambda h: (h.get('term_rank', 9), -(float(h.get('score') or 0) + float(h.get('goal_boost') or 0))))[:hard_cap]
    return res

def log_recall(con, text: str, r: Result, *, decision: str | None=None, reason: str | None=None, selected: 'set[str] | tuple[str, ...] | None'=None) -> int | None:
    import json
    from telemetry import Run
    if decision is None:
        decision = 'recall' if r.should_recall else 'skip'
    if reason is None:
        reason = r.reason
    with Run(con, query=text, decision=decision, reason=reason, terms=r.terms, mode='layered') as run:
        for i, h in enumerate(r.hits, 1):
            fid = con.execute('SELECT id FROM facts WHERE ext_id=?', (h['ext_id'],)).fetchone()
            if not fid:
                continue
            run.add_hit(fact_id=fid[0], rank=i, final_score=float(h.get('score') or 0), salience_score=float(h.get('score') or 0), layer=h.get('layer'), memory_type=h.get('fact_type'), snippet_chars=len(h.get('content') or ''), selected=h['ext_id'] in selected if selected is not None else i <= 3)
    if run.run_id:
        for i, st in enumerate(r.steps, 1):
            con.execute('INSERT OR IGNORE INTO recall_paths\n                     (run_id, step_no, task_type, layer, hits, fok, tot, enough,\n                      expects, answered, reason)\n                   VALUES (?,?,?,?,?,?,?,?,?,?,?)', (run.run_id, i, r.task_type, st.layer, st.hits, st.fok, int(st.tot), int(st.enough), json.dumps(list(st.expects), ensure_ascii=False), json.dumps(list(st.answered), ensure_ascii=False), st.reason))
        con.commit()
    return run.run_id

def retrieve_logged(con, text: str, *, who: str | None=None, **kw) -> Result:
    r = retrieve(con, text, who=who, **kw)
    log_recall(con, text, r)
    return r

def explain(r: Result) -> str:
    if not r.should_recall:
        return f'不查：{r.reason}'
    head = f'[{r.task_type}·{r.depth}] 词{r.terms} 顺序{'→'.join((TYPE_CN.get(x, x) for x in r.order))}'
    if r.goal_rel:
        head += f'\n  手上的事：{r.goal_why}（相关度 {r.goal_rel}）'

    def line(i: int, s: Step) -> str:
        if s.layer == '停手':
            return f'  {i + 1}. 停手：{s.reason}'
        return f'  {i + 1}. {TYPE_CN.get(s.layer, s.layer)}层：命中 {s.hits} 条，信心 {s.fok:.2f}{('，话到嘴边' if s.tot else '')} → {('停手' if s.enough else '升级')}（{s.reason}）'
    body = '\n'.join((line(i, st) for i, st in enumerate(r.steps)))
    tail = f'  共读 {len(r.steps)} 层，升级 {r.escalations} 次，返回 {len(r.hits)} 条'
    if r.conflicts:
        tail += '\n  ⚠️ 打架：' + '；'.join(r.conflicts) + '\n     别挑一条顺嘴的说，要么按时间取新的那条并说清，要么直接问用户。'
    return f'{head}\n{body}\n{tail}'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('text', nargs='*')
    ap.add_argument('--trace', action='store_true', help='连命中的内容一起打出来')
    ap.add_argument('--log', action='store_true', help='把这次召回记进遥测')
    args = ap.parse_args()
    samples = ['在吗', '你好', '好无聊', '上次开会是哪天', '第三章占多少分', '你还记得我说过同事把钥匙弄丢了吗', '帮我把那个脚本跑一下', '我喜欢红色，不太喜欢绿色', '今天吃什么']
    con = connect()
    runner = retrieve_logged if args.log else retrieve
    for text in args.text or samples:
        r = runner(con, text)
        print(f'\n{text!r}\n{explain(r)}')
        if args.trace:
            for h in r.hits[:6]:
                print(f'     · [{h['subject']}] {h['content'][:56]}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
