from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cards import cheap_enough, discover, expand
from router import route
from telemetry import Run
from store.store import connect, search, set_stance, pin, stats, touch
WHO = {'user': '用户', 'assistant': '助手', 'agent_b': '助手B', 'agent_c': '助手C'}
TOOLS = [{'name': 'engram_recall', 'description': '查长期记忆。只返回当下有效的事实，作废的和我标过「不认」的不出现。两个字的词也能查。带 why=true 会附上原始文件和行号。', 'inputSchema': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': '查询词，至少两个字'}, 'who': {'type': 'string', 'enum': list(WHO), 'description': '只看谁说的：user=用户 assistant=我 agent_b=助手B agent_c=助手C'}, 'limit': {'type': 'integer', 'description': '最多几条，默认 8'}, 'why': {'type': 'boolean', 'description': '是否附出处（文件+行号+核对状态）'}}, 'required': ['query']}}, {'name': 'engram_route', 'description': '把用户这句原话交给 Router：它决定该不该翻记忆库、查什么词、读几条，并直接把召回结果带回来。用户抛软话或撒娇时会明确回「不查」。', 'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '用户这句话的原文'}, 'who': {'type': 'string', 'enum': list(WHO)}}, 'required': ['text']}}, {'name': 'engram_think', 'description': '分层召回（router-0.2）：先判这是哪类问题、该叫醒哪类记忆，查一层就停下来判「够不够」——问句要的是年份／数字／人／地点而命中里没有，就自己升级去查下一层。答不上会直说答不上，不硬凑。问过去的事、问偏好、问做法、问规矩，都用这个，别用 engram_recall。', 'inputSchema': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '用户这句话的原文'}, 'who': {'type': 'string', 'enum': list(WHO)}}, 'required': ['text']}}, {'name': 'engram_wm', 'description': '工作记忆：此刻手上是什么事（目标／在办／话题／状态／用户的情绪）。换窗口后第一件事就该看这个——上一个窗口还有什么没办完都在里面。不带参数是查看；带 kind+content 是记一条；带 done/drop 是关掉一条。', 'inputSchema': {'type': 'object', 'properties': {'kind': {'type': 'string', 'enum': ['goal', 'task', 'topic', 'state', 'emotion'], 'description': 'goal=目标 task=在办 topic=话题 state=临时状态 emotion=用户的情绪'}, 'content': {'type': 'string', 'description': '记什么'}, 'done': {'type': 'integer', 'description': '把这个编号标成办完了'}, 'drop': {'type': 'integer', 'description': '把这个编号标成不做了'}, 'snapshot': {'type': 'boolean', 'description': 'true 就给一段能直接贴进新窗口的现状'}}}}, {'name': 'engram_expand', 'description': '把某条记忆的全文读出来。发现层只给 40 字卡片，确认这条有用再展开——省上下文用的。', 'inputSchema': {'type': 'object', 'properties': {'ext_id': {'type': 'string'}}, 'required': ['ext_id']}}, {'name': 'engram_stats', 'description': '看长期记忆库现在有多少证据、多少事实、多少条是用户钉住的。', 'inputSchema': {'type': 'object', 'properties': {}}}, {'name': 'engram_stance', 'description': '对某条记忆表态：own 认下 / disown 不认 / suspend 悬置。原文不动，改主意就再表一次，历史留痕。', 'inputSchema': {'type': 'object', 'properties': {'ext_id': {'type': 'string'}, 'stance': {'type': 'string', 'enum': ['own', 'disown', 'suspend']}, 'reason': {'type': 'string'}}, 'required': ['ext_id', 'stance']}}, {'name': 'engram_pin', 'description': '钉住一条记忆（用户明确要求长期保留的）。钉住的不参与衰减和冷层迁移。', 'inputSchema': {'type': 'object', 'properties': {'ext_id': {'type': 'string'}, 'on': {'type': 'boolean', 'description': 'false 取消钉住'}}, 'required': ['ext_id']}}]

def do_recall(con, a: dict) -> str:
    rows = search(con, a['query'], subject=a.get('who'), limit=int(a.get('limit', 8)))
    if not rows:
        return '没查到。库里没有，或者查询词只有一个字（一个字不做召回）。'
    out = []
    for i, r in enumerate(rows, 1):
        line = f'{i}. {r['content']}'
        if r['user_pinned']:
            line += '  📌'
        out.append(line)
        if a.get('why'):
            for e in con.execute('SELECT e.source_path, e.source_locator, e.verification_status\n                   FROM fact_evidence fe JOIN evidence e ON e.id = fe.evidence_id\n                   JOIN facts f ON f.id = fe.fact_id WHERE f.ext_id=?', (r['ext_id'],)):
                mark = {'verified': '✓', 'uncertain': '?', 'broken': '✗'}.get(e[2], '·')
                out.append(f'   出处{mark} {Path(e[0]).name}:{e[1]}')
        out.append(f'   [{r['ext_id']}]')
    return '\n'.join(out)

def handle(method: str, params: dict, con) -> dict:
    if method == 'initialize':
        return {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'engram', 'version': '0.1.0'}}
    if method == 'tools/list':
        return {'tools': TOOLS}
    if method == 'tools/call':
        name, a = (params.get('name'), params.get('arguments') or {})
        if name == 'engram_recall':
            text = do_recall(con, a)
        elif name == 'engram_wm':
            import working
            if a.get('snapshot'):
                text = working.snapshot(con)
            elif a.get('done'):
                text = '已标办完' if working.close(con, int(a['done']), 'done') else '没有这条开着的'
            elif a.get('drop'):
                text = '已标不做了' if working.close(con, int(a['drop']), 'dropped') else '没有这条开着的'
            elif a.get('kind') and a.get('content'):
                wid = working.put(con, a['kind'], a['content'], confidence=0.9)
                text = f'记下{working.KIND_CN[a['kind']]} #{wid}：{a['content']}'
            else:
                items = working.current(con)
                text = '\n'.join((f'#{it.id} {working.KIND_CN[it.kind]}：{it.content}' for it in items)) or '工作记忆是空的'
        elif name == 'engram_think':
            from retrieve import TYPE_CN, retrieve_logged
            r = retrieve_logged(con, a['text'], who=a.get('who'))
            if not r.should_recall:
                text = f'不查：{r.reason}'
            else:
                lines = [f'[{r.task_type}] 词{r.terms}  顺序' + '→'.join((TYPE_CN.get(x, x) for x in r.order))]
                for i, st in enumerate(r.steps, 1):
                    lines.append(f'  {i}. {TYPE_CN.get(st.layer, st.layer)}层 命中 {st.hits} 条，信心 {st.fok:.2f} → {('停手' if st.enough else '升级')}（{st.reason}）')
                if not r.hits:
                    lines.append('查不到。别硬凑，这种就直接跟用户说没有。')
                for h in r.hits[:8]:
                    lines.append(f'  · [{h['subject']}] {h['content'][:120]}')
                    touch(con, h['ext_id'])
                con.commit()
                text = '\n'.join(lines)
        elif name == 'engram_route':
            d = route(a['text'], who=a.get('who'))
            if not d.should_recall:
                with Run(con, query=a['text'], decision='skip', reason=d.reason):
                    pass
                text = f'不查：{d.reason}'
            else:
                blocks = [f'查 {d.queries}（每个最多 {d.limit} 条）：{d.reason}']
                for q in d.queries:
                    with Run(con, query=a['text'], decision='recall', reason=d.reason, terms=[q]) as run:
                        hits = discover(con, q, subject=d.who, limit=d.limit)
                        run.mode = 'direct' if cheap_enough(hits) else 'card'
                        for i, h in enumerate(hits, 1):
                            run.add_hit(fact_id=h['fact_id'], rank=i, final_score=h['score'], salience_score=h['score'], layer=h.get('layer'), memory_type=h.get('fact_type'), snippet_chars=h['full_len'] if run.mode == 'direct' else len(h['show']), selected=run.mode == 'direct' or i <= 2)
                    blocks.append(f'── 「{q}」──')
                    if not hits:
                        blocks.append('  薄表里没有')
                        continue
                    if cheap_enough(hits):
                        for row in expand(con, [h['fact_id'] for h in hits]):
                            blocks.append(f'  · {row['content']}')
                        continue
                    for i, h in enumerate(hits, 1):
                        tag = '命中' if h['hit'] else '卡片'
                        blocks.append(f'  {i}.（{tag}）{h['show']}  （全文 {h['full_len']} 字，要全文用 engram_expand: {h['ext_id']}）')
                text = '\n'.join(blocks)
        elif name == 'engram_expand':
            row = con.execute('SELECT content, subject, status, layer FROM facts WHERE ext_id=?', (a['ext_id'],)).fetchone()
            text = row[0] if row else f'没有这条：{a['ext_id']}'
        elif name == 'engram_stats':
            s = stats(con)
            text = f'证据 {s['evidence']} 条（核对通过 {s['evidence_verified']}）\n事实 {s['facts_by_status']}\n当下有效 {s['live']}｜钉住 {s['pinned']}'
        elif name == 'engram_stance':
            set_stance(con, a['ext_id'], actor='assistant', stance=a['stance'], reason=a.get('reason'))
            con.commit()
            text = f'已表态：{a['ext_id']} → {a['stance']}（原文没动，历史留痕）'
        elif name == 'engram_pin':
            pin(con, a['ext_id'], a.get('on', True))
            con.commit()
            text = f'{('已钉住' if a.get('on', True) else '已取消钉住')}：{a['ext_id']}'
        else:
            text = f'没有这个工具：{name}'
        return {'content': [{'type': 'text', 'text': text}]}
    raise ValueError(f'未知方法 {method}')

def main() -> int:
    con = connect()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rid = req.get('id')
        if rid is None:
            continue
        try:
            resp = {'jsonrpc': '2.0', 'id': rid, 'result': handle(req.get('method', ''), req.get('params') or {}, con)}
        except Exception as exc:
            resp = {'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32603, 'message': str(exc)}}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + '\n')
        sys.stdout.flush()
    return 0
if __name__ == '__main__':
    sys.exit(main())
