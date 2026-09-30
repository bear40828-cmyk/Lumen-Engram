from __future__ import annotations
import glob
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SESSIONS = Path.home() / '.claude/projects/-home-ubuntu-workspace-claude'
CANDIDATES = ROOT / 'corrections-candidates.jsonl'
OUT = ROOT / 'corrections.jsonl'
RE_CHANNEL = re.compile('<channel[^>]*\\bts="([^"]+)"[^>]*>(.*?)</channel>', re.S)
RE_CORRECT = re.compile('不是|不对|错了|记错|说反|搞错|弄错|纠正|作废|瞎说|乱说|胡说|编的|才不是|没有啊|哪有|我没说|不是这样|看反|想多|脑补')

def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join((x.get('text', '') for x in content if isinstance(x, dict)))
    return ''

def mine() -> None:
    seen, rows = (set(), [])
    for path in sorted(glob.glob(str(SESSIONS / '*.jsonl'))):
        prev_her, my_last = ('', '')
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for lineno, line in enumerate(fh, 1):
                if '"reply"' in line and '"tool_use"' in line:
                    try:
                        d = json.loads(line)
                        for b in d.get('message', {}).get('content', []):
                            if b.get('type') == 'tool_use' and b.get('name', '').endswith('reply'):
                                my_last = b.get('input', {}).get('text', '') or my_last
                    except Exception:
                        pass
                    continue
                if '"type":"user"' not in line or '<channel' not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                for ts, body in RE_CHANNEL.findall(_text(d.get('message', {}).get('content'))):
                    body = body.strip()
                    key = (ts, body[:80])
                    if RE_CORRECT.search(body) and my_last and (key not in seen):
                        seen.add(key)
                        rows.append({'ts': ts, 'source_path': path, 'line': lineno, 'her': body[:600], 'my_prev': my_last[:600], 'her_prev': prev_her[:300]})
                    prev_her = body
    with open(CANDIDATES, 'w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'候选 {len(rows)} 条 → {CANDIDATES.name}')
PROMPT = '下面每组是一段聊天：AI 先说了一句，用户接着回了一句。判断用户那句是不是在**纠正 AI 说错的事实**\n（AI 记错了、说反了、搞混了人/时间/事、编了没有的事、误解了用户的意思，用户指出来）。\n开玩笑、反问撒娇、讨论第三方、纠正别的 AI、对计划改主意，都不算。\n\n是纠正的，抽出：\n- wrong：AI 原来以为的（一句话，具体）\n- right：实际情况（一句话，具体，按用户说的）\n- topic：这件事的主题词，3~8 个字\n- ask：如果以后再聊到这件事，用户可能怎么问（一句自然的口语）\n\n{groups}\n\n只输出 JSON：{{"items": [{{"i": 编号, "is": true或false, "wrong": "", "right": "", "topic": "", "ask": ""}}, ...]}}，每组都要有。'

def classify(batch: int=8) -> None:
    import rerank
    import urllib.request
    env = rerank._env()
    rows = [json.loads(x) for x in open(CANDIDATES, encoding='utf-8')]
    done = {}
    if OUT.exists():
        for x in open(OUT, encoding='utf-8'):
            r = json.loads(x)
            done[r['ts'], r['her'][:80]] = r
    todo = [r for r in rows if (r['ts'], r['her'][:80]) not in done]
    judged = ROOT / 'corrections-judged.jsonl'
    if judged.exists():
        judged_keys = {tuple(json.loads(x)['key']) for x in open(judged, encoding='utf-8')}
        todo = [r for r in todo if (r['ts'], r['her'][:80]) not in judged_keys]
    print(f'待判 {len(todo)} 条')
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        groups = '\n\n'.join((f'[{i}]\nAI：{r['my_prev'][:400]}\n用户：{r['her'][:400]}' for i, r in enumerate(chunk)))
        body = {'model': env['DIGEST_MODEL'], 'temperature': 0, 'max_tokens': 1500, 'enable_thinking': False, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': PROMPT.format(groups=groups)}]}
        req = urllib.request.Request(env['DIGEST_BASE_URL'], data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {env['DIGEST_API_KEY']}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = json.load(resp)['choices'][0]['message']['content']
            items = json.loads(re.search('\\{.*\\}', text, re.S).group(0))['items']
        except Exception as e:
            print(f'  第 {s} 批失败，下次重跑补：{e}')
            continue
        by_i = {int(it.get('i', -1)): it for it in items}
        with open(judged, 'a', encoding='utf-8') as jf, open(OUT, 'a', encoding='utf-8') as of:
            for i, r in enumerate(chunk):
                it = by_i.get(i)
                if it is None:
                    continue
                jf.write(json.dumps({'key': [r['ts'], r['her'][:80]], 'is': bool(it.get('is'))}, ensure_ascii=False) + '\n')
                if it.get('is') and it.get('wrong') and it.get('right'):
                    r.update({k: it.get(k, '') for k in ('wrong', 'right', 'topic', 'ask')})
                    of.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  {min(s + batch, len(todo))}/{len(todo)}')
REFINE = '下面是 AI 被用户（用户）纠正过的记录。逐条判断 keep：\n这条纠正过几周后还有用吗？\n- 有用（keep=true）：关于用户本人（经历、家人、喜好、习惯、计划、学校、身体）、用户身边的人和宠物、\n  我们一起做过的东西（项目、工具、谁做的、怎么做的）、用户定下的规矩、发生过的事的时间和经过、AI 的身份设定。\n- 没用（keep=false）：聊天当下的误会（看错字、会错一句话的意思、以为用户在忙）、玩笑、\n  只对那一刻有意义的状态（额度还剩多少、用户这会儿在干嘛）。\n从严：拿不准的一律 false。「用户昨晚去哪了」「用户当时在哪个 app」「用户那句话指的是什么」这类一次性的，都是 false。\n判断标准：一个月后我在别的话题里重新提到这件事，还会不会再犯同样的错？会，才 keep。\nkeep 的，把 wrong/right 改写成第一人称：AI 是「我」，用户是「用户」，一句话，具体到能单独看懂。\n\n{items}\n\n只输出 JSON：{{"items": [{{"i": 编号, "keep": true或false, "wrong": "我以为……", "right": "其实……"}}, ...]}}，每条都要有。'

def refine(batch: int=10) -> None:
    import rerank
    import urllib.request
    env = rerank._env()
    rows = [json.loads(x) for x in open(OUT, encoding='utf-8')]
    todo = [i for i, r in enumerate(rows) if 'keep' not in r]
    print(f'待筛 {len(todo)} 条')
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        items = '\n'.join((f'[{k}] 主题：{rows[i]['topic']}｜AI 以为：{rows[i]['wrong']}｜实际：{rows[i]['right']}｜用户原话：{rows[i]['her'][:150]}' for k, i in enumerate(idx)))
        body = {'model': env['DIGEST_MODEL'], 'temperature': 0, 'max_tokens': 2000, 'enable_thinking': False, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': REFINE.format(items=items)}]}
        req = urllib.request.Request(env['DIGEST_BASE_URL'], data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {env['DIGEST_API_KEY']}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                text = json.load(resp)['choices'][0]['message']['content']
            got = {int(it['i']): it for it in json.loads(re.search('\\{.*\\}', text, re.S).group(0))['items']}
        except Exception as e:
            print(f'  第 {s} 批失败，下次重跑补：{e}')
            continue
        for k, i in enumerate(idx):
            it = got.get(k)
            if it is None:
                continue
            rows[i]['keep'] = bool(it.get('keep'))
            if rows[i]['keep'] and it.get('wrong') and it.get('right'):
                rows[i]['wrong'], rows[i]['right'] = (it['wrong'], it['right'])
        with open(OUT, 'w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  {min(s + batch, len(todo))}/{len(todo)}')
    print(f'留下 {sum((1 for r in rows if r.get('keep')))}/{len(rows)}')

def promote_all() -> None:
    from store.store import add_evidence, connect, promote
    con = connect()
    n = 0
    for x in open(OUT, encoding='utf-8'):
        r = json.loads(x)
        if not r.get('keep'):
            continue
        ev = add_evidence(con, source_kind='chat', source_actor='user', source_path=r['source_path'], source_locator=f'line:{r['line']}', content=r['her'], occurred_at=r['ts'])
        day = r['ts'][:10]
        content = f'【用户纠正过我·{day}·{r['topic']}】{r['wrong']}；{r['right']}'
        fid = promote(con, subject='user', fact_type='semantic', content=content, evidence_ids=[ev], status='confirmed', valid_from=r['ts'], importance=1.0, confidence=1.0, user_pinned=True)
        n += fid is not None
    con.commit()
    print(f'新写入 {n} 条纠正（已有的跳过）')
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    {'mine': mine, 'classify': classify, 'refine': refine, 'promote': promote_all}.get(cmd, lambda: print(__doc__))()
