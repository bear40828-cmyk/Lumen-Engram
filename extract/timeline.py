from __future__ import annotations
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router import keywords
ENV_FILE = os.environ.get('ENGRAM_ENV_FILE', '.env')
TIMEOUT_S = 30
NEG = re.compile('(不构成|未推翻|没有推翻|不算|仍然?成立|并不矛盾|不矛盾|补充|细化|一致)')
MAX_OLD = 6
PROMPT = '一个记忆库里已经有几条关于同一个人的旧记录。现在来了一条新记录。\n判断：新记录是不是让某一条旧记录**不再成立**了。\n\n只有这几种算「不再成立」：\n- 状态变了（在做→不做了、没人管→有人管了、打算A→改成B）\n- 旧说法被新说法推翻或更正了\n- 同一个问题有了新的答案（比如「用户用什么手机」换了）\n\n这些**不算**，两条都保留：\n- 新记录只是补充、细化、确认旧记录\n- 同一个话题下的另一件事（想用 A 写小说 / 想试 B 模型——是两件事）\n- 时间不同的两件事（上周去了医院、这周又去了医院）\n- 两个人说的两句不同的话（助手B 先说了 X、后来又说了 Y，两句都是真的发生过）\n- 旧记录是一整串清单，新记录只改了其中一项（会顶掉整张清单，不许）\n- 拿不准\n\n判「不再成立」之前，先写出新旧两条**回答的是同一个问题**是什么（比如「猫谁来喂」「那个项目还做不做」）。\n写不出同一个问题，就是 null。\n\n新记录（{new_when}）：{new}\n\n旧记录：\n{olds}\n\n只输出 JSON：{{"same_question": "新旧共同回答的那个问题，没有就空字符串", "replaces": 旧记录编号或 null, "why": "一句话"}}'

def _env() -> dict:
    out = {}
    try:
        for line in open(ENV_FILE, encoding='utf-8'):
            m = re.match('^\\s*([A-Z0-9_]+)\\s*=\\s*(.*)\\s*$', line)
            if m:
                out[m.group(1)] = m.group(2).strip('"\'')
    except OSError:
        pass
    return out

def candidates(con, subject: str, content: str, before: str | None) -> list[dict]:
    terms = [t for t in keywords(content, k=4) if len(t) >= 2]
    seen, out = (set(), [])
    for t in terms:
        table, match = ('facts_gram_fts', f'"{t}"') if len(t) == 2 else ('facts_fts', t)
        sql = f"SELECT f.id, f.ext_id, f.content, f.valid_from FROM {table} x\n                  JOIN facts f ON f.id = x.rowid\n                  WHERE {table} MATCH ? AND f.subject = ?\n                    AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL\n                    AND f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')"
        args: list = [match, subject]
        if before:
            sql += ' AND (f.valid_from IS NULL OR f.valid_from <= ?)'
            args.append(before)
        sql += ' ORDER BY f.valid_from DESC LIMIT 4'
        try:
            rows = con.execute(sql, args).fetchall()
        except Exception:
            continue
        for r in rows:
            if r['id'] not in seen:
                seen.add(r['id'])
                out.append(dict(r))
    return out[:MAX_OLD]

def judge(new: str, new_when: str, olds: list[dict]) -> tuple[int | None, str]:
    if not olds:
        return (None, '没有沾边的旧事')
    env = _env()
    key = os.environ.get('DIGEST_API_KEY') or env.get('DIGEST_API_KEY')
    url = os.environ.get('DIGEST_BASE_URL') or env.get('DIGEST_BASE_URL')
    model = os.environ.get('EXTRACT_MODEL') or env.get('EXTRACT_MODEL') or env.get('DIGEST_MODEL')
    if not (key and url and model):
        return (None, '没配模型')
    body = {'model': model, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': PROMPT.format(new=new, new_when=(new_when or '')[:10], olds='\n'.join((f'[{i}]（{(o.get('valid_from') or '')[:10]}）{o['content']}' for i, o in enumerate(olds))))}], 'temperature': 0, 'max_tokens': 150, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.load(resp)
        text = data['choices'][0]['message']['content'] or ''
        m = re.search('\\{.*\\}', text, re.S)
        got = json.loads(m.group(0) if m else text)
        idx = got.get('replaces')
        why = str(got.get('why', ''))
        q = str(got.get('same_question', '')).strip()
        if idx is None:
            return (None, why)
        if not q:
            return (None, '说不出新旧回答的同一个问题，不顶替')
        if NEG.search(why):
            return (None, f'理由自相矛盾，不顶替：{why}')
        why = f'{q}｜{why}'
        idx = int(idx)
        if not 0 <= idx < len(olds):
            return (None, '编号越界，当不顶替')
        return (idx, why)
    except Exception as e:
        return (None, f'调用失败：{e}')
