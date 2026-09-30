from __future__ import annotations
import json
import os
import re
import urllib.request
ENV_FILE = os.environ.get('ENGRAM_ENV_FILE', '.env')
TIMEOUT_S = 8
MAX_CANDIDATES = 12
PER_CANDIDATE = 220
PROMPT = '你在替一个 AI 判断：用户刚说的这句话，要不要从它的长期记忆里翻出旧事来接，翻的话哪几条对得上。\n\n用户这句：\n{query}\n\n候选记忆（编号在方括号里）：\n{cands}\n\n先判 need：这句话是不是在提、在问、或者需要接上过去发生的某件具体的事（人、约定、以前说过的话、做过的东西）。\n撒娇、闲聊、情绪、对眼前事的指令，need=false。\n\n再给每条候选打 0~10 分，从严：\n- 9~10：直接回答了这句话，或者正是这句话在说的那件事\n- 7~8：同一件事的关键信息（时间、人名、决定、原因）\n- 4~6：同一个话题，但不是这句在说的那件事\n- 1~3：只是有字面上重合的词\n- 0：无关\n\n只输出 JSON，scores 用编号当键、每条都要有：{{"need": true或false, "scores": {{"0": 分, "1": 分, ...}}}}'

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

def judge(query: str, candidates: list[str]) -> dict | None:
    env = _env()
    key = os.environ.get('DIGEST_API_KEY') or env.get('DIGEST_API_KEY')
    url = os.environ.get('DIGEST_BASE_URL') or env.get('DIGEST_BASE_URL')
    model = os.environ.get('RERANK_MODEL') or env.get('RERANK_MODEL') or env.get('DIGEST_MODEL')
    if not (key and url and model and candidates):
        return None
    cands = candidates[:MAX_CANDIDATES]
    body = {'model': model, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': PROMPT.format(query=query, cands='\n'.join((f'[{i}] {c[:PER_CANDIDATE]}' for i, c in enumerate(cands))))}], 'temperature': 0, 'max_tokens': 120, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.load(resp)
        text = data['choices'][0]['message']['content'] or ''
        m = re.search('\\{.*\\}', text, re.S)
        got = json.loads(m.group(0) if m else text)
        raw = got.get('scores', {})
        if isinstance(raw, dict):
            scores = [int(raw.get(str(i), 0) or 0) for i in range(len(cands))]
        else:
            raw = [int(x or 0) for x in raw][:len(cands)]
            scores = raw + [0] * (len(cands) - len(raw))
        return {'need': bool(got.get('need')), 'scores': scores}
    except Exception:
        return None
