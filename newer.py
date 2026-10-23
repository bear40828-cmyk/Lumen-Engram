from __future__ import annotations
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import keywords
ENV_FILE = os.environ.get('ENGRAM_ENV_FILE', '.env')
TIMEOUT_S = 6
MAX_NEWER = 5
STOP_GRAMS = {'用户', '克劳', '劳德', '我们', '觉得', '喜欢', '现在', '之前', '因为', '所以', '已经', '还是', '一个', '这个', '那个', '可以', '没有', '不是', '就是', '她的', '宝说', '时候', '什么', '自己'}
STOP = {'用户', '助手', '我们', '觉得', '喜欢', '说', '现在', '之前', '因为', '所以', '已经', '还是', '一个', '这个', '那个', '可以', '没有', '不是', '就是'}
PROMPT = '记忆库里有一条旧记录，和几条时间更晚的记录。判断：更晚的哪一条让旧记录说的事**不再成立**了。\n\n旧记录（{old_when}）：{old}\n\n更晚的记录：\n{news}\n\n只有这些算不再成立：状态变了（有→没了、在用→扔了、打算A→改成B）、说法被更正、同一个问题有了新答案。\n只是补充、细化、同一话题的另一件事，都不算。拿不准就答 null。\n\n只输出 JSON：{{"index": 编号或null, "why": "一句话"}}'

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

def _grams(text: str) -> set[str]:
    out = set()
    for run in re.findall('[\\u4e00-\\u9fff]+', text):
        out |= {run[i:i + 2] for i in range(len(run) - 1)}
    return {g for g in out if g not in STOP_GRAMS}

def later_siblings(con, hit: dict) -> list[dict]:
    when = hit.get('valid_from')
    subject = hit.get('subject')
    if not when or not subject:
        return []
    mine = _grams(hit.get('content') or '')
    if len(mine) < 3:
        return []
    rows = con.execute("SELECT ext_id, content, valid_from, subject FROM facts\n           WHERE subject = ? AND status IN ('candidate','confirmed') AND valid_to IS NULL\n             AND valid_from > datetime(?, '+12 hours')\n             AND content NOT LIKE '我：%' AND content NOT LIKE '助手（%'\n             AND content NOT LIKE '用户（%'", (subject, when)).fetchall()
    scored = []
    for r in rows:
        if (r['content'] or '').strip() == (hit.get('content') or '').strip():
            continue
        ov = len(mine & _grams(r['content'] or ''))
        if ov >= 3:
            scored.append((ov, r['valid_from'], dict(r)))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored[:MAX_NEWER]]

def superseded_by(con, hit: dict) -> dict | None:
    news = later_siblings(con, hit)
    if not news:
        return None
    env = _env()
    key = os.environ.get('DIGEST_API_KEY') or env.get('DIGEST_API_KEY')
    url = os.environ.get('DIGEST_BASE_URL') or env.get('DIGEST_BASE_URL')
    model = os.environ.get('RERANK_MODEL') or env.get('RERANK_MODEL') or env.get('DIGEST_MODEL')
    if not (key and url and model):
        return None
    body = {'model': model, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': PROMPT.format(old_when=(hit.get('valid_from') or '')[:10], old=(hit.get('content') or '')[:200], news='\n'.join((f'[{i}]（{(n['valid_from'] or '')[:10]}）{n['content'][:160]}' for i, n in enumerate(news))))}], 'temperature': 0, 'max_tokens': 80, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            text = json.load(resp)['choices'][0]['message']['content'] or ''
        m = re.search('\\{.*\\}', text, re.S)
        got = json.loads(m.group(0) if m else text)
        idx = got.get('index')
        if isinstance(idx, int) and 0 <= idx < len(news):
            return news[idx] | {'why': str(got.get('why') or '')[:60]}
    except Exception:
        pass
    return None
