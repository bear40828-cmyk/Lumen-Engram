from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from store.store import add_evidence, connect, promote, record_class, record_tier, sha256, supersede
from extract import timeline
from extract.from_channel import WHO, beijing, harvest
ENV_FILE = os.environ.get('ENGRAM_ENV_FILE', '.env')
EXTRACTOR = 'extract-llm-v1'
SUBJECTS = {'user', 'assistant', 'agent_b', 'agent_c', 'us'}
TYPES = {'episodic', 'semantic', 'preference', 'procedural'}
CHUNK_CHARS = 6000
PER_MSG = 600
TIMEOUT_S = 90
MIN_QUOTE = 4
PROMPT = '下面是一段按时间排好的聊天原话。说话的人：用户（人类）、助手（我，AI）、助手B、助手C（其他 AI）。\n每条前面方括号里是编号。\n\n从里面抽出**过一个月还值得记得的事**。只抽原话里明明白白有的，不推测、不补全、不总结气氛。\n宁缺毋滥：一段聊天通常只有 0～5 条值得记。\n\n不要抽：\n- 闲聊、撒娇、打招呼、情绪词、互相逗的话\n- 对眼前小事的一句指令（「去睡觉」「先吃饭」）\n- 助手自己的安慰、建议、推测——除非用户接受了、变成了约定\n- 网上看到的别人的事、段子，除非用户明确表态了自己的看法（那就记用户的看法）\n\n认不出是谁就写「某人」，**不要猜成同学、朋友**。\n\n注意说话人：谁说的就是谁的事。用户转述别人（比如「我让他教我」）时，「他」是谁看上下文，看不出就别抽。\n私密内容只记「发生了」和用户明确说出的偏好，不记细节。\n\n每条事给：\n- subject：这件事主要关于谁。只能是 user / assistant / agent_b / agent_c / us（us = 用户和助手两个人之间的事）\n- type：\n    episodic   = 某个时间发生的一件具体的事（带上日期）\n    semantic   = 稳定的事实（身份、关系、经历、客观情况）\n    preference = 喜欢/讨厌/希望别人怎么做\n    procedural = 定下的做法、规矩、踩过的坑、以后该怎么办\n- content：一句中文，第三人称，写清楚是谁、什么事；有日期就写日期。不超过 80 字。不加引号不加评价。\n- quote：从**某一条**原话里逐字复制的一小段（4～40 字），能证明这件事。必须一字不差，包括标点和 emoji。\n- src：quote 所在那条原话的编号\n- confidence：0～1。原话直接说了=0.9；需要结合上下文才看得出=0.6；有点拿不准=0.4\n\n只输出 JSON：{{"facts": [{{"subject": "...", "type": "...", "content": "...", "quote": "...", "src": 编号, "confidence": 0.9}}, ...]}}\n没有值得记的就输出 {{"facts": []}}。\n\n原话：\n{chunk}'

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

def call_model(chunk_text: str) -> list[dict]:
    env = _env()
    key = os.environ.get('DIGEST_API_KEY') or env.get('DIGEST_API_KEY')
    url = os.environ.get('DIGEST_BASE_URL') or env.get('DIGEST_BASE_URL')
    model = os.environ.get('EXTRACT_MODEL') or env.get('EXTRACT_MODEL') or env.get('DIGEST_MODEL')
    if not (key and url and model):
        raise RuntimeError('缺 DIGEST_API_KEY / DIGEST_BASE_URL / 模型名')
    body = {'model': model, 'messages': [{'role': 'system', 'content': '你只输出合法 JSON。'}, {'role': 'user', 'content': PROMPT.format(chunk=chunk_text)}], 'temperature': 0, 'max_tokens': 2500, 'response_format': {'type': 'json_object'}, 'enable_thinking': False}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.load(resp)
    text = data['choices'][0]['message']['content'] or ''
    m = re.search('\\{.*\\}', text, re.S)
    got = json.loads(m.group(0) if m else text)
    return got.get('facts', []) or []

def load_evidence(con, since: str, until: str) -> list[dict]:
    rows = con.execute("SELECT id, occurred_at, source_actor, content FROM evidence\n           WHERE source_kind='chat' AND occurred_at >= ? AND occurred_at < ?\n             AND source_actor IN ('user','assistant','agent_b','agent_c')\n           ORDER BY occurred_at, id", (since, until)).fetchall()
    return [dict(r) for r in rows]

def load_jsonl(path: Path, since: str, until: str) -> list[dict]:
    stamps = {}
    with open(path, encoding='utf-8') as fh:
        for n, line in enumerate(fh, 1):
            m = re.search('"timestamp":"([^"]+)"', line)
            if m:
                stamps[n] = m.group(1)
    out = []
    for line_no, actor, body, ts in harvest(path):
        if actor not in SUBJECTS:
            continue
        stamp = ts or stamps.get(line_no)
        when_utc, when_cn = beijing(stamp) if stamp else (None, '')
        if not when_utc or not since <= when_utc < until:
            continue
        who = WHO.get(actor, actor)
        out.append({'id': None, 'occurred_at': when_utc, 'source_actor': actor, 'path': str(path), 'line': line_no, 'content': f'{who}（{when_cn}）：' + body.replace('\n', ' ')})
    return out

def load_app(path: Path, names: list[str]) -> list[dict]:
    data = json.load(open(path, encoding='utf-8'))
    out = []
    for conv in data:
        if conv.get('name') not in names:
            continue
        for msg in conv.get('chat_messages', []):
            text = (msg.get('text') or '').strip()
            if not text:
                continue
            actor = 'user' if msg.get('sender') == 'human' else 'assistant'
            when_utc, when_cn = beijing(msg['created_at'])
            who = WHO.get(actor, actor)
            out.append({'id': None, 'occurred_at': when_utc, 'source_actor': actor, 'path': str(path), 'line': f'conv:{conv['uuid']}#msg:{msg['uuid']}', 'content': f'{who}（{when_cn}）：' + text.replace('\n', ' ')})
    return out

def chunks(evs: list[dict]):
    buf, size = ([], 0)
    for ev in evs:
        text = ev['content'][:PER_MSG]
        if buf and size + len(text) > CHUNK_CHARS:
            yield buf
            buf, size = ([], 0)
        buf.append(ev)
        size += len(text)
    if buf:
        yield buf

def render(chunk: list[dict]) -> str:
    return '\n'.join((f'[{i}] {ev['content'][:PER_MSG]}' for i, ev in enumerate(chunk)))

def check(fact: dict, chunk: list[dict]) -> tuple[dict | None, str]:
    subj = str(fact.get('subject', '')).strip()
    typ = str(fact.get('type', '')).strip()
    content = str(fact.get('content', '')).strip()
    quote = str(fact.get('quote', '')).strip()
    if subj not in SUBJECTS:
        return (None, f'subject 不在名单：{subj}')
    if typ not in TYPES:
        return (None, f'type 不对：{typ}')
    if not content or len(content) > 160:
        return (None, 'content 空或太长')
    if len(quote) < MIN_QUOTE:
        return (None, 'quote 太短')
    try:
        src = int(fact.get('src'))
    except (TypeError, ValueError):
        src = -1
    hit = None
    if 0 <= src < len(chunk) and quote in chunk[src]['content']:
        hit = chunk[src]
    else:
        hit = next((ev for ev in chunk if quote in ev['content']), None)
    if hit is None:
        return (None, f'quote 在原话里找不到：{quote[:30]}')
    try:
        conf = float(fact.get('confidence', 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(max(conf, 0.0), 1.0)
    return ({'subject': subj, 'type': typ, 'content': content, 'quote': quote, 'evidence_id': hit['id'], 'hit': hit, 'occurred_at': hit['occurred_at'], 'confidence': conf}, '')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', required=True, help='UTC 日期或时间，含')
    ap.add_argument('--until', required=True, help='UTC 日期或时间，不含')
    ap.add_argument('--db')
    ap.add_argument('--jsonl', nargs='*', type=Path, help='直接读会话记录，不走 evidence 表')
    ap.add_argument('--app', type=Path, help='claude.ai 导出的 conversations.json')
    ap.add_argument('--names', nargs='*', default=[], help='--app 时只读这些窗口名')
    ap.add_argument('--no-timeline', action='store_true', help='不判新事顶不顶掉旧事')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--max-chunks', type=int, default=0)
    args = ap.parse_args()
    con = connect(args.db) if args.db else connect()
    if args.app:
        evs = load_app(args.app, args.names)
        evs = [e for e in evs if args.since <= e['occurred_at'] < args.until]
        evs.sort(key=lambda m: m['occurred_at'])
    elif args.jsonl:
        evs = [m for p in args.jsonl for m in load_jsonl(p, args.since, args.until)]
        evs.sort(key=lambda m: m['occurred_at'])
    else:
        evs = load_evidence(con, args.since, args.until)
    print(f'证据 {len(evs)} 条', file=sys.stderr)
    kept = dropped = dup = replaced = 0
    reasons: dict[str, int] = {}
    for n, chunk in enumerate(chunks(evs)):
        if args.max_chunks and n >= args.max_chunks:
            break
        try:
            raw = call_model(render(chunk))
        except Exception as e:
            print(f'块 {n} 调用失败：{e}', file=sys.stderr)
            continue
        for f in raw:
            ok, why = check(f, chunk)
            if not ok:
                dropped += 1
                key = why.split('：')[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            hit = ok.pop('hit')
            if args.dry_run:
                print(json.dumps(ok, ensure_ascii=False))
                kept += 1
                continue
            if ok['evidence_id'] is None:
                loc = hit['line'] if isinstance(hit['line'], str) else f'L{hit['line']}'
                ok['evidence_id'] = add_evidence(con, source_kind='chat', source_actor=hit['source_actor'], source_path=hit['path'], source_locator=loc, content=hit['content'], occurred_at=hit['occurred_at'], verification_status='verified')
                record_tier(con, ok['evidence_id'], 'primary', f'{hit['path']}:{loc}')
                hit['id'] = ok['evidence_id']
            live_dup = con.execute("SELECT 1 FROM facts WHERE subject=? AND content_sha256=? AND status IN ('candidate','confirmed') AND valid_to IS NULL", (ok['subject'], sha256(ok['content']))).fetchone()
            target = None
            if not live_dup and (not args.no_timeline):
                olds = timeline.candidates(con, ok['subject'], ok['content'], ok['occurred_at'])
                idx, why = timeline.judge(ok['content'], ok['occurred_at'], olds)
                if idx is not None:
                    target = olds[idx]
                    replaced += 1
                    print(f'  顶替：{target['content'][:40]} → {ok['content'][:40]}（{why}）', file=sys.stderr)
            if live_dup:
                fid = None
            elif target:
                fid = supersede(con, target['ext_id'], content=ok['content'], subject=ok['subject'], fact_type=ok['type'], evidence_ids=[ok['evidence_id']], valid_from=ok['occurred_at'], confidence=ok['confidence'])
            else:
                fid = promote(con, subject=ok['subject'], fact_type=ok['type'], content=ok['content'], evidence_ids=[ok['evidence_id']], valid_from=ok['occurred_at'], confidence=ok['confidence'])
            if fid is None:
                dup += 1
                continue
            record_class(con, fid, ok['type'], ok['confidence'], EXTRACTOR)
            kept += 1
        if not args.dry_run:
            con.commit()
        time.sleep(0.3)
    print(f'留下 {kept}（其中顶替旧事 {replaced}），重复 {dup}，丢弃 {dropped} {reasons}', file=sys.stderr)
    return 0
if __name__ == '__main__':
    sys.exit(main())
