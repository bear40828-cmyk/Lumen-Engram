from __future__ import annotations
import json
import os
import pathlib
import re
import sys
import time

def _alt(var: str) -> str:
    terms = [re.escape(t.strip()) for t in os.environ.get(var, '').split(',') if t.strip()]
    return '|'.join(terms) + '|' if terms else ''
ENTITY_ALT = _alt('ENGRAM_ENTITY_TERMS')
TOPIC_ALT = _alt('ENGRAM_TOPIC_TERMS')
if '#cron:' in os.environ.get('CC_SESSION_KEY', ''):
    raise SystemExit(0)
ENGRAM = os.environ.get('ENGRAM_HOME', str(pathlib.Path(__file__).resolve().parent))
LOG = os.path.join(ENGRAM, 'hook.log')
MAX_CHARS = 420
MAX_HITS = 2
PER_HIT = 140
MAX_AUTO_INPUT = 700
STATUS_MARK = {'candidate': '待核', 'confirmed': '确认', 'uncertain': '存疑', 'superseded': '已被取代', 'invalid': '已作废'}
RE_CHANNEL = re.compile('<channel[^>]*>(?P<body>.*?)</channel>', re.S)
RE_TELEGRAM = re.compile('\\[TELEGRAM_EXTERNAL_MESSAGE\\].*?original_body=(?P<q>[\\"\'])(?P<body>.*?)(?P=q).*?\\[/TELEGRAM_EXTERNAL_MESSAGE\\]', re.S)
SYNTHETIC = re.compile('^\\s*(?:<task-notification>|<system-reminder>|\\[tool|\\[document|\\(document:)', re.I)
EXPLICIT_RECALL = re.compile('(上次|上回|之前|以前|当时|那天|昨天|前天|还记得|记不记得|你说过|我说过|我跟你说过|原话|查一下|翻一下|翻翻)')
REINJECT_COOLDOWN_MIN = 30
SENT_END = '。！？；!?;\n'

def clip(text: str, limit: int=PER_HIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max((head.rfind(c) for c in SENT_END))
    if cut >= limit * 0.5:
        head = head[:cut + 1]
    return head.rstrip() + '……（后文略，未完，别据此下结论；要全文按 id 查）'

def recently_injected(con, minutes: int=REINJECT_COOLDOWN_MIN) -> set[str]:
    try:
        rows = con.execute("SELECT DISTINCT f.ext_id\n                 FROM recall_hits h\n                 JOIN recall_runs r ON r.id = h.run_id\n                 JOIN facts f ON f.id = h.fact_id\n                WHERE h.selected_for_context = 1\n                  AND r.router_decision = 'recall'\n                  AND datetime(r.created_at) >= datetime('now', ?)\n                  AND f.status IN ('candidate','confirmed')\n                  AND f.valid_to IS NULL", (f'-{int(minutes)} minutes',))
        return {row[0] for row in rows}
    except Exception as e:
        log(f'去重查询失败（按不去重处理）：{e}')
        return set()

def log(msg: str) -> None:
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(f'{time.strftime('%F %T')} {msg}\n')
    except Exception:
        pass

def incoming(data: dict) -> str:
    text = data.get('prompt') or ''
    bodies = [m.group('body').strip() for m in RE_CHANNEL.finditer(text)]
    if bodies:
        return bodies[-1]
    telegram = list(RE_TELEGRAM.finditer(text))
    return (telegram[-1].group('body') if telegram else text).strip()

def skip_input(text: str) -> str | None:
    if SYNTHETIC.search(text):
        return '系统／工具消息'
    if len(text) > MAX_AUTO_INPUT and (not EXPLICIT_RECALL.search(text)):
        return '长篇粘贴且未要求回忆'
    return None
DISTINCTIVE = re.compile('([0-9]{1,2}\\s*月\\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}|' + ENTITY_ALT + '[A-Za-z][A-Za-z0-9_.\\-]{3,}|' + TOPIC_ALT + '记忆库|交接|日记|看板|唤醒|额度|封号)')
STRONG = re.compile('[0-9]{1,2}\\s*月\\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}' + ('|' + ENTITY_ALT.rstrip('|') if ENTITY_ALT else ''))
TOO_COMMON = {'claude', 'code', 'http', 'https', 'json', 'text', 'post', 'photo', 'document'}
QUESTION = re.compile('[？?]|吗|来着|到底|是不是|谁|哪个|哪里|什么时候|第一次|最开始|怎么回事')
RELAY_SPEAKER = re.compile('^\\s*(?:agent_b|助手|agent_b|助手B|助手C|deepseek|gemini|grok)\\s*[：:]', re.I)
PLAYFUL_RELAY = re.compile('(床上|推屁股|打屁股|捏|涩涩|调情|玩笑|黄梗|🌚|🤡|呗[？?]?$)')
LOW_INFO_REPLY = re.compile('^\\s*(?:做?好了|弄好了|完成了|可以了|现在呢|然后呢|看到了?|就是这个|对[的吧]?|嗯+|行了?|好啦)[呀啊吧呢嘛~～。！!？?\\s\\U0001F300-\\U0001FAFF]*$')
IMAGE_MARK = re.compile('(?:image_path\\s*=|<image\\b|\\[image\\]|\\[图片\\])', re.I)
CONTEXT_TURNS = 6
CONTEXT_CHARS = 900
ENTITY_GUARDS = ({'query_any': ('摄像头', '当眼睛', '随身挂件', '项链吊坠'), 'candidate_forbidden': '共感娃娃'},)

def pinned_candidates(con, text: str, k: int=3) -> list[dict]:
    try:
        rows = con.execute("SELECT id, ext_id, subject, content, status, user_pinned FROM facts WHERE user_pinned=1 AND status='confirmed' AND valid_to IS NULL").fetchall()
    except Exception:
        return []
    q = {text[i:i + 2] for i in range(len(text) - 1)} - {'其实', '我以', '以为'}
    scored = []
    for row in rows:
        c = row['content'] or ''
        overlap = sum((1 for g in q if g in c))
        if overlap >= 2:
            scored.append((overlap, dict(row)))
    scored.sort(key=lambda x: -x[0])
    return [d | {'fact_id': d['id']} for _, d in scored[:k]]

def distinctive_terms(text: str) -> list[str]:
    found = []
    for m in DISTINCTIVE.finditer(text):
        w = m.group(1)
        if w.lower() in TOO_COMMON:
            continue
        if w not in found:
            found.append(w)
    return found

def pre_recall_skip(text: str) -> str | None:
    if RELAY_SPEAKER.search(text) and PLAYFUL_RELAY.search(text) and (not EXPLICIT_RECALL.search(text)):
        return '转述／玩笑，不翻旧账'
    return None

def _entry_text(entry: dict) -> str:
    msg = entry.get('message')
    if not isinstance(msg, dict):
        return ''
    content = msg.get('content')
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join((block.get('text', '') for block in content if isinstance(block, dict) and block.get('type') == 'text'))

def _clean_context_text(text: str) -> str:
    bodies = [m.group('body').strip() for m in RE_CHANNEL.finditer(text)]
    if bodies:
        text = bodies[-1]
    text = re.sub('<[^>]+>', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def recent_dialogue_context(data: dict, current: str) -> list[str]:
    path = data.get('transcript_path')
    if not path or not os.path.isfile(path):
        return []
    rows = []
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get('isSidechain') or entry.get('type') not in ('user', 'assistant'):
                    continue
                body = _clean_context_text(_entry_text(entry))
                if not body or SYNTHETIC.search(body):
                    continue
                rows.append(body[:260])
    except Exception:
        return []
    if rows and rows[-1].strip() == current.strip():
        rows.pop()
    picked, used = ([], 0)
    for body in reversed(rows):
        if used + len(body) > CONTEXT_CHARS:
            body = body[:max(0, CONTEXT_CHARS - used)]
        if body:
            picked.append(body)
            used += len(body)
        if len(picked) >= CONTEXT_TURNS or used >= CONTEXT_CHARS:
            break
    return list(reversed(picked))

def contextual_query(data: dict, current: str) -> tuple[str, bool]:
    raw = data.get('prompt') or ''
    if not (LOW_INFO_REPLY.search(current) or IMAGE_MARK.search(raw)):
        return (current, False)
    context = recent_dialogue_context(data, current)
    if not context:
        return (current, False)
    return ('\n'.join(context + [current]), True)

def entity_compatible(query: str, candidate: str) -> bool:
    q = query.lower()
    c = candidate.lower()
    for guard in ENTITY_GUARDS:
        forbidden = guard['candidate_forbidden'].lower()
        if forbidden in c and forbidden not in q and any((cue.lower() in q for cue in guard['query_any'])):
            return False
    return True
CHOICE_ACTION = re.compile('(点个|点一|吃|喝|买|选|穿|用|带|要不要|可以.+吗)')
PREFERENCE_CUE = re.compile('(喜欢|不喜欢|爱吃|要配|不要|偏好|习惯|只要|最爱|配(?:米饭|饭)|要(?:椒盐|辣|甜|咸|冰|热))')
CONTEXT_ENTITY_MARKS = {'工作台', '信箱'}

def _cjk_bigrams(text: str) -> set[str]:
    out = set()
    for chunk in re.findall('[一-鿿]{2,}', text):
        out.update((chunk[i:i + 2] for i in range(len(chunk) - 1)))
    return out

def deterministic_relevance(query: str, candidate: str) -> bool:
    if not entity_compatible(query, candidate):
        return False
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    current = lines[-1] if lines else query
    if LOW_INFO_REPLY.search(current) and len(lines) > 1:
        context = '\n'.join(lines[:-1])
        for mark in distinctive_terms(context):
            if STRONG.fullmatch(mark) and mark.lower() in candidate.lower():
                return True
    if CHOICE_ACTION.search(current) and PREFERENCE_CUE.search(candidate) and (not EXPLICIT_RECALL.search(current)):
        overlap = _cjk_bigrams(current) & _cjk_bigrams(candidate)
        if overlap - {'可以', '要不', '不要'}:
            return True
    return False

def contextual_candidates(con, query: str, k: int=3) -> list[dict]:
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) < 2 or not LOW_INFO_REPLY.search(lines[-1]):
        return []
    marks = [m for m in distinctive_terms('\n'.join(lines[:-1])) if m in CONTEXT_ENTITY_MARKS]
    found = []
    seen = set()
    for mark in marks:
        try:
            rows = con.execute("SELECT id, ext_id, subject, content, status, user_pinned FROM facts WHERE status IN ('candidate','confirmed') AND valid_to IS NULL AND content LIKE ? ORDER BY user_pinned DESC, id DESC LIMIT ?", (f'%{mark}%', k)).fetchall()
        except Exception:
            continue
        for row in rows:
            item = dict(row)
            if item['ext_id'] not in seen:
                found.append(item | {'fact_id': item['id']})
                seen.add(item['ext_id'])
    return found[:k]

def main() -> int:
    try:
        data = json.loads(sys.stdin.read() or '{}')
    except Exception:
        return 0
    text = incoming(data)
    if not text or len(text) > 2000:
        return 0
    skipped = skip_input(text)
    if skipped:
        log(f'不查 | {text[:40]} | {skipped}')
        return 0
    marks = distinctive_terms(text)
    if not marks and (not EXPLICIT_RECALL.search(text)) and (not QUESTION.search(text)):
        log(f'不查 | {text[:40]} | 无指向性词')
        return 0
    sys.path.insert(0, ENGRAM)
    try:
        from retrieve import log_recall, retrieve
        from store.store import connect
    except Exception as e:
        log(f'import 失败：{e}')
        return 0

    def tel(r, decision, reason, selected=None):
        try:
            log_recall(con, text, r, decision=decision, reason=reason, selected=selected)
        except Exception as e:
            log(f'遥测失败：{e}')
    try:
        con = connect()
        r = retrieve(con, text, who=None)
    except Exception as e:
        log(f'召回失败：{e}')
        return 0
    pinned = pinned_candidates(con, text)
    if not r.should_recall and (not pinned or '软话' in (r.reason or '')):
        tel(r, 'skip', f'router:{r.reason}')
        log(f'不查 | {text[:40]} | {r.reason}')
        return 0
    seen_ids = {p['ext_id'] for p in pinned}
    r.hits = pinned + [h for h in (r.hits if r.should_recall else []) if h.get('ext_id') not in seen_ids]
    if not r.hits:
        if EXPLICIT_RECALL.search(text):
            print('【记忆库】已查，没有可靠记录；别猜，直说想不起来或问用户。')
        tel(r, 'skip', 'no_hits')
        log(f'查了没有 | {text[:40]} | 词{r.terms}')
        return 0
    strong = [m for m in marks if STRONG.fullmatch(m)]

    def relevant(h):
        body = (h.get('content') or '').lower()
        hit = [m for m in marks if m.lower() in body]
        return any((m in strong for m in hit)) or len(hit) >= 2
    how = '关键词闸'
    try:
        from rerank import judge
        j = judge(text, [h.get('content') or '' for h in r.hits])
    except Exception:
        j = None
    if j is not None:
        how = '模型精排'
        if not j['need']:
            tel(r, 'skip', f'rerank_no_need:{len(r.hits)}')
            log(f'模型判不用翻旧账 | {text[:40]} | {len(r.hits)}条')
            return 0
        scored = sorted(zip(j['scores'], range(len(r.hits))), reverse=True)
        on_topic = [r.hits[i] for sc, i in scored if sc >= 7]
    else:
        on_topic = [h for h in r.hits if relevant(h)] if marks else []
    if not on_topic:
        if EXPLICIT_RECALL.search(text):
            print('【记忆库】已查，没有跟这句相关的记录；别猜，直说想不起来或问用户。')
        tel(r, 'skip', f'off_topic:{len(r.hits)}')
        log(f'查到但不相关，不注（{how}） | {text[:40]} | 指向词{marks} | {len(r.hits)}条')
        return 0
    fresh = recently_injected(con)
    hits = [h for h in on_topic if h.get('ext_id') not in fresh]
    if not hits:
        tel(r, 'skip', f'recently_injected:{len(r.hits)}')
        log(f'刚注过，不重复 | {text[:40]} | 词{r.terms} | {len(r.hits)}条')
        return 0
    lines = [f'【记忆库·{r.task_type}】线索，非结论；当规则执行前先核对来源与适用范围。']
    if r.conflicts:
        lines.append('⚠️ 有打架的记录：' + '；'.join(r.conflicts) + '。按时间新的那条为准，并且要说清有两条。')
    used = len(lines[0])
    injected = set()
    for h in hits[:MAX_HITS]:
        mark = STATUS_MARK.get(h.get('status') or '', h.get('status') or '?')
        if h.get('user_pinned'):
            mark += '·钉'
        if h.get('my_stance') == 'disown':
            mark += '·我已不认'
        sid = (h.get('ext_id') or '')[5:13]
        piece = f'· [{h['subject']}·{mark} {sid}] {(h['content'] or '')[:PER_HIT]}'
        if used + len(piece) > MAX_CHARS:
            break
        lines.append(piece)
        used += len(piece)
        injected.add(h['ext_id'])
    lines.append('（库内记录；冲突时核对原文。）')
    print('\n'.join(lines)[:MAX_CHARS])
    tel(r, 'recall', f'injected:{len(injected)}/{len(r.hits)}', selected=injected)
    log(f'召回 {len(r.hits)} 条、注入 {len(injected)} 条（{how}）（跳过刚注过的 {len(r.hits) - len(hits)} 条） | {text[:40]} | 词{r.terms} | 层{r.layers_read}')
    return 0
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
