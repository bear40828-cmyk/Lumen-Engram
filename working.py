from __future__ import annotations
import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import gramset
BOUND = {'goal': 3, 'task': 7, 'topic': 5, 'state': 5, 'emotion': 1}
KIND_CN = {'goal': '目标', 'task': '在办', 'topic': '话题', 'state': '状态', 'emotion': '情绪'}
EMOTION_CUES = [(re.compile('(生气|气死|不高兴|烦|操你|)'), '生气'), (re.compile('(难过|想哭|委屈|不开心)'), '难过'), (re.compile('(开心|高兴|笑死|好耶|太好了)'), '开心'), (re.compile('(累|困|歇会|不想动|好累)'), '累'), (re.compile('(想你|抱抱|亲亲|好想)'), '想念')]

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def session_id() -> str:
    return os.environ.get('CLAUDE_SESSION_ID') or f'pid-{os.getpid()}'

@dataclass(frozen=True)
class Item:
    id: int
    kind: str
    content: str
    status: str
    confidence: float
    updated_at: str

def _rows(con, sql: str, args=()) -> list[Item]:
    return [Item(r['id'], r['kind'], r['content'], r['status'], r['confidence'], r['updated_at']) for r in con.execute(sql, args)]

def current(con, kind: str | None=None) -> list[Item]:
    sql = "SELECT * FROM working_memory WHERE status='open'" + (' AND kind=?' if kind else '') + ' ORDER BY updated_at DESC, id DESC'
    return _rows(con, sql, (kind,) if kind else ())

def put(con, kind: str, content: str, *, confidence: float=0.6, turn_no: int | None=None, evidence_id: int | None=None, note: str | None=None) -> int:
    if kind not in BOUND:
        raise ValueError(f'未知 kind：{kind}')
    content = (content or '').strip()
    if not content:
        raise ValueError('内容为空')
    row = con.execute("SELECT id FROM working_memory WHERE kind=? AND content=? AND status='open'", (kind, content)).fetchone()
    if row:
        con.execute('UPDATE working_memory SET updated_at=?, confidence=MAX(confidence,?) WHERE id=?', (now(), confidence, row['id']))
        con.commit()
        _bound(con, kind)
        return row['id']
    cur = con.execute("INSERT INTO working_memory\n             (session_id, kind, content, status, turn_no, confidence,\n              created_at, updated_at, evidence_id, note)\n           VALUES (?,?,?,'open',?,?,?,?,?,?)", (session_id(), kind, content, turn_no, confidence, now(), now(), evidence_id, note))
    con.commit()
    _bound(con, kind)
    return cur.lastrowid

def _bound(con, kind: str) -> int:
    keep = BOUND[kind]
    olds = con.execute("SELECT id FROM working_memory WHERE kind=? AND status='open'\n           ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?", (kind, keep)).fetchall()
    for r in olds:
        con.execute("UPDATE working_memory SET status='evicted', closed_at=?, updated_at=? WHERE id=?", (now(), now(), r['id']))
    con.commit()
    return len(olds)

def close(con, item_id: int, status: str='done') -> bool:
    if status not in ('done', 'dropped', 'evicted'):
        raise ValueError('status 只能是 done / dropped / evicted')
    cur = con.execute("UPDATE working_memory SET status=?, closed_at=?, updated_at=? WHERE id=? AND status='open'", (status, now(), now(), item_id))
    con.commit()
    return cur.rowcount > 0

def note_emotion(con, text: str) -> str | None:
    for pat, label in EMOTION_CUES:
        if pat.search(text or ''):
            put(con, 'emotion', label, confidence=0.7, note='从用户原话里读到的')
            return label
    return None
MIN_SHARED = 2

def _overlap(a: str, b: str) -> float:
    ga, gb = (gramset(a), gramset(b))
    if not ga or not gb:
        return 0.0
    shared = ga & gb
    if len(shared) < MIN_SHARED:
        return 0.0
    return len(shared) / len(ga)

def relevance(con, text: str) -> tuple[float, str]:
    best, why = (0.0, '')
    for it in current(con):
        if it.kind not in ('goal', 'task', 'topic'):
            continue
        v = _overlap(text, it.content)
        v *= {'goal': 1.0, 'task': 0.9, 'topic': 0.7}[it.kind]
        if v > best:
            best, why = (v, f'{KIND_CN[it.kind]}「{it.content}」')
    return (round(best, 3), why)

def snapshot(con) -> str:
    lines: list[str] = []
    for kind in ('goal', 'task', 'topic', 'state', 'emotion'):
        items = current(con, kind)
        if not items:
            continue
        lines.append(f'{KIND_CN[kind]}：')
        for it in items:
            lines.append(f'  - {it.content}')
    return '\n'.join(lines) or '（工作记忆是空的）'

def main() -> int:
    from store.store import connect
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', action='append')
    ap.add_argument('--task', action='append')
    ap.add_argument('--topic', action='append')
    ap.add_argument('--state', action='append')
    (ap.add_argument('--done', type=int), ap.add_argument('--drop', type=int))
    ap.add_argument('--relevance')
    ap.add_argument('--snapshot', action='store_true')
    ap.add_argument('--all', action='store_true', help='连关掉的一起看')
    args = ap.parse_args()
    con = connect()
    for kind in ('goal', 'task', 'topic', 'state'):
        for v in getattr(args, kind) or []:
            print(f'记下{KIND_CN[kind]} #{put(con, kind, v, confidence=0.9)}：{v}')
    if args.done:
        print('已办完' if close(con, args.done, 'done') else '没有这条开着的')
    if args.drop:
        print('已放弃' if close(con, args.drop, 'dropped') else '没有这条开着的')
    if args.relevance:
        v, why = relevance(con, args.relevance)
        print(f'相关度 {v}  ← {why or '跟手上的事都不搭'}')
    if args.snapshot:
        print(snapshot(con))
        return 0
    print('\n当前工作记忆：')
    for it in current(con):
        print(f'  #{it.id:<4}{KIND_CN[it.kind]:<4}{it.content}  （信心 {it.confidence:.1f}）')
    if args.all:
        print('\n已关掉的：')
        for it in _rows(con, "SELECT * FROM working_memory WHERE status<>'open' ORDER BY updated_at DESC LIMIT 20"):
            print(f'  #{it.id:<4}{KIND_CN[it.kind]:<4}[{it.status}] {it.content}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
