from __future__ import annotations
import datetime as dt
import sqlite3
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import bigrams, sha256
from salience import register as register_salience
from versions import CLASSIFIER_VERSION, REGISTRY_VERSION
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / 'engram.db'

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def connect(db: Path | str=DEFAULT_DB) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.executescript((ROOT / 'schema.sql').read_text(encoding='utf-8'))
    con.executescript((ROOT / 'derived.sql').read_text(encoding='utf-8'))
    con.executescript((ROOT / 'telemetry.sql').read_text(encoding='utf-8'))
    con.executescript((ROOT / 'working.sql').read_text(encoding='utf-8'))
    register_salience(con)
    return con

def add_evidence(con, *, source_kind, source_actor, source_locator, content, source_path=None, occurred_at=None, verification_status='unverified') -> int:
    digest = sha256(content)
    row = con.execute('SELECT id FROM evidence WHERE source_locator=? AND content_sha256=?', (source_locator, digest)).fetchone()
    if row:
        return row['id']
    cur = con.execute('INSERT INTO evidence (ext_id, captured_at, occurred_at, source_kind, source_actor,\n             source_path, source_locator, content, content_sha256, verification_status)\n           VALUES (?,?,?,?,?,?,?,?,?,?)', (f'ev_{uuid.uuid4().hex[:12]}', now(), occurred_at, source_kind, source_actor, source_path, source_locator, content, digest, verification_status))
    return cur.lastrowid

def promote(con, *, subject, fact_type, content, evidence_ids=(), status='candidate', valid_from=None, importance=None, confidence=None, emotion=None, user_pinned=False, supersedes_id=None) -> int | None:
    digest = sha256(content)
    dup = con.execute("SELECT id FROM facts WHERE subject=? AND content_sha256=? AND status IN ('candidate','confirmed') AND valid_to IS NULL", (subject, digest)).fetchone()
    if dup:
        return None
    ts = now()
    cur = con.execute('INSERT INTO facts (ext_id, subject, fact_type, content, grams, content_sha256,\n             status, valid_from, supersedes_id, user_pinned, importance, emotion, confidence,\n             created_at, updated_at)\n           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (f'fact_{uuid.uuid4().hex[:12]}', subject, fact_type, content, bigrams(content), digest, status, valid_from, supersedes_id, int(bool(user_pinned)), importance, emotion, confidence, ts, ts))
    fact_id = cur.lastrowid
    for ev in evidence_ids:
        con.execute('INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id) VALUES (?,?)', (fact_id, ev))
    return fact_id

def record_class(con, fact_id: int, fact_type: str, confidence: float, reason: str) -> None:
    ts = now()
    con.execute('INSERT OR REPLACE INTO fact_class (fact_id, fact_type, confidence, reason, classifier_version, classified_at, built_at) VALUES (?,?,?,?,?,?,?)', (fact_id, fact_type, confidence, reason, CLASSIFIER_VERSION, ts, ts))

def record_tier(con, evidence_id: int, tier: str, origin_root: str) -> None:
    con.execute('INSERT OR REPLACE INTO evidence_tier (evidence_id, tier, origin_root, built_at) VALUES (?,?,?,?)', (evidence_id, tier, origin_root, now()))

def record_key(con, fact_id: int, fact_key: str, value: str | None, polarity: str='pos', eligible: bool=True) -> None:
    con.execute('INSERT OR REPLACE INTO fact_key_map (fact_id, fact_key, value, polarity, eligible, registry_version, built_at) VALUES (?,?,?,?,?,?,?)', (fact_id, fact_key, value, polarity, int(bool(eligible)), REGISTRY_VERSION, now()))

def relate(con, parent_id: int, child_id: int, relation: str='derived_from') -> None:
    con.execute('INSERT OR IGNORE INTO fact_relations (parent_fact_id, child_fact_id, relation, created_at) VALUES (?,?,?,?)', (parent_id, child_id, relation, now()))

def _bump(con, fact_id: int, expect_version: int | None, **cols) -> None:
    cols['updated_at'] = now()
    sets = ', '.join((f'{k}=?' for k in cols)) + ', version=version+1'
    args = [*cols.values(), fact_id]
    sql = f'UPDATE facts SET {sets} WHERE id=?'
    if expect_version is not None:
        sql += ' AND version=?'
        args.append(expect_version)
    if con.execute(sql, args).rowcount == 0:
        raise RuntimeError(f'乐观锁没对上：fact {fact_id} 已被改过，重读再来')

def supersede(con, old_ext_id: str, *, content, expect_version=None, valid_from=None, **kw) -> int:
    old = con.execute('SELECT * FROM facts WHERE ext_id=?', (old_ext_id,)).fetchone()
    if not old:
        raise KeyError(old_ext_id)
    ts = valid_from or now()
    _bump(con, old['id'], expect_version, status='superseded', valid_to=ts)
    return promote(con, subject=kw.pop('subject', old['subject']), fact_type=kw.pop('fact_type', old['fact_type']), content=content, valid_from=ts, supersedes_id=old['id'], **kw)

def invalidate(con, ext_id: str, *, reason: str, actor: str='assistant', expect_version=None) -> None:
    row = con.execute('SELECT id FROM facts WHERE ext_id=?', (ext_id,)).fetchone()
    if not row:
        raise KeyError(ext_id)
    _bump(con, row['id'], expect_version, status='invalid', valid_to=now())
    set_stance(con, ext_id, actor=actor, stance='disown', reason=reason)

def confirm(con, ext_id: str, *, expect_version=None) -> None:
    row = con.execute('SELECT id FROM facts WHERE ext_id=?', (ext_id,)).fetchone()
    _bump(con, row['id'], expect_version, status='confirmed')

def pin(con, ext_id: str, on: bool=True, *, expect_version=None) -> None:
    row = con.execute('SELECT id FROM facts WHERE ext_id=?', (ext_id,)).fetchone()
    _bump(con, row['id'], expect_version, user_pinned=int(bool(on)))

def touch(con, ext_id: str) -> None:
    con.execute('UPDATE facts SET access_count=access_count+1, last_accessed=?, updated_at=? WHERE ext_id=?', (now(), now(), ext_id))

def set_stance(con, ext_id: str, *, actor: str, stance: str, reason: str | None=None) -> None:
    row = con.execute('SELECT id FROM facts WHERE ext_id=?', (ext_id,)).fetchone()
    if not row:
        raise KeyError(ext_id)
    con.execute('INSERT INTO stances (fact_id, actor, stance, reason, created_at) VALUES (?,?,?,?,?)', (row['id'], actor, stance, reason, now()))

def search(con, query: str, *, subject=None, limit=10, include_disowned=False, fact_types=None, extracted_only=False):
    q = query.strip()
    if len(q) < 2:
        return []
    if len(q) == 2:
        table, match = ('facts_gram_fts', f'"{q}"')
    else:
        table, match = ('facts_fts', q)
    sql = f"\n      SELECT f.ext_id, f.subject, f.content, f.status, f.importance, f.confidence,\n             f.user_pinned, f.access_count, f.last_accessed, f.valid_from,\n             COALESCE(cs.stance, 'none') AS my_stance,\n             engram_activation(f.importance, f.confidence, f.emotion, f.access_count,\n                               julianday('now') - julianday(f.valid_from),\n                               f.user_pinned, f.fact_type) AS score,\n             f.layer\n      FROM {table} t\n      JOIN facts f ON f.id = t.rowid\n      LEFT JOIN current_stances cs ON cs.fact_id = f.id AND cs.actor = 'assistant'\n      WHERE {table} MATCH ?\n        AND f.status IN ('candidate','confirmed')\n        AND f.valid_to IS NULL\n    "
    args: list = [match]
    if subject:
        sql += ' AND f.subject = ?'
        args.append(subject)
    if fact_types:
        sql += ' AND f.fact_type IN (%s)' % ','.join('?' * len(fact_types))
        args.extend(fact_types)
    if extracted_only:
        sql += " AND f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')"
    if not include_disowned:
        sql += " AND COALESCE(cs.stance,'none') <> 'disown'"
    sql += ' ORDER BY score DESC, f.valid_from DESC LIMIT ?'
    args.append(limit)
    return [dict(r) for r in con.execute(sql, args)]

def stats(con) -> dict:
    return {'evidence': con.execute('SELECT COUNT(*) FROM evidence').fetchone()[0], 'evidence_verified': con.execute("SELECT COUNT(*) FROM evidence WHERE verification_status='verified'").fetchone()[0], 'facts_by_status': dict(con.execute('SELECT status, COUNT(*) FROM facts GROUP BY status').fetchall()), 'live': con.execute('SELECT COUNT(*) FROM live_facts').fetchone()[0], 'pinned': con.execute('SELECT COUNT(*) FROM facts WHERE user_pinned=1').fetchone()[0]}
