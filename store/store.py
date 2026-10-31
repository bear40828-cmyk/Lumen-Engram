#!/usr/bin/env python3
"""Lumen Engram · store —— 增删改查 / 作废 / 乐观锁 / 检索

设计出自 app 里的 GPT（它额度见底了，代码由助手落盘）。
落盘时守住三条铁律：
  1. 分数只影响排序和冷热，不影响存在
  2. 来源字段本身不是证据
  3. 遗忘是不再容易被叫醒，不是烧历史

「删」这个词在这里一律指改状态。物理删除在 schema 层就被触发器挡住了。
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import bigrams, sha256  # noqa: E402
import redact  # noqa: E402
from salience import COLD_WEIGHT, register as register_salience  # noqa: E402
from versions import CLASSIFIER_VERSION, REGISTRY_VERSION  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "engram.db"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def connect(db: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
    con.executescript((ROOT / "derived.sql").read_text(encoding="utf-8"))
    con.executescript((ROOT / "telemetry.sql").read_text(encoding="utf-8"))
    con.executescript((ROOT / "working.sql").read_text(encoding="utf-8"))
    con.executescript((ROOT / "actor_fix.sql").read_text(encoding="utf-8"))     # 说话人标注修正层（只加表/视图）
    register_salience(con)          # 让排序能在库里算衰减和激活度
    return con


# ── evidence ────────────────────────────────────────────────────────────────
def add_evidence(con, *, source_kind, source_actor, source_locator, content,
                 source_path=None, occurred_at=None,
                 verification_status="unverified") -> int:
    """幂等：同一 locator + 同一内容只存一次，重复调用返回已有的 id。

    **密钥打码（没有开关）**：存之前先过 redact；content_sha256 按打码后的文本算。
    老证据是原样存的（不可改），所以去重时也认「打码前的哈希」，免得重扫同一行又存一份打码版。"""
    raw_digest = sha256(content)
    content = redact.scrub(content, "evidence")
    digest = sha256(content)
    row = con.execute(
        "SELECT id FROM evidence WHERE source_locator=? AND content_sha256=?",
        (source_locator, digest)).fetchone()
    if row:
        return row["id"]
    if raw_digest != digest:
        row = con.execute(
            "SELECT id FROM evidence WHERE source_locator=? AND content_sha256=?",
            (source_locator, raw_digest)).fetchone()
        if row:
            return row["id"]
    cur = con.execute(
        """INSERT INTO evidence (ext_id, captured_at, occurred_at, source_kind, source_actor,
             source_path, source_locator, content, content_sha256, verification_status)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (f"ev_{uuid.uuid4().hex[:12]}", now(), occurred_at, source_kind, source_actor,
         source_path, source_locator, content, digest, verification_status))
    return cur.lastrowid


# ── facts ───────────────────────────────────────────────────────────────────
def promote(con, *, subject, fact_type, content, evidence_ids=(), status="candidate",
            valid_from=None, importance=None, confidence=None, emotion=None,
            user_pinned=False, supersedes_id=None) -> int | None:
    """把候选记忆写进 facts。内容重复（同 subject 同 sha）就不再新增，返回 None。

    防回声兜底（task2-08，**没有开关**）：证据说话人是 X，subject 只能是 X 或 us（task2-09：us 要双方原话）；
    不符就拒绝并返回 None（stderr 记一行聚合日志）。任何入库路径都绕不过这一道。
    例外：status='confirmed' 或 user_pinned（她自己的纠正 / 她自己钉的）不拦。"""
    evidence_ids = list(evidence_ids or ())
    if status != "confirmed" and not user_pinned and evidence_ids:
        import echo
        ok, why = echo.evidence_allows(con, subject, evidence_ids)
        if not ok:
            echo.note_block(subject, why)
            return None
    content = redact.scrub(content, "fact")          # 事实正文也不留密钥（没有开关）
    digest = sha256(content)
    # 只跟还活着的比。作废/被顶掉的那条不算重复——旧值可以重新成立，是新的一行、新的有效期。
    dup = con.execute(
        "SELECT id FROM facts WHERE subject=? AND content_sha256=?"
        " AND status IN ('candidate','confirmed') AND valid_to IS NULL",
        (subject, digest)).fetchone()
    if dup:
        return None
    ts = now()
    cur = con.execute(
        """INSERT INTO facts (ext_id, subject, fact_type, content, grams, content_sha256,
             status, valid_from, supersedes_id, user_pinned, importance, emotion, confidence,
             created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"fact_{uuid.uuid4().hex[:12]}", subject, fact_type, content, bigrams(content), digest,
         status, valid_from, supersedes_id, int(bool(user_pinned)),
         importance, emotion, confidence, ts, ts))
    fact_id = cur.lastrowid
    for ev in evidence_ids:
        con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id) VALUES (?,?)",
                    (fact_id, ev))
    return fact_id


def record_class(con, fact_id: int, fact_type: str, confidence: float, reason: str) -> None:
    """把分类的置信度和理由记进派生层，带上分类器版本，方便以后整库重跑。"""
    ts = now()
    con.execute(
        "INSERT OR REPLACE INTO fact_class"
        " (fact_id, fact_type, confidence, reason, classifier_version, classified_at, built_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (fact_id, fact_type, confidence, reason, CLASSIFIER_VERSION, ts, ts))


def record_tier(con, evidence_id: int, tier: str, origin_root: str) -> None:
    """证据分级 + 同源聚簇。只有 primary 才在巩固时算独立一票。"""
    con.execute(
        "INSERT OR REPLACE INTO evidence_tier (evidence_id, tier, origin_root, built_at)"
        " VALUES (?,?,?,?)", (evidence_id, tier, origin_root, now()))


def record_key(con, fact_id: int, fact_key: str, value: str | None,
               polarity: str = "pos", eligible: bool = True) -> None:
    """记这条事实占了哪个槽位。key 必须出自 keys.yaml。

    polarity 区分正反（「不喜欢紫色」不能跟「喜欢紫色」并一组）；
    eligible=False 表示命中了槽位但不该参与自动巩固（说的是别人、值是从数字串里抠的）。
    """
    con.execute(
        "INSERT OR REPLACE INTO fact_key_map (fact_id, fact_key, value, polarity, eligible,"
        " registry_version, built_at) VALUES (?,?,?,?,?,?,?)",
        (fact_id, fact_key, value, polarity, int(bool(eligible)), REGISTRY_VERSION, now()))


def relate(con, parent_id: int, child_id: int, relation: str = "derived_from") -> None:
    """巩固是抽象不是覆盖：把支撑的 episode 连到抽象出来的那条上，旧的原样留着。"""
    con.execute(
        "INSERT OR IGNORE INTO fact_relations (parent_fact_id, child_fact_id, relation, created_at)"
        " VALUES (?,?,?,?)", (parent_id, child_id, relation, now()))


def _bump(con, fact_id: int, expect_version: int | None, **cols) -> None:
    """带乐观锁的状态迁移。只动 schema 允许改的列。"""
    cols["updated_at"] = now()
    sets = ", ".join(f"{k}=?" for k in cols) + ", version=version+1"
    args = [*cols.values(), fact_id]
    sql = f"UPDATE facts SET {sets} WHERE id=?"
    if expect_version is not None:
        sql += " AND version=?"
        args.append(expect_version)
    if con.execute(sql, args).rowcount == 0:
        raise RuntimeError(f"乐观锁没对上：fact {fact_id} 已被改过，重读再来")


def supersede(con, old_ext_id: str, *, content, expect_version=None, valid_from=None,
              hard: bool = False, **kw) -> int:
    """事实变了：旧的封上有效区间标 superseded，新的插一条指回去。旧行内容一个字不改。

    valid_from 给的是新事实**实际发生**的时刻（抽取时就是原话的时间），不给就用现在。
    """
    # 2026-10-28（任务二·步骤4）冻结：取代/作废这条写入链全仓没有产出过结果（见
    # tests 里「supersede/invalidate 无调用方」那条已知缺陷）。新旧冲突改由召回时的 newer.py 处理，不改库。
    # 默认抛 FrozenError、不动库；ENGRAM_INVALIDATE_CHAIN=1 恢复。
    # 2026-10-29：自动判新旧取代的准确率只有 29–59%，管线里不再有任何调用方会封旧事（改成 stale.py 的
    # 「可能过时」软标记）。这条硬取代只留给手工、明说 hard=True 的场合；开关和 hard 缺一不可。
    from flags import FrozenError, require
    require("INVALIDATE_CHAIN", "store.supersede")
    if not hard:
        raise FrozenError("store.supersede 已降级：自动管线只做「可能过时」软标记（stale.py）；手工硬取代要显式 hard=True")
    old = con.execute("SELECT * FROM facts WHERE ext_id=?", (old_ext_id,)).fetchone()
    if not old:
        raise KeyError(old_ext_id)
    ts = valid_from or now()
    _bump(con, old["id"], expect_version, status="superseded", valid_to=ts)
    return promote(con, subject=kw.pop("subject", old["subject"]),
                   fact_type=kw.pop("fact_type", old["fact_type"]),
                   content=content, valid_from=ts, supersedes_id=old["id"], **kw)


def invalidate(con, ext_id: str, *, reason: str, actor: str = "claude",
               expect_version=None, hard: bool = False) -> None:
    """抽错了。不删，标 invalid，并在表态层留一条不认。"""
    # 冻结，同 supersede（任务二·步骤4）。
    from flags import FrozenError, require
    require("INVALIDATE_CHAIN", "store.invalidate")
    if not hard:
        raise FrozenError("store.invalidate 已降级：只留给手工、显式 hard=True 的场合")
    row = con.execute("SELECT id FROM facts WHERE ext_id=?", (ext_id,)).fetchone()
    if not row:
        raise KeyError(ext_id)
    _bump(con, row["id"], expect_version, status="invalid", valid_to=now())
    set_stance(con, ext_id, actor=actor, stance="disown", reason=reason)


def confirm(con, ext_id: str, *, expect_version=None) -> None:
    row = con.execute("SELECT id FROM facts WHERE ext_id=?", (ext_id,)).fetchone()
    _bump(con, row["id"], expect_version, status="confirmed")


def pin(con, ext_id: str, on: bool = True, *, expect_version=None) -> None:
    """她钉住的东西不参与衰减和冷层迁移。"""
    row = con.execute("SELECT id FROM facts WHERE ext_id=?", (ext_id,)).fetchone()
    _bump(con, row["id"], expect_version, user_pinned=int(bool(on)))


def touch(con, ext_id: str) -> None:
    """只记「被看见过」，不加权。

    以前这里给 access_count +1，而 access_count 进激活度公式——engram_think 对全部命中都调它，
    等于误召回一次就排得更前、更容易再被误召回，自己把错误喂胖（tests 七、不许无条件强化）。
    现在被召回只更新 last_accessed；要加权走 reinforce()，只在确实选用时调。"""
    con.execute("UPDATE facts SET last_accessed=?, updated_at=? WHERE ext_id=?",
                (now(), now(), ext_id))


def reinforce(con, ext_id: str) -> None:
    """确实用上了才加权：展开读全文、她确认、被钉住这类有选择的动作。

    防回声（task2-08）：「只有转述」的事实（主体是人、证据全来自别人、未 confirmed/钉住）不加权，
    只记看见——否则助手的话被召回、展开几次就越排越前，长成长期事实。"""
    import echo
    if echo.is_echo_only(echo.provenance(con, [ext_id]).get(ext_id)):
        con.execute("UPDATE facts SET last_accessed=?, updated_at=? WHERE ext_id=?",
                    (now(), now(), ext_id))
        return
    con.execute(
        "UPDATE facts SET access_count=access_count+1, last_accessed=?, updated_at=? WHERE ext_id=?",
        (now(), now(), ext_id))


def set_stance(con, ext_id: str, *, actor: str, stance: str, reason: str | None = None) -> None:
    """认 / 不认 / 悬置。只插不改，改主意就再插一条。"""
    row = con.execute("SELECT id FROM facts WHERE ext_id=?", (ext_id,)).fetchone()
    if not row:
        raise KeyError(ext_id)
    con.execute(
        "INSERT INTO stances (fact_id, actor, stance, reason, created_at) VALUES (?,?,?,?,?)",
        (row["id"], actor, stance, reason, now()))


# ── 检索 ────────────────────────────────────────────────────────────────────
def search(con, query: str, *, subject=None, limit=10, include_disowned=False,
           fact_types=None, extracted_only=False):
    """按查询字数选索引：>=3 字 trigram，==2 字 bigram，1 字不做正常召回。

    排序 = 重要度 + 置信度 + 新近 + 用得多 + 钉住加权。情绪不参与召回排序的硬门槛，
    只作为一个小权重——它永远不决定一条记录是否存在。
    """
    # 按空白拆词，每个词各自加引号当短语（防 FTS5 语法报错），词之间取交集。
    # 不能把整句包成一个短语：「阿禾 穗穗」会要求原文里一字不差出现这串带空格的字。
    terms = [t for t in query.split() if len(t) >= 2]
    if not terms:
        return []
    conds, margs = [], []
    for t in terms:
        if len(t) == 2:
            table = "facts_gram_fts"
        else:
            table = "facts_fts"
        conds.append(f"f.id IN (SELECT rowid FROM {table} WHERE {table} MATCH ?)")
        margs.append('"%s"' % t.replace('"', '""'))

    sql = f"""
      SELECT f.ext_id, f.subject, f.content, f.status, f.importance, f.confidence,
             f.user_pinned, f.access_count, f.last_accessed, f.valid_from, f.fact_type,
             COALESCE(cs.stance, 'none') AS my_stance,
             engram_activation(f.importance, f.confidence, f.emotion, f.access_count,
                               julianday('now') - julianday(COALESCE(f.valid_from, f.created_at)),
                               f.user_pinned, f.fact_type)
               * CASE WHEN f.layer = 'cold' THEN {COLD_WEIGHT} ELSE 1.0 END AS score,
             f.layer
      FROM facts f
      LEFT JOIN current_stances cs ON cs.fact_id = f.id AND cs.actor = 'claude'
      WHERE {" AND ".join(conds)}
        AND f.status IN ('candidate','confirmed')
        AND f.valid_to IS NULL
    """
    args: list = list(margs)
    if subject:
        sql += " AND f.subject = ?"
        args.append(subject)
    if fact_types:
        # 路由分层召回用：只在指定的记忆类型里找（快路径查情景，慢路径查语义等）
        sql += " AND f.fact_type IN (%s)" % ",".join("?" * len(fact_types))
        args.extend(fact_types)
    if extracted_only:
        # 抽取器整理过的「事」（原话→事），比原话片段干净，召回时先看这一层
        sql += " AND f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')"
    if not include_disowned:
        sql += " AND COALESCE(cs.stance,'none') <> 'disown'"
    sql += " ORDER BY score DESC, f.valid_from DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in con.execute(sql, args)]


def stats(con) -> dict:
    return {
        "evidence": con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0],
        "evidence_verified": con.execute(
            "SELECT COUNT(*) FROM evidence WHERE verification_status='verified'").fetchone()[0],
        "facts_by_status": dict(con.execute(
            "SELECT status, COUNT(*) FROM facts GROUP BY status").fetchall()),
        "live": con.execute("SELECT COUNT(*) FROM live_facts").fetchone()[0],
        "pinned": con.execute("SELECT COUNT(*) FROM facts WHERE user_pinned=1").fetchone()[0],
    }
