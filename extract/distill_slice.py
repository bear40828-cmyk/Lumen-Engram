#!/usr/bin/env python3
"""Lumen Engram · 切片蒸馏：会话安静 ≥3 小时才整理，带游标、退避重试、死信箱（任务三，2026-10-28）

原话已经由 extract/collect.py 原样存进 evidence。这里把一个会话里「游标之后」的那一段交给模型抽「事」。

什么时候整理一个会话（用户 2026-10-28 定的）：
  · 安静：这个会话最后一条话距现在 ≥ 3 小时（IDLE_MIN）——基本就是她睡觉的时候，便宜；
  · 保底：没安静，但游标之后最早那条已经 ≥ 6 小时（CAP_HOURS），或者游标之后攒了 ≥ 300 条（CAP_MSGS）
    ——强制先整理一次，最新 5 分钟（SETTLE_MIN）的话不碰，免得一轮话被切成两半；
  代价：当天聊的事要等到夜里（或保底触发）才变成「事」；原话 evidence 一直能搜到。

游标 distill_cursor 按 evidence.id（只增不减）。基线 engram_meta.distill_baseline_id：id ≤ 基线的从不整理
（不碰历史）；基线没设，--apply 直接拒绝（退出码 3）。

写入：一个块的事实、fact_origin、游标推进、distill_runs 在**同一个 SQLite 事务**里——任何一点出错或被杀，
要么都没发生要么都发生，不会重复、不会漏。

失败：游标不动，进待办（distill_cursor 的 fail_count / next_try_at）。等待翻倍：10、20、40、60、60 分钟，
最多试 5 次，从第一次失败起不超过 24 小时；到头进死信箱 distill_dead，**游标越过那一块**，不堵后面的消息。
夜里 --audit 把死信箱列出来，由人 --dead-retry / --dead-drop。

防回声 + 数字校验：每条事实过 from_evidence.check（说话人本人 / us 要双方原话 / 数字必须在原话里），
store.promote 再按库里证据兜底；本模块没有任何绕过它们的写路径。confirmed / 钉住的行从不被改，
近重复只给 candidate 的已有那条补证据链接。

用法：
    python3 -m extract.distill_slice --init-baseline        # 基线 = 当前最大 evidence.id（只设一次）
    python3 -m extract.distill_slice --plan                  # 默认：只看哪些会话到点、多少条，不连模型不写库
    python3 -m extract.distill_slice --apply                 # 真跑（flock 由调用方管）
    python3 -m extract.distill_slice --audit                 # 死信箱 / 排队 / 落后 / 派生待补
    python3 -m extract.distill_slice --dead-retry ID | --dead-drop ID
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

import echo                                                        # noqa: E402,F401
import llm                                                         # noqa: E402
from extract import from_evidence as FE                            # noqa: E402
from extract.from_channel import WHO, beijing                      # noqa: E402
from store.store import connect, now, promote            # noqa: E402

PROMPT_VERSION = "slice-v1"
IDLE_MIN = int(os.environ.get("ENGRAM_SLICE_IDLE_MIN", 180))
CAP_HOURS = float(os.environ.get("ENGRAM_SLICE_CAP_HOURS", 6))
CAP_MSGS = int(os.environ.get("ENGRAM_SLICE_CAP_MSGS", 300))
SETTLE_MIN = 5
CONTEXT_MSGS = 6
MAX_CHUNKS = 8
BACKOFF_MIN = (10, 20, 40, 60, 60)
MAX_ATTEMPTS = 5
DEAD_AFTER_H = 24
POISON_ATTEMPTS = 3            # 请求本身有问题（400/404/422）：试 3 次就进死信箱，不再等
NEAR_DUP = 0.65
NEAR_DUP_DAYS = 14
ACTORS = persona.ACTORS
PERMANENT_HTTP = {400, 404, 422}


def _t(s) -> dt.datetime | None:
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _iso(d: dt.datetime) -> str:
    return d.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


# ── 模型 ────────────────────────────────────────────────────────────────────
class PermanentError(RuntimeError):
    """请求本身有问题，重试没用。"""


def call_model_ex(chunk_text: str) -> tuple[list[dict], dict]:
    """主路由失败自动换备用（llm.chat_ex）。返回 (事实列表, {route, model, tokens_in, tokens_out})。"""
    try:
        r = llm.chat_ex([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": FE.build_prompt(chunk_text)},
        ], purpose="extract", max_tokens=2500, timeout=FE.TIMEOUT_S)
    except urllib.error.HTTPError as e:
        if e.code in PERMANENT_HTTP:
            raise PermanentError(f"HTTP {e.code}") from e
        raise
    m = re.search(r"\{.*\}", r.text, re.S)
    got = json.loads(m.group(0) if m else r.text)
    facts = (got.get("facts", []) or []) + FE.call_model_old(chunk_text)   # 召回补偿（默认关）：旧抽取器候选并进来
    u = r.usage or {}
    return facts, {"route": r.route, "model": r.model,
                   "tokens_in": u.get("prompt_tokens"), "tokens_out": u.get("completion_tokens")}


# ── 取数 / 渲染 ──────────────────────────────────────────────────────────────
def baseline(con) -> int | None:
    r = con.execute("SELECT value FROM engram_meta WHERE key='distill_baseline_id'").fetchone()
    return int(r[0]) if r else None


def init_baseline(con, force: bool = False) -> int:
    cur = baseline(con)
    if cur is not None and not force:
        return cur
    mx = con.execute("SELECT COALESCE(MAX(id),0) FROM evidence").fetchone()[0]
    con.execute("INSERT INTO engram_meta (key, value, updated_at) VALUES ('distill_baseline_id',?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (str(mx), now()))
    con.commit()
    return mx


def _render(row) -> dict:
    when = _t(row["occurred_at"]) or _t(row["captured_at"])
    cn = beijing(_iso(when))[1] if when else ""
    who = WHO.get(row["source_actor"], row["source_actor"])
    body = (row["content"] or "")
    if "：" in body[:30] and body.startswith(who + "（"):    # from_evidence 老存法自带前缀，不重复加
        text = body.replace("\n", " ")
    else:
        text = f"{who}（{cn}）：" + body.replace("\n", " ")
    return {"id": row["id"], "occurred_at": row["occurred_at"] or row["captured_at"],
            "source_actor": row["source_actor"], "content": text,
            "path": row["source_path"], "line": row["source_locator"]}


def _render_rows(con, rows) -> list[dict]:
    """渲染一批证据；说话人标注修正层（actor_fix overlay）改判过的按改判渲染：
    leading-block 改标成真正的发言人、embedded-block 拆成前半（原说话人）+ 佐证过的块（块里的人），没佐证的块不进片段。
    原 evidence 行一字不动；修正层出错＝按原标注。同一条证据的几段共用同一个 id。"""
    try:
        import actor_fix
        fixes = actor_fix.fixes_for_rows(con, rows)
    except Exception:                                                # noqa: BLE001
        fixes = {}
    out: list[dict] = []
    for r in rows:
        fx = fixes.get(r["id"])
        if not fx:
            out.append(_render(r))
            continue
        for actor, text in actor_fix.segments(fx, r["source_actor"], r["content"] or ""):
            if actor not in ACTORS or not text.strip():
                continue
            out.append(_render({"id": r["id"], "occurred_at": r["occurred_at"], "captured_at": r["captured_at"],
                                "source_actor": actor, "content": text, "source_path": r["source_path"],
                                "source_locator": r["source_locator"]}))
    return out


def _rows(con, key, lo, hi=None):
    q = ("SELECT id, occurred_at, captured_at, source_actor, content, source_path, source_locator FROM evidence"
         " WHERE source_path=? AND source_kind='chat' AND id>? AND source_actor IN (%s)" %
         ",".join("?" * len(ACTORS)))
    args = [key, lo, *ACTORS]
    if hi is not None:
        q += " AND id<=?"
        args.append(hi)
    return con.execute(q + " ORDER BY id", args).fetchall()


def _context(con, key, before_id):
    rs = con.execute(
        "SELECT id, occurred_at, captured_at, source_actor, content, source_path, source_locator FROM evidence"
        " WHERE source_path=? AND source_kind='chat' AND id<? AND source_actor IN (%s)"
        " ORDER BY id DESC LIMIT ?" % ",".join("?" * len(ACTORS)),
        (key, before_id, *ACTORS, CONTEXT_MSGS)).fetchall()
    return _render_rows(con, list(reversed(rs)))


def _prompt_text(ctx: list[dict], chunk: list[dict]) -> str:
    head = ""
    if ctx:
        head = "（以下是上文，只给你看背景，**不要从里面抽事，quote 也不能取自这里**）\n" + \
               "\n".join(c["content"][:FE.PER_MSG] for c in ctx) + "\n（上文到此为止，下面开始抽）\n"
    return head + FE.render(chunk)


# ── 游标 / 失败处理 ──────────────────────────────────────────────────────────
def _cursor(con, key):
    r = con.execute("SELECT * FROM distill_cursor WHERE session_key=?", (key,)).fetchone()
    return dict(r) if r else {"session_key": key, "last_evidence_id": 0, "fail_count": 0,
                              "first_fail_at": None, "next_try_at": None, "last_error": None}


def _save_cursor(con, c, **kw):
    c = {**c, **kw}
    con.execute(
        "INSERT INTO distill_cursor (session_key, last_evidence_id, fail_count, first_fail_at, next_try_at,"
        " last_error, updated_at) VALUES (?,?,?,?,?,?,?)"
        " ON CONFLICT(session_key) DO UPDATE SET last_evidence_id=excluded.last_evidence_id,"
        " fail_count=excluded.fail_count, first_fail_at=excluded.first_fail_at,"
        " next_try_at=excluded.next_try_at, last_error=excluded.last_error, updated_at=excluded.updated_at",
        (c["session_key"], c["last_evidence_id"], c["fail_count"], c["first_fail_at"], c["next_try_at"],
         c["last_error"], now()))
    return c


def _run_row(con, run_id, kind, key, chunk, n_chunks, meta, kept, dup, near, dropped, status, err, t0):
    con.execute(
        "INSERT INTO distill_runs (run_id, kind, session_key, ev_lo, ev_hi, n_msgs, n_chunks, route, model,"
        " tokens_in, tokens_out, kept, dup, near_dup, dropped, status, error, started_at, finished_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, kind, key, chunk[0]["id"], chunk[-1]["id"], len(chunk), n_chunks,
         (meta or {}).get("route"), (meta or {}).get("model"), (meta or {}).get("tokens_in"),
         (meta or {}).get("tokens_out"), kept, dup, near, dropped, status, (err or "")[:300], _iso(t0), now()))


def _bigr(s: str) -> set:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _near_dup(con, subject, content, when, ev_ids):
    """近重复：同 subject、近 14 天、活着的 candidate/confirmed 里 bigram Jaccard ≥ 0.65。
    返回已有那条 (id, status, user_pinned) 或 None。"""
    base = _t(when) or dt.datetime.now(dt.timezone.utc)
    since = _iso(base - dt.timedelta(days=NEAR_DUP_DAYS))
    mine = _bigr(content)
    if not mine:
        return None
    for r in con.execute(
            "SELECT id, status, user_pinned, content FROM facts WHERE subject=? AND valid_to IS NULL"
            " AND status IN ('candidate','confirmed') AND COALESCE(valid_from, created_at) >= ?",
            (subject, since)):
        theirs = _bigr(r["content"])
        if theirs and len(mine & theirs) / len(mine | theirs) >= NEAR_DUP:
            return r
    return None


def _timeline_target(con, ok):
    """新事是否让某条旧事「可能过时」：两个开关（TIMELINE、INVALIDATE_CHAIN）都开才判，否则 None。
    返回 (旧事, 槽描述) 或 None。旧事只在 candidate、没钉住的里面找（见 timeline.candidates）。
    2026-10-29 降级：判中也只在新事入库后写一条「可能过时」软标记（stale.mark），**不封旧事、不改状态**。"""
    import flags
    if not (flags.on("TIMELINE") and flags.on("INVALIDATE_CHAIN")):
        return None
    # 先把必然过不了 stale.check 的挡在模型调用之前（省钱）：新的必须是她的事、而且有她本人说话的证据
    import echo
    import persona
    if ok.get("subject") != persona.USER:
        return None
    ev = [e for e in (ok.get("evidence_id"), ok.get("evidence_id2")) if e]
    if persona.USER not in echo.evidence_actors(con, ev):
        return None
    from extract import timeline
    from store.store import sha256
    if con.execute("SELECT 1 FROM facts WHERE subject=? AND content_sha256=?"
                   " AND status IN ('candidate','confirmed') AND valid_to IS NULL",
                   (ok["subject"], sha256(ok["content"]))).fetchone():
        return None
    olds = timeline.candidates(con, ok["subject"], ok["content"], ok["occurred_at"])
    idx, why = timeline.judge(ok["content"], ok["occurred_at"], olds)
    return (olds[idx], why.split("｜", 1)[0]) if idx is not None else None


def _write_chunk(con, key, chunk, facts, run_id, kind, c, meta, t0, n_chunks, *, advance=True, inject=None):
    """一个事务：事实 + fact_origin + 游标 + distill_runs。出错整体回滚并重新抛出。"""
    kept = dup = near = dropped = 0
    reasons: dict[str, int] = {}
    origin = "distill-cap" if kind == "cap" else "distill"
    con.commit()
    # 判「新事让哪条旧事可能过时」要调模型（最长 30 秒/次），不能在写事务里做：先在事务外判完（只读），
    # 事务里 stale.mark 还会再验一遍（2026-10-29 修：以前在 BEGIN IMMEDIATE 里调模型，会长时间占着写锁）
    gated = FE.gate_pass(facts, chunk)
    targets = [(_timeline_target(con, ok) if ok else None) for ok, _why in gated]
    con.execute("BEGIN IMMEDIATE")
    try:
        import value_gate as VG
        for (ok, why), target in zip(gated, targets):                    # 闸开：去重 + 每片段 3 条封顶（大事可到 5 条）
            if not ok:
                dropped += 1
                k = why.split("：")[0]
                reasons[k] = reasons.get(k, 0) + 1
                continue
            hit, hit2 = ok.pop("hit"), ok.pop("hit2", None)
            ok.pop("category", None)
            ok.pop("priority", None)
            ok.pop("old", None)
            ev_ids = [ok["evidence_id"]] + ([ok["evidence_id2"]] if ok.get("evidence_id2") else [])
            nd = _near_dup(con, ok["subject"], ok["content"], ok["occurred_at"], ev_ids)
            if nd is not None:
                near += 1
                if nd["status"] == "candidate" and not nd["user_pinned"]:    # 只给没人动过的候选补证据链接
                    for e in ev_ids:
                        con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id, role)"
                                    " VALUES (?,?, 'support')", (nd["id"], e))
                continue
            fid = promote(con, subject=ok["subject"], fact_type=ok["type"], content=ok["content"],
                          evidence_ids=ev_ids, valid_from=ok["occurred_at"], confidence=ok["confidence"])
            if fid is None:
                dup += 1
                continue
            if target:                               # 只标「可能过时」（stale.check 里还有主体/本人原话/槽的硬卡）
                import stale
                new_ext = con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]
                stale.mark(con, target[0]["ext_id"], new_ext, slot=target[1], judged_by="llm")
            con.execute("INSERT INTO fact_origin (fact_id, origin, run_id, session_key, quote, prompt_version,"
                        " created_at) VALUES (?,?,?,?,?,?,?)",
                        (fid, origin, run_id, key, ok["quote"], PROMPT_VERSION, now()))
            kept += 1
            if inject:
                inject(kept)
        if advance:
            _save_cursor(con, c, last_evidence_id=chunk[-1]["id"], fail_count=0, first_fail_at=None,
                         next_try_at=None, last_error=None)
        _run_row(con, run_id, kind, key, chunk, n_chunks, meta, kept, dup, near, dropped, "ok", None, t0)
        con.commit()
    except BaseException:
        con.rollback()
        raise
    return {"kept": kept, "dup": dup, "near_dup": near, "dropped": dropped, "reasons": reasons}


def _fail(con, key, c, chunk, err, permanent, now_dt, kind, meta, t0, n_chunks):
    """返回 True = 进了死信箱（游标已越过这一块，可以接着做下一块）。"""
    fc = c["fail_count"] + 1
    first = _t(c["first_fail_at"]) or now_dt
    dead = (fc >= MAX_ATTEMPTS or (now_dt - first) >= dt.timedelta(hours=DEAD_AFTER_H)
            or (permanent and fc >= POISON_ATTEMPTS))
    run_id = "run_" + uuid.uuid4().hex[:12]
    con.commit()
    con.execute("BEGIN IMMEDIATE")
    try:
        if dead:
            con.execute(
                "INSERT INTO distill_dead (session_key, ev_lo, ev_hi, n_msgs, attempts, first_failed_at, dead_at,"
                " last_error, status) VALUES (?,?,?,?,?,?,?,?, 'dead')",
                (key, chunk[0]["id"], chunk[-1]["id"], len(chunk), fc, _iso(first), _iso(now_dt), err[:300]))
            _save_cursor(con, c, last_evidence_id=chunk[-1]["id"], fail_count=0, first_fail_at=None,
                         next_try_at=None, last_error=None)
        else:
            wait = BACKOFF_MIN[min(fc - 1, len(BACKOFF_MIN) - 1)]
            _save_cursor(con, c, fail_count=fc, first_fail_at=_iso(first),
                         next_try_at=_iso(now_dt + dt.timedelta(minutes=wait)), last_error=err[:300])
        _run_row(con, run_id, kind, key, chunk, n_chunks, meta, 0, 0, 0, 0, "dead" if dead else "failed", err, t0)
        con.commit()
    except BaseException:
        con.rollback()
        raise
    return dead


# ── 主流程 ───────────────────────────────────────────────────────────────────
def due_sessions(con, now_dt: dt.datetime, only: str | None = None) -> list[dict]:
    """哪些会话到点了。返回 [{key, kind, rows, state}]（state 不是 due 的也列出来，plan 用）。"""
    base = baseline(con)
    base = 0 if base is None else base
    keys = [r[0] for r in con.execute(
        "SELECT DISTINCT source_path FROM evidence WHERE id>? AND source_kind='chat' AND source_path IS NOT NULL",
        (base,))]
    out = []
    for key in keys:
        if only and key != only:
            continue
        c = _cursor(con, key)
        lo = max(c["last_evidence_id"], base)
        rows = _rows(con, key, lo)
        if not rows:
            continue
        times = [(_t(r["occurred_at"]) or _t(r["captured_at"])) for r in rows]
        times = [t for t in times if t]
        last_t = max(times) if times else now_dt
        first_t = min(times) if times else now_dt
        nxt = _t(c["next_try_at"])
        if nxt and now_dt < nxt:
            out.append({"key": key, "kind": None, "rows": rows, "state": "backoff", "cursor": c})
            continue
        idle = now_dt - last_t >= dt.timedelta(minutes=IDLE_MIN)
        cap = (now_dt - first_t >= dt.timedelta(hours=CAP_HOURS)) or len(rows) >= CAP_MSGS
        if idle:
            out.append({"key": key, "kind": "idle", "rows": rows, "state": "due", "cursor": c})
        elif cap:
            cut = now_dt - dt.timedelta(minutes=SETTLE_MIN)
            ok_rows = [r for r in rows if ((_t(r["occurred_at"]) or _t(r["captured_at"])) or now_dt) <= cut]
            if ok_rows:
                out.append({"key": key, "kind": "cap", "rows": ok_rows, "state": "due", "cursor": c})
            else:
                out.append({"key": key, "kind": None, "rows": rows, "state": "settling", "cursor": c})
        else:
            out.append({"key": key, "kind": None, "rows": rows, "state": "waiting", "cursor": c})
    return out


def run(con, *, now_dt: dt.datetime | None = None, caller=None, only: str | None = None,
        max_chunks: int = MAX_CHUNKS, inject=None) -> dict:
    """处理所有到点的会话。caller(text) -> (facts, meta)，测试里换成假的。"""
    caller = caller or call_model_ex
    now_dt = now_dt or dt.datetime.now(dt.timezone.utc)
    tot = {"sessions": 0, "chunks": 0, "kept": 0, "dup": 0, "near_dup": 0, "dropped": 0, "failed": 0,
           "dead": 0, "backoff": 0}
    for s in due_sessions(con, now_dt, only):
        if s["state"] == "backoff":
            tot["backoff"] += 1
            continue
        if s["kind"] is None:
            continue
        tot["sessions"] += 1
        key, c, kind = s["key"], s["cursor"], s["kind"]
        rendered = FE.prefilter_evs(_render_rows(con, s["rows"]))    # 值得记闸开：心跳/应答/工具输出等不送模型
        chunks = list(FE.chunks(rendered))
        if not chunks and s["rows"]:                                      # 整段都被预过滤掉：游标直接越过，不堵后面
            _save_cursor(con, c, last_evidence_id=s["rows"][-1]["id"], fail_count=0, first_fail_at=None,
                         next_try_at=None, last_error=None)
            con.commit()
            continue
        for n, chunk in enumerate(chunks):
            if n >= max_chunks:
                break
            t0 = dt.datetime.now(dt.timezone.utc)
            ctx = _context(con, key, chunk[0]["id"])
            run_id = "run_" + uuid.uuid4().hex[:12]
            try:
                facts, meta = caller(_prompt_text(ctx, chunk))
                res = _write_chunk(con, key, chunk, facts, run_id, kind, c, meta, t0, len(chunks), inject=inject)
            except BaseException as e:                              # noqa: BLE001
                if isinstance(e, KeyboardInterrupt):
                    raise
                con.rollback() if con.in_transaction else None
                c = _cursor(con, key)
                perm = isinstance(e, PermanentError)
                dead = _fail(con, key, c, chunk, f"{type(e).__name__}: {e}", perm, now_dt, kind, None, t0,
                             len(chunks))
                c = _cursor(con, key)
                tot["failed"] += 1
                if dead:
                    tot["dead"] += 1
                    continue                       # 越过这一块，接着做下一块
                break                              # 退避：这个会话本轮到此为止
            c = _cursor(con, key)
            tot["chunks"] += 1
            for k in ("kept", "dup", "near_dup", "dropped"):
                tot[k] += res[k]
    return tot


# ── 死信箱 ───────────────────────────────────────────────────────────────────
def dead_list(con, status="dead"):
    return [dict(r) for r in con.execute("SELECT * FROM distill_dead WHERE status=? ORDER BY dead_id", (status,))]


def dead_retry(con, dead_id: int, *, caller=None, now_dt=None) -> dict:
    """人工重试一个死信：重新蒸这一段（游标不动，它早就越过了）。成功标 retried，失败仍是 dead。"""
    caller = caller or call_model_ex
    d = con.execute("SELECT * FROM distill_dead WHERE dead_id=? AND status='dead'", (dead_id,)).fetchone()
    if not d:
        return {"ok": False, "why": "没有这条（或已处理）"}
    rows = _rows(con, d["session_key"], d["ev_lo"] - 1, d["ev_hi"])
    chunk = _render_rows(con, rows)
    if not chunk:
        return {"ok": False, "why": "这段证据不在了"}
    t0 = dt.datetime.now(dt.timezone.utc)
    run_id = "run_" + uuid.uuid4().hex[:12]
    try:
        facts, meta = caller(_prompt_text(_context(con, d["session_key"], chunk[0]["id"]), chunk))
        c = _cursor(con, d["session_key"])
        res = _write_chunk(con, d["session_key"], chunk, facts, run_id, "manual", c, meta, t0, 1, advance=False)
    except BaseException as e:                                       # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        con.rollback() if con.in_transaction else None
        con.execute("UPDATE distill_dead SET attempts=attempts+1, last_error=? WHERE dead_id=?",
                    (f"{type(e).__name__}: {e}"[:300], dead_id))
        con.commit()
        return {"ok": False, "why": f"又失败了：{type(e).__name__}"}
    con.execute("UPDATE distill_dead SET status='retried', resolved_at=?, resolved_by='manual' WHERE dead_id=?",
                (now(), dead_id))
    con.commit()
    return {"ok": True, **res}


def dead_drop(con, dead_id: int) -> bool:
    """丢掉 = 标成放弃，不是删；原话 evidence 还在。"""
    n = con.execute("UPDATE distill_dead SET status='dropped', resolved_at=?, resolved_by='manual'"
                    " WHERE dead_id=? AND status='dead'", (now(), dead_id)).rowcount
    con.commit()
    return bool(n)


# ── 审计 ─────────────────────────────────────────────────────────────────────
def audit(con, now_dt=None) -> dict:
    now_dt = now_dt or dt.datetime.now(dt.timezone.utc)
    ds = due_sessions(con, now_dt)
    states: dict[str, int] = {}
    for s in ds:
        states[s["state"]] = states.get(s["state"], 0) + 1
    base = baseline(con) or 0
    lag = con.execute("SELECT COALESCE(MAX(id),0) FROM evidence").fetchone()[0]
    out = {"baseline": baseline(con), "sessions_pending": states,
           "dead": len(dead_list(con)), "in_backoff": states.get("backoff", 0),
           "evidence_after_baseline": con.execute("SELECT COUNT(*) FROM evidence WHERE id>?", (base,)).fetchone()[0],
           "max_evidence_id": lag}
    try:
        import derive_repair
        out["derive_pending"] = derive_repair.pending(con)
    except Exception:                                                # noqa: BLE001
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="默认：只看到点的会话，不连模型不写库")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--init-baseline", action="store_true")
    g.add_argument("--audit", action="store_true")
    g.add_argument("--dead-retry", type=int)
    g.add_argument("--dead-drop", type=int)
    ap.add_argument("--force", action="store_true", help="--init-baseline 已设过时强制重设")
    ap.add_argument("--db")
    args = ap.parse_args()
    con = connect(args.db) if args.db else connect()
    if args.init_baseline:
        print(f"基线 evidence.id = {init_baseline(con, args.force)}")
        return 0
    if args.audit:
        a = audit(con)
        print("审计：" + json.dumps(a, ensure_ascii=False))
        for d in dead_list(con):
            print(f"  死信 #{d['dead_id']} 会话…{str(d['session_key'])[-14:]} 证据 {d['ev_lo']}–{d['ev_hi']} "
                  f"{d['n_msgs']} 条 试了 {d['attempts']} 次：{d['last_error']}")
        return 0
    if args.dead_drop is not None:
        print("已丢弃" if dead_drop(con, args.dead_drop) else "没有这条")
        return 0
    if args.dead_retry is not None:
        print(json.dumps(dead_retry(con, args.dead_retry), ensure_ascii=False))
        return 0
    if baseline(con) is None:
        print("还没设基线：先 --init-baseline（不碰历史）", file=sys.stderr)
        return 3 if args.apply else 0
    if not args.apply:
        for s in due_sessions(con, dt.datetime.now(dt.timezone.utc)):
            print(f"  …{s['key'][-14:]} {s['state']} {s['kind'] or ''} {len(s['rows'])} 条")
        print("（--plan：没连模型、没写库；真跑加 --apply）")
        return 0
    if not llm.routes_for("extract"):
        print("没配模型路由，退出", file=sys.stderr)
        return 2
    t = run(con)
    print(f"切片：会话 {t['sessions']}，块 {t['chunks']}，留下 {t['kept']}，重复 {t['dup']}，"
          f"近重复 {t['near_dup']}，丢弃 {t['dropped']}，失败 {t['failed']}（进死信箱 {t['dead']}），退避中 {t['backoff']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
