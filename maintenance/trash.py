#!/usr/bin/env python3
# 2026-11-08 起：原话复述类流水可真删（备份＋导出 jsonl.gz 后物理删除，见 backups/）；其他类别仍按原规则，永不真删。
"""Lumen Engram · 回收站（2026-11-03；保留期 15 天，默认永不真删）

和已有东西的关系：
  · facts_archive 是**总账**：所有状态迁移（系统清理 K/D/US/GA…、取代 E1、作废 N1、Obsidian 收起）都在这里登一行，
    只追加、可撤回（action=restored）。它不等于回收站。
  · 回收站＝总账里「被删掉的记忆」那几批（trash_batches 登记）：Obsidian 勾掉的 obsidian-* 批、人工批准直接作废的批、
    以后用本脚本 delete 删的批。每批从登记时刻起放 15 天，期间整批或单条可撤回。
    不进回收站的：取代（E1：旧行是新行的历史版本，新行在用）、系统审计时置 uncertain 的清理批（B202610xx-*，是「靠边站」不是删除）。
  · 到期＝整批（含只被它们引用的派生行、证据原文）导出到 archive/trash/<批号>.jsonl.gz（带 sha256 清单，读回校验），
    登记 trash_expired。**主库一行不删**（facts 的删除触发器照旧 ABORT）；事实早已退出召回（status 不是 candidate/confirmed），
    到期后再标 layer=cold。真要从主库瘦身，是另一个决定，不在这里。
  · 借的做法：mem0 的软删除 + 历史表（删除只记一笔、内容不动、可回放）。

用法（缺省都只看，不写库；--apply 才写；调用方负责 flock /srv/engram-live/.ingest.lock 和备份）：
    python3 maintenance/trash.py list
    python3 maintenance/trash.py register <批号> [--opened ISO时间] --approved-by 谁 [--apply]   # 把已有的作废批放进回收站
    python3 maintenance/trash.py delete --ids 1,2,3 --reason 为什么 --approved-by 谁 [--apply]     # 删除＝进回收站
    python3 maintenance/trash.py restore <批号> [--ids 1,2] [--force] [--apply]                   # 撤回（到期后要 --force）
    python3 maintenance/trash.py expire [--now ISO时间] [--apply]                                 # 到期归档（手动跑；定时另配）
绝不碰 confirmed / user_pinned；绝不 DELETE。
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEEP_DAYS = 15
LIVE = ("candidate", "confirmed")
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS trash_batches (
  batch_id    TEXT PRIMARY KEY,                -- = facts_archive.batch_id
  kind        TEXT NOT NULL,                   -- obsidian_flag / approved_void / manual_delete
  opened_at   TEXT NOT NULL,                   -- 15 天从这里算
  expires_at  TEXT NOT NULL,
  approved_by TEXT,
  note        TEXT,
  created_at  TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS trash_expired (
  batch_id     TEXT NOT NULL REFERENCES trash_batches (batch_id),
  fact_id      INTEGER NOT NULL,
  archive_path TEXT NOT NULL,
  archive_sha256 TEXT NOT NULL,
  expired_at   TEXT NOT NULL,
  UNIQUE (batch_id, fact_id))""",
    "CREATE TRIGGER IF NOT EXISTS trash_batches_no_delete BEFORE DELETE ON trash_batches BEGIN SELECT RAISE(ABORT, 'trash_batches cannot be deleted'); END",
    "CREATE TRIGGER IF NOT EXISTS trash_batches_no_update BEFORE UPDATE ON trash_batches BEGIN SELECT RAISE(ABORT, 'trash_batches is append-only'); END",
    "CREATE TRIGGER IF NOT EXISTS trash_expired_no_delete BEFORE DELETE ON trash_expired BEGIN SELECT RAISE(ABORT, 'trash_expired cannot be deleted'); END",
    "CREATE TRIGGER IF NOT EXISTS trash_expired_no_update BEFORE UPDATE ON trash_expired BEGIN SELECT RAISE(ABORT, 'trash_expired is append-only'); END",
]
# 只被这批事实引用的派生行（导出时带上，便于离线核对/回放）
RELATED = {
    "fact_evidence": "fact_id", "fact_class": "fact_id", "fact_cards": "fact_id", "fact_key_map": "fact_id",
    "fact_origin": "fact_id", "stances": "fact_id", "facts_archive": "fact_id",
}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_ts(s: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def ensure(con) -> None:
    """建表（在当前事务里；干看时随 rollback 一起撤掉）。"""
    if not con.in_transaction:
        con.execute("BEGIN")
    for stmt in SCHEMA:
        con.execute(stmt)


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def batch_rows(con, batch: str):
    """这一批里「被删掉」的事实：facts_archive action=archived，且还没被撤回。"""
    return con.execute(
        """SELECT a.*, f.status AS cur_status, f.user_pinned, f.version AS cur_version, f.valid_to
             FROM facts_archive a JOIN facts f ON f.id = a.fact_id
            WHERE a.batch_id=? AND a.action='archived'
              AND NOT EXISTS (SELECT 1 FROM facts_archive r WHERE r.batch_id=a.batch_id AND r.fact_id=a.fact_id
                              AND r.action='restored')""", (batch,)).fetchall()


def listing(con) -> list[dict]:
    out = []
    for b in con.execute("SELECT * FROM trash_batches ORDER BY opened_at"):
        live = batch_rows(con, b["batch_id"])
        exp = con.execute("SELECT COUNT(*) FROM trash_expired WHERE batch_id=?", (b["batch_id"],)).fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM facts_archive WHERE batch_id=? AND action='archived'",
                            (b["batch_id"],)).fetchone()[0]
        out.append({"batch": b["batch_id"], "kind": b["kind"], "opened_at": b["opened_at"], "expires_at": b["expires_at"],
                    "in_bin": len(live), "restored": total - len(live), "expired": exp,
                    "state": "expired" if exp else ("restored" if not live else "open")})
    return out


def register(con, batch: str, *, kind: str, opened_at: str | None, approved_by: str, note: str = "") -> int:
    rows = batch_rows(con, batch)
    if not rows:
        raise SystemExit(f"{batch}：facts_archive 里没有 action=archived 的行，不能进回收站")
    bad = [r["fact_id"] for r in rows if r["cur_status"] in LIVE or r["user_pinned"]]
    if bad:
        raise SystemExit(f"{batch}：有 {len(bad)} 条现在是活的或钉住的（{bad[:5]}），不收")
    op = opened_at or min(r["archived_at"] for r in rows)
    exp = (parse_ts(op) + dt.timedelta(days=KEEP_DAYS)).isoformat(timespec="seconds")
    con.execute("INSERT INTO trash_batches (batch_id, kind, opened_at, expires_at, approved_by, note, created_at)"
                " VALUES (?,?,?,?,?,?,?)", (batch, kind, op, exp, approved_by, note, now()))
    return len(rows)


def delete(con, ids: list[int], *, reason: str, approved_by: str, batch: str | None = None) -> str:
    """删除＝进回收站：status=invalid + valid_to（同 store.invalidate 的写法），登记 facts_archive + trash_batches。"""
    from store.store import _bump
    batch = batch or "trash-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    ts = now()
    for i in ids:
        f = con.execute("SELECT * FROM facts WHERE id=?", (i,)).fetchone()
        if f is None:
            raise SystemExit(f"没有事实 {i}")
        if f["status"] == "confirmed" or f["user_pinned"]:
            raise SystemExit(f"{i} 是 confirmed/钉住的，不删")
        if f["status"] not in LIVE:
            raise SystemExit(f"{i} 已经是 {f['status']}，不用再删")
        con.execute("""INSERT INTO facts_archive (fact_id,batch_id,category,reason,prev_status,prev_layer,prev_version,
                       action,archived_at,actor,approved_by) VALUES (?,?,?,?,?,?,?,'archived',?,'grok',?)""",
                    (i, batch, "trash", reason, f["status"], f["layer"], f["version"], ts, approved_by))
        _bump(con, i, f["version"], status="invalid", valid_to=ts)
    register(con, batch, kind="manual_delete", opened_at=ts, approved_by=approved_by, note=reason)
    return batch


def restore(con, batch: str, *, ids: list[int] | None = None, force: bool = False) -> int:
    from store.store import _bump
    b = con.execute("SELECT * FROM trash_batches WHERE batch_id=?", (batch,)).fetchone()
    if b is None:
        raise SystemExit(f"{batch} 不在回收站")
    expired = con.execute("SELECT COUNT(*) FROM trash_expired WHERE batch_id=?", (batch,)).fetchone()[0]
    if expired and not force:
        raise SystemExit(f"{batch} 已到期归档，撤回要 --force（主库行还在，能撤）")
    ts, n = now(), 0
    for a in batch_rows(con, batch):
        if ids and a["fact_id"] not in ids:
            continue
        if a["cur_status"] in LIVE:
            continue
        cols = {"status": a["prev_status"]}
        if a["prev_status"] in LIVE and a["valid_to"]:
            cols["valid_to"] = None                     # 删除时补的 valid_to 去掉（活事实 valid_to 本来就是空）
        if a["prev_layer"]:
            cols["layer"] = a["prev_layer"]
        _bump(con, a["fact_id"], a["cur_version"], **cols)
        con.execute("""INSERT INTO facts_archive (fact_id,batch_id,category,reason,prev_status,prev_layer,prev_version,
                       action,archived_at,actor,approved_by) VALUES (?,?,?,?,?,?,?,'restored',?,'grok',?)""",
                    (a["fact_id"], batch, a["category"], "从回收站撤回", a["cur_status"], a["prev_layer"],
                     a["prev_version"], ts, "trash-restore"))
        n += 1
    return n


def _dump(con, fid: int) -> dict:
    con.row_factory = sqlite3.Row
    tabs = _tables(con)
    rec = {"fact": dict(con.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone())}
    for t, col in RELATED.items():
        if t in tabs:
            rec[t] = [dict(r) for r in con.execute(f"SELECT * FROM {t} WHERE {col}=?", (fid,))]
    if "fact_relations" in tabs:
        rec["fact_relations"] = [dict(r) for r in con.execute(
            "SELECT * FROM fact_relations WHERE parent_fact_id=? OR child_fact_id=?", (fid, fid))]
    eids = [r["evidence_id"] for r in rec.get("fact_evidence", [])]
    rec["evidence"] = [dict(r) for r in con.execute(
        f"SELECT * FROM evidence WHERE id IN ({','.join('?' * len(eids))})", eids)] if eids else []
    return rec


def expire(con, *, at: str | None = None, out_dir: Path | None = None, apply: bool = False) -> list[dict]:
    """到期的批：导出 jsonl.gz（读回校验）→ 登记 trash_expired → layer=cold。主库不删。"""
    from store.store import _bump
    t = parse_ts(at) if at else dt.datetime.now(dt.timezone.utc)
    out_dir = out_dir or ROOT / "archive" / "trash"
    done = []
    for b in con.execute("SELECT * FROM trash_batches ORDER BY opened_at").fetchall():
        if parse_ts(b["expires_at"]) > t:
            continue
        if con.execute("SELECT 1 FROM trash_expired WHERE batch_id=?", (b["batch_id"],)).fetchone():
            continue
        rows = batch_rows(con, b["batch_id"])
        if not rows:
            continue
        recs = [_dump(con, r["fact_id"]) for r in rows]
        blob = ("\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in recs) + "\n").encode()
        sha = hashlib.sha256(blob).hexdigest()
        path = out_dir / f"{b['batch_id']}.jsonl.gz"
        info = {"batch": b["batch_id"], "facts": len(recs), "path": str(path), "sha256": sha}
        if apply:
            out_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise SystemExit(f"{path} 已存在，不覆盖")
            with gzip.open(path, "wb") as fh:
                fh.write(blob)
            with gzip.open(path, "rb") as fh:                    # 读回校验
                assert hashlib.sha256(fh.read()).hexdigest() == sha, path
            path.chmod(0o640)
            (out_dir / f"{b['batch_id']}.manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
            ts = now()
            for r in rows:
                con.execute("INSERT INTO trash_expired (batch_id, fact_id, archive_path, archive_sha256, expired_at)"
                            " VALUES (?,?,?,?,?)", (b["batch_id"], r["fact_id"], str(path), sha, ts))
                f = con.execute("SELECT layer, version, status, user_pinned FROM facts WHERE id=?", (r["fact_id"],)).fetchone()
                assert f["status"] not in LIVE and not f["user_pinned"]
                if f["layer"] != "cold":
                    _bump(con, r["fact_id"], f["version"], layer="cold")
        done.append(info)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["list", "register", "delete", "restore", "expire"])
    ap.add_argument("batch", nargs="?")
    ap.add_argument("--db", default=str(ROOT / "engram.db"))
    ap.add_argument("--ids", default="")
    ap.add_argument("--reason", default="")
    ap.add_argument("--approved-by", default="")
    ap.add_argument("--kind", default="approved_void")
    ap.add_argument("--opened")
    ap.add_argument("--now")
    ap.add_argument("--out-dir")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    from store.store import connect
    con = connect(a.db)
    con.row_factory = sqlite3.Row
    ensure(con)
    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    if a.cmd in ("register", "delete") and not a.approved_by:
        raise SystemExit("要写 --approved-by")
    if a.cmd == "list":
        print(json.dumps(listing(con), ensure_ascii=False, indent=1))
    elif a.cmd == "register":
        print("register", a.batch, register(con, a.batch, kind=a.kind, opened_at=a.opened, approved_by=a.approved_by))
    elif a.cmd == "delete":
        print("delete →", delete(con, ids, reason=a.reason or "删除", approved_by=a.approved_by), len(ids))
    elif a.cmd == "restore":
        print("restore", a.batch, restore(con, a.batch, ids=ids or None, force=a.force))
    elif a.cmd == "expire":
        print(json.dumps(expire(con, at=a.now, out_dir=Path(a.out_dir) if a.out_dir else None, apply=a.apply),
                         ensure_ascii=False, indent=1))
    if a.apply and a.cmd != "list":
        con.commit()
    else:
        con.rollback()
        if a.cmd != "list":
            print("（干看：没写库。加 --apply 才写）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
