#!/usr/bin/env python3
"""收集 Obsidian 仓库里她勾上的事实，可逆地收起（facts_archive 登记，库里的行不删）。

    python3 maintenance/obsidian_collect.py --vault <仓库>                  # 默认：只列要收起哪些，不动库
    python3 maintenance/obsidian_collect.py --vault <仓库> --apply          # 真收起（批号 obsidian-YYYYmmdd-HHMMSS）
    python3 maintenance/obsidian_collect.py --restore <批号>                # 整批撤回

收起 ＝ status 改成 uncertain（退出召回）+ facts_archive 登记（category=obsidian_flag，action=archived）+
完整原行存到 archive/<批号>-obsidian.jsonl。**绝不碰 confirmed、user_pinned**（勾了也拒绝，计入 refused）；绝不真删；绝不改成 confirmed。
保险：一次最多 --max 条（缺省 200）；勾的数量超过仓库事实总数的 30% 视为误全选，拒绝（--force 才放行）。
--apply 之后，仓库里已收起的行下次导出就不在了；勾格在库里已不是 candidate 的行直接跳过（计入 skipped）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CATEGORY = "obsidian_flag"
TICK = re.compile(r"^- \[[xX]\] (.*?)\s*\^f(\d+)\s*$")


def collect_ticks(vault: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    d = vault / "事实"
    for p in sorted(d.glob("*.md")) if d.is_dir() else []:
        for line in p.read_text(encoding="utf-8").splitlines():
            m = TICK.match(line)
            if m:
                out[int(m.group(2))] = m.group(1)
    return out


def plan(con, ticks: dict[int, str]) -> dict:
    ok, refused, skipped, missing = [], [], [], []
    for fid in sorted(ticks):
        r = con.execute("SELECT id, status, user_pinned, layer, version, subject, content FROM facts WHERE id=?",
                        (fid,)).fetchone()
        if r is None:
            missing.append(fid)
        elif r["status"] == "confirmed" or r["user_pinned"]:
            refused.append(fid)
        elif r["status"] != "candidate":
            skipped.append(fid)
        else:
            ok.append(r)
    return {"ok": ok, "refused": refused, "skipped": skipped, "missing": missing}


def apply_batch(con, rows, batch: str, archive_dir: Path) -> int:
    from store.store import _bump
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    n = 0
    with open(archive_dir / f"{batch}-obsidian.jsonl", "a", encoding="utf-8") as ex:
        for row in rows:
            cur = con.execute("SELECT status, user_pinned, layer, version FROM facts WHERE id=?", (row["id"],)).fetchone()
            if cur["status"] != "candidate" or cur["user_pinned"]:
                continue                                           # 中途被别人改过：不碰
            full = dict(con.execute("SELECT * FROM facts WHERE id=?", (row["id"],)).fetchone())
            ex.write(json.dumps({"batch": batch, "fact": full}, ensure_ascii=False) + "\n")
            con.execute(
                """INSERT INTO facts_archive (fact_id, batch_id, category, reason, prev_status,
                       prev_layer, prev_version, action, archived_at, actor, approved_by)
                   VALUES (?,?,?,?,?,?,?,'archived',?, 'grok', 'obsidian-tick')""",
                (row["id"], batch, CATEGORY, "她在 Obsidian 里勾了「不对/过时/不要了」", cur["status"],
                 cur["layer"], cur["version"], ts))
            _bump(con, row["id"], cur["version"], status="uncertain")
            n += 1
    con.commit()
    return n


def restore_batch(con, batch: str) -> int:
    from store.store import _bump
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    n = 0
    for a in con.execute("SELECT * FROM facts_archive WHERE batch_id=? AND action='archived'", (batch,)).fetchall():
        done = con.execute("SELECT 1 FROM facts_archive WHERE fact_id=? AND batch_id=? AND action='restored'",
                           (a["fact_id"], batch)).fetchone()
        cur = con.execute("SELECT status, version FROM facts WHERE id=?", (a["fact_id"],)).fetchone()
        if done or cur is None or cur["status"] != "uncertain":
            continue
        _bump(con, a["fact_id"], cur["version"], status=a["prev_status"])
        con.execute(
            """INSERT INTO facts_archive (fact_id, batch_id, category, reason, prev_status, prev_layer, prev_version,
                   action, archived_at, actor, approved_by) VALUES (?,?,?,?,?,?,?,'restored',?, 'grok', 'obsidian-tick')""",
            (a["fact_id"], batch, a["category"], "撤回整批", "uncertain", a["prev_layer"], a["prev_version"], ts))
        n += 1
    con.commit()
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault")
    ap.add_argument("--db", default=str(ROOT / "engram.db"))
    ap.add_argument("--apply", action="store_true", help="真收起（缺省只列）")
    ap.add_argument("--restore", metavar="批号")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--archive-dir", default=str(ROOT / "archive"))
    ap.add_argument("--show", action="store_true", help="dry-run 时打印每条内容（含个人内容，别贴进 TALK/日志）")
    args = ap.parse_args(argv)
    if args.restore:
        from store.store import connect
        con = connect(args.db)
        print(f"撤回 {args.restore}：恢复 {restore_batch(con, args.restore)} 条")
        con.close()
        return 0
    if not args.vault:
        ap.error("--vault 必填")
    ticks = collect_ticks(Path(args.vault))
    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        total = ro.execute("SELECT COUNT(*) FROM facts WHERE status='candidate'").fetchone()[0]
        p = plan(ro, ticks)
    finally:
        ro.close()
    print(f"勾了 {len(ticks)} 条；可收起 {len(p['ok'])}，拒绝(confirmed/钉住) {len(p['refused'])}，"
          f"已不是候选 {len(p['skipped'])}，库里找不到 {len(p['missing'])}")
    if args.show:
        for r in p["ok"]:
            print(f"  #{r['id']} [{r['subject']}] {r['content'][:80]}")
    if len(p["ok"]) > args.max:
        print(f"超过 --max {args.max}，拒绝。")
        return 2
    if total and len(ticks) > 0.3 * total and not args.force:
        print("勾的数量超过候选事实的 30%，像误全选，拒绝（确认无误加 --force）。")
        return 2
    if not args.apply:
        print("（dry-run：没动库。加 --apply 才收起。）")
        return 0
    from store.store import connect
    batch = "obsidian-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    con = connect(args.db)
    n = apply_batch(con, p["ok"], batch, Path(args.archive_dir))
    con.close()
    print(f"已收起 {n} 条，批号 {batch}；撤回：python3 maintenance/obsidian_collect.py --restore {batch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
