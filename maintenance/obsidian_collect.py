#!/usr/bin/env python3
"""收集 Obsidian 仓库里她点的两种标记：「不对/过时/不要了」可逆收起；「📌钉住」可逆提升为已确认。

    python3 maintenance/obsidian_collect.py --vault <仓库>                  # 默认：只列要收起/要钉住哪些，不动库
    python3 maintenance/obsidian_collect.py --vault <仓库> --apply          # 真处理（收起批号 obsidian-…，钉住批号 obsidian-pin-…）
    python3 maintenance/obsidian_collect.py --restore <批号>                # 整批撤回（收起、钉住两种批号都认）

收起 ＝ status 改成 uncertain（退出召回）+ facts_archive 登记（category=obsidian_flag，action=archived）+
完整原行存到 archive/<批号>-obsidian.jsonl。**绝不碰 confirmed、user_pinned**；绝不真删。
钉住 ＝ 只对 candidate：status→confirmed、user_pinned→1（一次乐观锁更新）；追加一条证据
（source_kind=human_stated、说话人=她、内容「pinned by user via Obsidian」，证据只增不改）并挂到这条事实上；
facts_archive 登记（category=obsidian_pin，action=registered_only，记下原状态）；原行存到 archive/<批号>-obsidian-pin.jsonl。
撤回钉住＝状态改回原状态、user_pinned→0，登记 restored；证据留着（只增不删），登记里写明已撤回。
同一条事实两种框都点了（矛盾）：两个都不执行，列在「矛盾」里。
保险：一次最多收起 --max 条（缺省 200）、钉住 --max-pins 条（缺省 100）；某一种的数量超过候选事实总数的 30% 视为误全选，拒绝（--force 才放行）。
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
sys.path.insert(0, str(ROOT / "maintenance"))

CATEGORY = "obsidian_flag"
PIN_CATEGORY = "obsidian_pin"
PIN_EVIDENCE = "pinned by user via Obsidian"
PARENT = re.compile(r"^- \[([ xX])\] (.*?)\s*\^f(\d+)\s*$")
PINROW = re.compile(r"^[ \t]+- \[([ xX])\] .*\^p(\d+)\s*$")
PIN_EMOJI = "📌"


def collect_marks(vault: Path) -> tuple[dict[int, str], dict[int, str]]:
    """递归读 事实/ 下所有 .md（含旧布局、冲突副本），取并集。返回 ({勾「不对」的 id: 文字}, {钉住的 id: 文字})。"""
    ticks: dict[int, str] = {}
    pins: dict[int, str] = {}
    d = vault / "事实"
    for p in sorted(d.rglob("*.md")) if d.is_dir() else []:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            m = PARENT.match(line)
            if m:
                fid, body = int(m.group(3)), m.group(2)
                if m.group(1) in "xX":
                    ticks[fid] = body
                if body.lstrip().startswith(PIN_EMOJI):
                    pins[fid] = body
                continue
            m = PINROW.match(line)
            if m and m.group(1) in "xX":
                pins.setdefault(int(m.group(2)), "")
    return ticks, pins


def collect_ticks(vault: Path) -> dict[int, str]:          # 向后兼容
    return collect_marks(vault)[0]


def plan(con, ticks: dict[int, str], pins: dict[int, str] | None = None) -> dict:
    pins = pins or {}
    conflict = sorted(set(ticks) & set(pins))
    ok, refused, skipped, missing = [], [], [], []
    for fid in sorted(set(ticks) - set(conflict)):
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
    pin_ok, pin_already, pin_skipped, pin_missing = [], [], [], []
    for fid in sorted(set(pins) - set(conflict)):
        r = con.execute("SELECT id, status, user_pinned, layer, version, subject, content FROM facts WHERE id=?",
                        (fid,)).fetchone()
        if r is None:
            pin_missing.append(fid)
        elif r["content"].lstrip().startswith(PIN_EMOJI) and not pins[fid] == "":
            pin_skipped.append(fid)                       # 事实本身就以 📌 开头：文字里的 📌 不算她打的
        elif r["status"] == "confirmed" or r["user_pinned"]:
            pin_already.append(fid)
        elif r["status"] != "candidate":
            pin_skipped.append(fid)
        else:
            pin_ok.append(r)
    return {"ok": ok, "refused": refused, "skipped": skipped, "missing": missing, "conflict": conflict,
            "pin_ok": pin_ok, "pin_already": pin_already, "pin_skipped": pin_skipped, "pin_missing": pin_missing}


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


def apply_pins(con, rows, batch: str, archive_dir: Path) -> int:
    """candidate → confirmed + user_pinned=1，追加一条「她钉住」的证据。其余一概不碰。"""
    import persona
    from store.store import _bump, add_evidence
    archive_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    n = 0
    with open(archive_dir / f"{batch}-obsidian-pin.jsonl", "a", encoding="utf-8") as ex:
        for row in rows:
            cur = con.execute("SELECT status, user_pinned, layer, version FROM facts WHERE id=?", (row["id"],)).fetchone()
            if cur["status"] != "candidate" or cur["user_pinned"]:
                continue
            full = dict(con.execute("SELECT * FROM facts WHERE id=?", (row["id"],)).fetchone())
            ex.write(json.dumps({"batch": batch, "fact": full}, ensure_ascii=False) + "\n")
            ev = add_evidence(con, source_kind="human_stated", source_actor=persona.USER,
                              source_locator=f"obsidian-pin:{batch}#f{row['id']}",
                              content=f"{PIN_EVIDENCE}（事实 #{row['id']}，批号 {batch}）",
                              source_path="obsidian-vault", occurred_at=dt.datetime.now(dt.timezone.utc).isoformat())
            con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,'support')",
                        (row["id"], ev))
            con.execute(
                """INSERT INTO facts_archive (fact_id, batch_id, category, reason, prev_status,
                       prev_layer, prev_version, action, archived_at, actor, approved_by)
                   VALUES (?,?,?,?,?,?,?,'registered_only',?, 'grok', 'obsidian-pin')""",
                (row["id"], batch, PIN_CATEGORY, PIN_EVIDENCE, cur["status"], cur["layer"], cur["version"], ts))
            _bump(con, row["id"], cur["version"], status="confirmed", user_pinned=1)
            n += 1
    con.commit()
    return n


def restore_batch(con, batch: str) -> int:
    from store.store import _bump
    ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    n = 0
    rows = con.execute("SELECT * FROM facts_archive WHERE batch_id=? AND (action='archived' OR "
                       "(action='registered_only' AND category=?))", (batch, PIN_CATEGORY)).fetchall()
    for a in rows:
        done = con.execute("SELECT 1 FROM facts_archive WHERE fact_id=? AND batch_id=? AND action='restored'",
                           (a["fact_id"], batch)).fetchone()
        cur = con.execute("SELECT status, user_pinned, version FROM facts WHERE id=?", (a["fact_id"],)).fetchone()
        if done or cur is None:
            continue
        if a["category"] == PIN_CATEGORY:
            if cur["status"] != "confirmed" or not cur["user_pinned"]:
                continue
            _bump(con, a["fact_id"], cur["version"], status=a["prev_status"], user_pinned=0)
            why, cur_status = "撤回钉住（证据留着，只增不删）", "confirmed"
        else:
            if cur["status"] != "uncertain":
                continue
            _bump(con, a["fact_id"], cur["version"], status=a["prev_status"])
            why, cur_status = "撤回整批", "uncertain"
        con.execute(
            """INSERT INTO facts_archive (fact_id, batch_id, category, reason, prev_status, prev_layer, prev_version,
                   action, archived_at, actor, approved_by) VALUES (?,?,?,?,?,?,?,'restored',?, 'grok', ?)""",
            (a["fact_id"], batch, a["category"], why, cur_status, a["prev_layer"], a["prev_version"], ts,
             "obsidian-pin" if a["category"] == PIN_CATEGORY else "obsidian-tick"))
        n += 1
    con.commit()
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault")
    ap.add_argument("--db", default=str(ROOT / "engram.db"))
    ap.add_argument("--apply", action="store_true", help="真处理（缺省只列）")
    ap.add_argument("--restore", metavar="批号")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--max-pins", type=int, default=100)
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
    ticks, pins = collect_marks(Path(args.vault))
    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        total = ro.execute("SELECT COUNT(*) FROM facts WHERE status='candidate'").fetchone()[0]
        p = plan(ro, ticks, pins)
    finally:
        ro.close()
    print(f"【不对/过时/不要了】勾了 {len(ticks)} 条；可收起 {len(p['ok'])}，拒绝(confirmed/钉住) {len(p['refused'])}，"
          f"已不是候选 {len(p['skipped'])}，库里找不到 {len(p['missing'])}")
    print(f"【📌钉住】点了 {len(pins)} 条；可钉住 {len(p['pin_ok'])}，早就是已确认/钉住 {len(p['pin_already'])}，"
          f"已不是候选 {len(p['pin_skipped'])}，库里找不到 {len(p['pin_missing'])}")
    if p["conflict"]:
        print(f"矛盾（又勾「不对」又点「钉住」）{len(p['conflict'])} 条，两个都不执行：{p['conflict'][:20]}")
    if args.show:
        for r in p["ok"]:
            print(f"  收起 #{r['id']} [{r['subject']}] {r['content'][:80]}")
        for r in p["pin_ok"]:
            print(f"  钉住 #{r['id']} [{r['subject']}] {r['content'][:80]}")
    if len(p["ok"]) > args.max:
        print(f"收起超过 --max {args.max}，拒绝。")
        return 2
    if len(p["pin_ok"]) > args.max_pins:
        print(f"钉住超过 --max-pins {args.max_pins}，拒绝。")
        return 2
    if total and not args.force and (len(ticks) > 0.3 * total or len(pins) > 0.3 * total):
        print("勾/钉的数量超过候选事实的 30%，像误全选，拒绝（确认无误加 --force）。")
        return 2
    if not args.apply:
        print("（dry-run：没动库。加 --apply 才处理。）")
        return 0
    from store.store import connect
    con = connect(args.db)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if p["ok"]:
        batch = "obsidian-" + stamp
        n = apply_batch(con, p["ok"], batch, Path(args.archive_dir))
        if n:                                              # 勾掉＝删除：整批进回收站，15 天后到期归档（maintenance/trash.py）
            import trash
            trash.ensure(con)
            trash.register(con, batch, kind="obsidian_flag", opened_at=None, approved_by="obsidian-tick")
            con.commit()
        print(f"已收起 {n} 条，批号 {batch}；撤回：python3 maintenance/obsidian_collect.py --restore {batch}")
    if p["pin_ok"]:
        pbatch = "obsidian-pin-" + stamp
        n = apply_pins(con, p["pin_ok"], pbatch, Path(args.archive_dir))
        print(f"已钉住 {n} 条（提升为已确认），批号 {pbatch}；撤回：python3 maintenance/obsidian_collect.py --restore {pbatch}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
