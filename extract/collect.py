#!/usr/bin/env python3
"""Lumen Engram · 原话采集器（RAW_ONLY 的「只存原话」那一步，任务三，2026-10-28）

每一条说出口的话原样落进 evidence：谁说的、哪个会话（source_path）、第几行、什么时候、原文。
**没有模型调用，没有门控，不写 facts。** 事实由 extract/distill_slice.py 在会话安静之后一次性整理。

沿用 from_channel.harvest 的认人规矩（只有 role=user 里的 channel 块才算入站、认不出的是 unknown:*、
后台会话整份跳过、assistant 侧伪造的 channel 块只会被记成「我的输出」），一个字没改；
说话人再过 speakers.normalize_actor：认不出的（unknown:*）计数后跳过，不入库。

幂等：同一行（source_path + 行号）里同一说话人、同一段话只存一次；之前 from_evidence 带「用户（时间）：」前缀存过的
也算已有，不再写第二份。游标 ingest_cursor(source_path, line_no, file_size, file_mtime) 只负责提速：
文件没变就整份跳过；丢了游标也不影响正确性（回退成全量读 + 幂等）。

用法：
    python3 -m extract.collect [jsonl ...] [--since-days 3] [--full] [--dry-run] [--db PATH]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import speakers                                                    # noqa: E402
from extract.from_channel import PROJ, beijing, harvest           # noqa: E402
from store.store import add_evidence, connect, now, record_tier   # noqa: E402
import redact   # noqa: E402


def _stamps(path: Path) -> dict[int, str]:
    out = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for n, line in enumerate(fh, 1):
            m = re.search(r'"timestamp"\s*:\s*"([^"]+)"', line)
            if m:
                out[n] = m.group(1)
    return out


def collect_file(con, path: Path, *, full: bool = False, dry: bool = False) -> dict:
    path = Path(path)
    tally = {"new": 0, "have": 0, "unknown": 0, "skipped_file": 0, "relabeled": 0}
    st = path.stat()
    cur = con.execute("SELECT line_no, file_size, file_mtime FROM ingest_cursor WHERE source_path=?",
                      (str(path),)).fetchone()
    start = 0
    if cur and not full:
        if cur["file_size"] == st.st_size and cur["file_mtime"] == st.st_mtime:
            tally["skipped_file"] = 1
            return tally
        start = cur["line_no"] if st.st_size >= (cur["file_size"] or 0) else 0
    have: dict[str, list[tuple[str, str]]] = {}
    for r in con.execute("SELECT source_locator, source_actor, content FROM evidence WHERE source_path=?",
                         (str(path),)):
        have.setdefault(r["source_locator"], []).append((r["source_actor"], r["content"] or ""))
    stamps = _stamps(path)
    max_line = start
    for line_no, actor, body, ts in harvest(path):
        max_line = max(max_line, line_no)
        if line_no <= start and not full:
            continue
        actor = speakers.normalize_actor(actor)
        if actor is None:
            tally["unknown"] += 1
            continue
        # 最后一道：正文本身是 <channel user=… user_id=…> 转发包装时，说话人以包装为准（bot 转发不能标成她）；
        # 认不出的 user 计数后跳过，和 unknown:* 一样。
        actor, note = speakers.reconcile(actor, body)
        if actor is None:
            tally["unknown"] += 1
            continue
        if note == "relabeled":
            tally["relabeled"] += 1
        loc = f"L{line_no}"
        flat = body.replace("\n", " ")
        rbody, rcnt = redact.redact(body)       # 库里存的是打码后的（新）或原样的（老），两种都认
        rflat = rbody.replace("\n", " ")
        if any(a == actor and (c == body or c == rbody or c.endswith("：" + flat) or c.endswith("：" + rflat))
               for a, c in have.get(loc, [])):
            tally["have"] += 1
            continue
        if dry:
            tally["new"] += 1
            continue
        redact.note("collect", rcnt)            # 只记计数，不记原文
        stamp = ts or stamps.get(line_no)
        when_utc = beijing(stamp)[0] if stamp else None
        ev = add_evidence(con, source_kind="chat", source_actor=actor, source_path=str(path),
                          source_locator=loc, content=rbody, occurred_at=when_utc,
                          verification_status="verified")
        record_tier(con, ev, "primary", f"{path}:{loc}")
        have.setdefault(loc, []).append((actor, body))
        tally["new"] += 1
    if not dry:
        con.execute(
            "INSERT INTO ingest_cursor (source_path, line_no, file_size, file_mtime, updated_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(source_path) DO UPDATE SET line_no=excluded.line_no, file_size=excluded.file_size,"
            " file_mtime=excluded.file_mtime, updated_at=excluded.updated_at",
            (str(path), max_line, st.st_size, st.st_mtime, now()))
        con.commit()
    return tally


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--since-days", type=float, default=3, help="只扫最近这么多天动过的文件（默认 3）")
    ap.add_argument("--full", action="store_true", help="忽略游标，全量读（幂等）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db")
    args = ap.parse_args()
    try:
        import flags
        if flags.on("RAW_ONLY") and not flags.on("DISTILL_SLICE"):
            print("警告：RAW_ONLY=1 但 DISTILL_SLICE=0——只存原话、没有任何东西把它们变成事。", file=sys.stderr)
    except Exception:                                              # noqa: BLE001
        pass
    import time
    files = args.files or sorted(p for p in PROJ.glob("*.jsonl")
                                 if time.time() - p.stat().st_mtime <= args.since_days * 86400)
    con = connect(args.db) if args.db else connect()
    tot = {"new": 0, "have": 0, "unknown": 0, "skipped_file": 0, "relabeled": 0}
    for p in files:
        t = collect_file(con, p, full=args.full, dry=args.dry_run)
        for k in tot:
            tot[k] += t[k]
    print(f"采集 文件 {len(files)}（没变化跳过 {tot['skipped_file']}）：新增 {tot['new']}，已有 {tot['have']}，"
          f"认不出说话人跳过 {tot['unknown']}，按转发包装改标 {tot['relabeled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
