#!/usr/bin/env python3
"""说话人标注修正层：算出历史上标错说话人的证据，写进 evidence_actor_fix（overlay），不改 evidence 一个字。

用法：
    python3 maintenance/build_actor_fix.py DB                    # 默认：只读，打印计数（不写库）
    python3 maintenance/build_actor_fix.py DB --apply            # 只往 evidence_actor_fix 里插新行（INSERT OR IGNORE，可重复跑）
    python3 maintenance/build_actor_fix.py DB --report DIR       # 另外写清单：DIR/actor-fix-rows-raw.md（原文，仅密钥打码，600）
                                                                 #            DIR/actor-fix-rows.md（打码样本，不含原话）
最初只在副本上跑；现正式库已建好该 overlay（evidence_actor_fix 有记录），需要重跑由用户/助手决定。evidence 的触发器（只增不改不删）照常生效，本脚本没有任何 UPDATE/DELETE evidence。
"""
import argparse
import collections
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import actor_fix  # noqa: E402
import persona  # noqa: E402
import redact  # noqa: E402

NAMES = re.compile(persona.mask_re())


def mask(s: str, n: int = 20) -> str:
    s = redact.redact((s or "").replace("\n", " "))[0]
    s = re.sub(r"〔[^〕]*〕", "〔密〕", s)
    s = NAMES.sub("〔名〕", s)
    s = re.sub(r"\d", "#", s)
    return s[:n]


def raw(s: str, n: int = 60) -> str:
    s = redact.redact((s or "").replace("\n", " ⏎ "))[0]
    return s if len(s) <= n else s[:n] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report")
    a = ap.parse_args()
    ro = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    fixes = actor_fix.compute_all(ro)
    by = collections.Counter((f["kind"], f["original_actor"], f["effective_actor"], f["block_actor"], f["corroborated"]) for f in fixes.values())
    print(f"需要改判的证据 {len(fixes)} 行：")
    for k, v in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {v:>3}  kind={k[0]} 原标={k[1]} 改判={k[2]} 块里的人={k[3]} 佐证={k[4]}")
    if a.report:
        d = Path(a.report)
        d.mkdir(parents=True, exist_ok=True)
        rows = {r["id"]: r for r in ro.execute("select id, source_actor, content, source_path, source_locator from evidence")
                if r["id"] in fixes}
        head = ["# 说话人标注 overlay：受影响的证据行", "", f"共 {len(fixes)} 行（evidence 本身一字未动；改判只在 evidence_actor_fix）。", ""]
        for k, v in sorted(by.items(), key=lambda x: -x[1]):
            head.append(f"- {v} 行：kind={k[0]}，原标 {k[1]} → " + (f"改判 {k[2]}" if k[0] == "leading-block" else f"块里是 {k[3]}，佐证={k[4]}"))
        for name, rawmode in (("actor-fix-rows-raw.md", True), ("actor-fix-rows.md", False)):
            L = list(head) + ["", "## 逐行" if rawmode else "## 打码样本（每类前 3 行）", ""]
            seen = collections.Counter()
            for eid in sorted(fixes):
                f, r = fixes[eid], rows[eid]
                key = (f["kind"], f["original_actor"], f["effective_actor"], f["block_actor"], f["corroborated"])
                seen[key] += 1
                if not rawmode and seen[key] > 3:
                    continue
                txt = raw(r["content"], 60) if rawmode else mask(r["content"], 20)
                L.append(f"- [{eid}] {f['kind']} · 原标 {f['original_actor']} → " + (f"{f['effective_actor']}" if f["kind"] == "leading-block" else f"{f['block_actor']}（佐证={f['corroborated']}）")
                         + (f" · {raw(f['reason'], 120)}" if rawmode else "") + f" · {txt}")
            p = d / name
            p.write_text("\n".join(L) + "\n", encoding="utf-8")
            os.chmod(p, 0o600)
        print(f"清单已写 {d}/actor-fix-rows*.md")
    if a.apply:
        from store.store import connect
        con = connect(a.db)
        print("写入：", actor_fix.build(con))
        con.close()
    else:
        print("（只读：没写库；要写加 --apply）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
