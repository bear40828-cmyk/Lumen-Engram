#!/usr/bin/env python3
"""把 Engram 的事实导成一个 Obsidian 仓库（只读导出；库一个字不改）。

    python3 maintenance/obsidian_export.py --vault /srv/engram-live/obsidian-vault [--db 库路径] [--handoff-dir 交接只读镜像]

仓库里有：
  README.md           怎么用（给她看的）
  事实/<主体>_<类型>_<id段>.md   一行一条事实（按事实 id 每 500 个一段，同一条永远在同一个文件），行首是复选框：  - [ ] 内容 · 日期 ^f123
                      勾上 [x] ＝「这条不对 / 过时 / 不要了」，之后由 obsidian_collect.py 收起（默认只列、不动库）
                      已确认(confirmed)和她钉住的事实没有复选框（🔒），勾不了
  事实/_总览.md        每个文件多少条
  交接/                交接只读镜像里读得到的文件（index.md、windows/*.md、handoff.txt）的单向拷贝；在这里改了不会回写
  .engram-export.json  上次导出的时间、条数（给脚本自己看）

安全：
  · 打开库用 sqlite3 的 mode=ro URI——**不走 store.connect**（它每次连接都会跑建表脚本）
  · 重新导出前先读出仓库里已勾、还没被收起的格子，导出时保持勾着（不会因为定时重导就丢了她的勾）
  · 只写 --vault 目录；只删 事实/ 下自己上次生成、这次不再生成的文件；文件权限 660、目录 2770（umask 007）
  · 内容原样输出（只把换行折成一行、去掉行首的列表符号），不改一个字
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = 500          # 按事实 id 分段：同一条事实永远落在同一个文件里，新事实只动最新那段（同步时少冲突）
TICK = re.compile(r"^- \[[xX]\] .*\^f(\d+)\s*$")
LINE_ID = re.compile(r"\^f(\d+)\s*$")
MARK = "engram_export: true"


def open_ro(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _safe(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\-]+", "-", s).strip("-") or "x"


def one_line(s: str) -> str:
    s = re.sub(r"\s*\n\s*", " ⏎ ", (s or "").strip())
    return re.sub(r"^[-*•]\s+", "", s)


def ticked_ids(vault: Path) -> set[int]:
    out: set[int] = set()
    d = vault / "事实"
    if not d.is_dir():
        return out
    for p in d.glob("*.md"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                m = TICK.match(line)
                if m:
                    out.add(int(m.group(1)))
        except OSError:
            pass
    return out


def load_facts(con: sqlite3.Connection) -> tuple[list[sqlite3.Row], dict[int, int]]:
    rows = con.execute(
        """SELECT id, subject, fact_type, content, status, user_pinned, valid_from, created_at, layer
           FROM facts WHERE status IN ('candidate','confirmed') AND valid_to IS NULL
           ORDER BY subject, fact_type, COALESCE(valid_from, created_at), id""").fetchall()
    stale: dict[int, int] = {}
    try:
        for r in con.execute("SELECT old_fact_id, MAX(new_fact_id) n FROM fact_stale_marks GROUP BY old_fact_id"):
            stale[r["old_fact_id"]] = r["n"]
    except sqlite3.Error:
        pass
    return rows, stale


def render_line(r: sqlite3.Row, stale: dict[int, int], keep_ticked: set[int]) -> str:
    when = (r["valid_from"] or r["created_at"] or "")[:10]
    tags = []
    if when:
        tags.append(when)
    if r["id"] in stale:
        tags.append(f"⏳可能已过时（有更新的事 #{stale[r['id']]}）")
    if r["layer"] == "cold":
        tags.append("冷")
    tail = (" · " + " · ".join(tags)) if tags else ""
    body = one_line(r["content"]) + tail
    if r["status"] == "confirmed" or r["user_pinned"]:
        return f"- 🔒 {body} ^f{r['id']}"
    box = "x" if r["id"] in keep_ticked else " "
    return f"- [{box}] {body} ^f{r['id']}"


def export_facts(con, vault: Path, now: str) -> dict:
    rows, stale = load_facts(con)
    keep = ticked_ids(vault)
    fdir = vault / "事实"
    fdir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault((r["subject"], r["fact_type"]), []).append(r)
    made: dict[str, int] = {}
    wrote = 0
    for (subj, typ), rs in sorted(groups.items()):
        buckets: dict[int, list[sqlite3.Row]] = {}
        for r in rs:
            buckets.setdefault(r["id"] // BUCKET, []).append(r)
        for b, part in sorted(buckets.items()):
            name = f"{_safe(subj)}_{typ}_{b:03d}.md"
            head = [
                "---", MARK, f"subject: {subj}", f"fact_type: {typ}",
                f"id_range: {b * BUCKET}-{b * BUCKET + BUCKET - 1}", f"count: {len(part)}", "---", "",
                f"# {subj} · {typ} · id {b * BUCKET}–{b * BUCKET + BUCKET - 1}",
                "", "> 勾上 `[x]` ＝ 这条不对 / 过时 / 不要了。勾完告诉助手跑收起脚本；🔒 的是已确认或钉住的，勾不了。", "",
            ]
            lines = [render_line(r, stale, keep) for r in part]
            wrote += _atomic_write(fdir / name, "\n".join(head + lines) + "\n")
            made[name] = len(part)
    # 清掉自己上次生成、这次不再生成的文件
    for p in fdir.glob("*.md"):
        if p.name in made or p.name == "_总览.md":
            continue
        try:
            if MARK in p.read_text(encoding="utf-8")[:300]:
                p.unlink()
        except OSError:
            pass
    ov = ["---", MARK, "---", "", "# 事实总览", "",
          f"共 {len(rows)} 条（候选 + 已确认，不含已被取代/已收起的）。", ""]
    for name, n in made.items():
        ov.append(f"- [[{name[:-3]}]] — {n} 条")
    wrote += _atomic_write(fdir / "_总览.md", "\n".join(ov) + "\n")
    return {"facts": len(rows), "files": len(made), "files_written": wrote, "kept_ticks": len(keep & {r["id"] for r in rows})}


def mirror_handoff(src: Path, vault: Path) -> dict:
    """单向拷贝交接只读镜像里读得到的文件；读不到的只计数、不报错。"""
    dst = vault / "交接"
    got, denied = 0, 0
    if not src.is_dir():
        return {"handoff_copied": 0, "handoff_unreadable": 0, "handoff_dir": False}
    (dst / "windows").mkdir(parents=True, exist_ok=True)
    wanted = [(src / "handoff.txt", dst / "handoff.md")]
    try:
        wanted += [(p, dst / "windows" / p.name) for p in sorted((src / "windows").glob("*.md"))]
    except OSError:
        denied += 1
    for s, d in wanted:
        try:
            _atomic_write(d, s.read_text(encoding="utf-8"))
            got += 1
        except OSError:
            denied += 1
    return {"handoff_copied": got, "handoff_unreadable": denied, "handoff_dir": True}


README = """# Engram 事实仓库（Obsidian）

打开方式：Obsidian →「打开本地仓库」→ 选这个文件夹。（文件夹在服务器上，拷到你电脑上再开；不走第三方同步。）

## 怎么用
- `事实/` 里一行一条事实。**勾上方框 `[x]` ＝ 这条不对 / 过时 / 不要了。**
- 🔒 开头的是已确认或你钉住的，勾不了、也不会被收起。
- 你勾完，让助手跑收起脚本：默认只列出要收起哪些、不动库；确认后才真收。
- 收起不是删除：库里的行原样留着，只是不再被召回；每次收起有批号，可以整批撤回。
- 你勾的格子不会因为重新导出而丢（导出会保持勾着，直到被收起）。
- `交接/` 是上个窗口的交接笔记的单向拷贝，在这里改了不会回写。

## 提示
- `^f123` 是这条事实的编号，别删；编辑内容没有用（内容不可改，改了也不会回库）。
- 想看某条的前后文，在 Obsidian 里搜编号或关键词。
"""


def _atomic_write(path: Path, text: str) -> int:
    """内容没变就不写（不动修改时间——手机端同步靠修改时间判断冲突）。返回 1＝写了，0＝没变。"""
    try:
        if path.read_text(encoding="utf-8") == text:
            return 0
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vault", required=True)
    ap.add_argument("--db", default=str(ROOT / "engram.db"))
    ap.add_argument("--handoff-dir", default=os.environ.get("ENGRAM_HANDOFF_DIR") or str(ROOT / "handoff-ro"))
    args = ap.parse_args(argv)
    os.umask(0o007)
    vault = Path(args.vault)
    vault.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    con = open_ro(args.db)
    try:
        st = export_facts(con, vault, now)
    finally:
        con.close()
    st.update(mirror_handoff(Path(args.handoff_dir), vault))
    _atomic_write(vault / "README.md", README)
    _atomic_write(vault / ".engram-export.json", json.dumps({"exported_at": now, **st}, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
