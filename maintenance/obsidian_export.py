#!/usr/bin/env python3
"""把 Engram 的事实导成一个 Obsidian 仓库（只读导出；库一个字不改）。

    python3 maintenance/obsidian_export.py --vault /srv/engram-live/obsidian-vault [--db 库路径] [--handoff-dir 交接只读镜像]

仓库里有：
  README.md           怎么用（给她看的，两种标记都在里面）
  事实/<谁>/<类型>_<id段>.md   按「这条事实记在谁名下（facts.subject）」分文件夹，绝不混在一个文件里：
                      用户/（用户本人）· GPT/ · Claude/ · 助手C/ · 其他/（us 等说不清归谁的）
                      每个文件夹里仍按事实 id 每 500 个一段，同一条永远在同一个文件。
                      每条两行：
                        - [ ] 内容 · 日期 ^f123          ← 方框＝「不对 / 过时 / 不要了」（勾上＝收起）
                        \t- [ ] 📌钉住 ^p123              ← 缩进的方框＝「这条重要，钉住」（勾上＝提升为已确认）
                      🔒 开头的是已确认/已钉住的，没有方框。
  事实/_总览.md        每个文件夹每个文件多少条
  交接/                交接只读镜像里读得到的文件的单向拷贝
  .engram-export.json  上次导出的时间、条数（给脚本自己看）

安全：
  · 打开库用 sqlite3 的 mode=ro URI——**不走 store.connect**
  · 重新导出前先把仓库里**所有**文件（含旧的平铺布局、同步冲突副本）里的勾/钉住读出来，并集后写回新文件——
    换布局、搬文件都不会丢她已经点的标记；还把这批标记另存一份到 obsidian-marks-backup/（不在仓库里）
  · 她也可以直接在方框那行文字最前面打 📌（没法点方框时的备用做法），导出时会换成标准的「钉住」方框并保持勾着
  · 内容没变就不写文件（不动修改时间），减少与手机端同步的冲突
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
PARENT = re.compile(r"^- \[([ xX])\] (.*?)\s*\^f(\d+)\s*$")      # 事实行：方框＝不对/过时/不要了
PINROW = re.compile(r"^[ \t]+- \[([ xX])\] .*\^p(\d+)\s*$")        # 缩进的钉住行
MARK = "engram_export: true"
PIN_EMOJI = "📌"
PIN_LABEL = "📌钉住"
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

# 文件夹 ← facts.subject。库里 subject 就是「这条事实记在谁名下」（防回声：只能是说话人本人或 us）。
# 别的（us 等）→ 其他。
FOLDERS = {persona.USER: "用户", "gpt": "GPT", "claude": "Claude", persona.AGENT_C: "助手C"}
OTHER = "其他"
ORDER = ["用户", "GPT", "Claude", "助手C", OTHER]


def folder_of(subject: str) -> str:
    return FOLDERS.get(subject, OTHER)


def open_ro(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _safe(s: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff\-]+", "-", s).strip("-") or "x"


def one_line(s: str) -> str:
    s = re.sub(r"\s*\n\s*", " ⏎ ", (s or "").strip())
    return re.sub(r"^[-*•]\s+", "", s)


def read_marks(vault: Path) -> tuple[set[int], set[int]]:
    """读出仓库 事实/ 下**所有** .md（平铺旧布局、子文件夹、同步冲突副本）里的标记，取并集。
    返回 (勾了「不对」的 id 集合, 钉住的 id 集合)。钉住＝缩进的钉住方框勾上，或事实行文字最前面打了 📌。"""
    ticks: set[int] = set()
    pins: set[int] = set()
    d = vault / "事实"
    if not d.is_dir():
        return ticks, pins
    for p in sorted(d.rglob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            m = PARENT.match(line)
            if m:
                fid = int(m.group(3))
                if m.group(1) in "xX":
                    ticks.add(fid)
                if m.group(2).lstrip().startswith(PIN_EMOJI):
                    pins.add(fid)
                continue
            m = PINROW.match(line)
            if m and m.group(1) in "xX":
                pins.add(int(m.group(2)))
    return ticks, pins


def ticked_ids(vault: Path) -> set[int]:           # 向后兼容的旧名字
    return read_marks(vault)[0]


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


def render_line(r: sqlite3.Row, stale: dict[int, int], keep_ticked: set[int], keep_pinned: set[int] | None = None) -> str:
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
    pbox = "x" if r["id"] in (keep_pinned or ()) else " "
    return f"- [{box}] {body} ^f{r['id']}\n\t- [{pbox}] {PIN_LABEL} ^p{r['id']}"


def _backup_marks(ticks: set[int], pins: set[int], bdir: Path | None, now: str) -> None:
    """标记另存一份（不在仓库里）。和上一份一样就不重复写。"""
    if bdir is None or not (ticks or pins):
        return
    snap = {"ticks": sorted(ticks), "pins": sorted(pins)}
    try:
        bdir.mkdir(parents=True, exist_ok=True)
        last = bdir / "latest.json"
        if last.exists() and json.loads(last.read_text(encoding="utf-8")) == snap:
            return
        stamp = re.sub(r"[^0-9]", "", now)[:12]
        _atomic_write(bdir / f"marks-{stamp}.json", json.dumps({"at": now, **snap}, ensure_ascii=False))
        _atomic_write(last, json.dumps(snap, ensure_ascii=False))
    except OSError:
        pass


def export_facts(con, vault: Path, now: str, marks_backup: Path | None = None) -> dict:
    rows, stale = load_facts(con)
    keep, pinned = read_marks(vault)               # 先收集再重写：换布局也不丢
    _backup_marks(keep, pinned, marks_backup, now)
    fdir = vault / "事实"
    fdir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        groups.setdefault((folder_of(r["subject"]), r["fact_type"]), []).append(r)
    made: dict[str, int] = {}                      # "文件夹/文件名" → 条数
    per_folder: dict[str, int] = {}
    subjects_in_other: dict[str, int] = {}
    wrote = 0
    for (fold, typ), rs in sorted(groups.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        buckets: dict[int, list[sqlite3.Row]] = {}
        for r in rs:
            buckets.setdefault(r["id"] // BUCKET, []).append(r)
            if fold == OTHER:
                subjects_in_other[r["subject"]] = subjects_in_other.get(r["subject"], 0) + 1
        for b, part in sorted(buckets.items()):
            rel = f"{fold}/{typ}_{b:03d}.md"
            head = [
                "---", MARK, f"folder: {fold}", f"fact_type: {typ}",
                f"id_range: {b * BUCKET}-{b * BUCKET + BUCKET - 1}", f"count: {len(part)}", "---", "",
                f"# {fold} · {typ} · id {b * BUCKET}–{b * BUCKET + BUCKET - 1}",
                "", f"> 事实行的方框 ＝ 不对 / 过时 / 不要了；下面缩进的「{PIN_LABEL}」方框 ＝ 这条重要、钉住。🔒 是已确认或已钉住的。勾完告诉助手「收一下」。", "",
            ]
            lines = [render_line(r, stale, keep, pinned) for r in part]
            wrote += _atomic_write(fdir / rel, "\n".join(head + lines) + "\n")
            made[rel] = len(part)
            per_folder[fold] = per_folder.get(fold, 0) + len(part)
    # 清掉自己上次生成、这次不再生成的文件（含旧的平铺布局、同步冲突副本；标记上面已经读走了）
    for p in sorted(fdir.rglob("*.md")):
        rel = p.relative_to(fdir).as_posix()
        if rel in made or rel == "_总览.md":
            continue
        try:
            if MARK in p.read_text(encoding="utf-8")[:300]:
                p.unlink()
        except OSError:
            pass
    for sub in sorted((x for x in fdir.rglob("*") if x.is_dir()), reverse=True):
        try:
            sub.rmdir()                            # 只删空的
        except OSError:
            pass
    ov = ["---", MARK, "---", "", "# 事实总览", "",
          f"共 {len(rows)} 条（候选 + 已确认，不含已被取代/已收起的）。", ""]
    for fold in ORDER:
        if fold not in per_folder:
            continue
        ov += [f"## {fold} — {per_folder[fold]} 条", ""]
        ov += [f"- [[{name[:-3]}]] — {n} 条" for name, n in made.items() if name.startswith(fold + "/")]
        ov.append("")
    wrote += _atomic_write(fdir / "_总览.md", "\n".join(ov) + "\n")
    ids_now = {r["id"] for r in rows if not (r["status"] == "confirmed" or r["user_pinned"])}
    after_t, after_p = read_marks(vault)
    lost = len(((keep | pinned) & ids_now) - (after_t | after_p))
    return {"facts": len(rows), "files": len(made), "files_written": wrote,
            "per_folder": per_folder, "other_subjects": subjects_in_other,
            "kept_ticks": len(keep & ids_now), "kept_pins": len(pinned & ids_now), "marks_lost": lost}


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

打开方式：Obsidian →「打开本地仓库」→ 选这个文件夹（手机上是同步下来的那份）。

## 文件夹
`事实/` 里按「这条记在谁名下」分开，绝不混在一个文件里：
- `用户/` ＝ 你自己的事
- `GPT/` ＝ GPT 名下的
- `Claude/` ＝ Claude 名下的
- `助手C/` ＝ 第三个助手名下的
- `其他/` ＝ 说不清归谁的（比如「我们俩」的事），很少

## 每条事实有两个框，手机上点一下就行
```
- [ ] 这是一条事实 · 2026-10-01 ^f123
	- [ ] 📌钉住 ^p123
```
1. **事实那一行前面的方框** ＝「这条不对 / 过时 / 不要了」。点一下变 ✅，之后会被**收起**（不是删除）。
2. **下面缩进的「📌钉住」方框** ＝「这条重要，帮我钉住」。点一下变 ✅，之后会变成 **🔒 已确认**。
3. 🔒 开头的，已经是确认/钉住的了，不用再点。

## 点完以后
- 告诉助手一句「收一下」。助手会先**只列清单给你看**，你同意了才真的动库。
- **收起** ＝ 这条不再被想起，但库里原样留着，可以整批撤回。
- **钉住** ＝ 变成已确认，不会被自动整理掉，并记一条「你在 Obsidian 里钉住的」证据；也可以整批撤回。
- 你点的框不会因为重新导出而丢，直到被处理掉。
- 两个框**都点了**（又说不对又说钉住）＝ 自相矛盾，两个都不执行，等你改。

## 小提示
- 手机上点不到方框时，也可以在事实那行文字**最前面打一个 📌**，效果和点「钉住」一样。
- `^f123`、`^p123` 是编号，别删。编辑内容没有用（内容不可改，改了也不会回库）。
- `交接/` 是上个窗口的交接笔记的单向拷贝，在这里改了不会回写。
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
    ap.add_argument("--marks-backup-dir", default=str(ROOT / "obsidian-marks-backup"),
                    help="导出前把仓库里已有的勾/钉住另存一份到这里（不在仓库里）")
    args = ap.parse_args(argv)
    os.umask(0o007)
    vault = Path(args.vault)
    vault.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    con = open_ro(args.db)
    try:
        st = export_facts(con, vault, now, Path(args.marks_backup_dir))
    finally:
        con.close()
    st.update(mirror_handoff(Path(args.handoff_dir), vault))
    _atomic_write(vault / "README.md", README)
    _atomic_write(vault / ".engram-export.json", json.dumps({"exported_at": now, **st}, ensure_ascii=False, indent=1))
    print(json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
