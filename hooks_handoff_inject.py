#!/usr/bin/env python3
"""SessionStart hook：开新窗口时，把「上个窗口的交接笔记」里最要紧的三段注入上下文（开关 ENGRAM_HANDOFF_INJECT，默认关）。

数据来源：交接目录的**只读镜像**（缺省 <ENGRAM_HOME>/handoff-ro，可用 ENGRAM_HANDOFF_DIR 改）。不改镜像、不写任何东西。
取哪一份：windows/index.md 里最新的一行指向的文件；读不到（没权限、文件没了）依次退回 windows/ 里文件名最新的、
再退回根目录的 handoff.txt。

只注入三段：摘要（话题）、悬着的事、她的状态（或「近况」）。不带原话、不带最后几轮对话。
带明确标签「交接笔记，不是她刚说的话，可能已过时，仅供接续」，总长有上限（ENGRAM_HANDOFF_MAX_CHARS，缺省 1200）。

规矩（和 hooks_engram_recall.py 一致）：
  · 任何异常静默 exit 0，只往 hook.log 记一行；**不 print 调试信息**（stdout 会进对话上下文）
  · 压缩/续接（source=compact）、定时任务会话（CC_SESSION_KEY 含 #cron:）、守夜闸门（CLAUDE_WAKE_GATE）一律不注入
  · 同一个窗口（session_id 等于交接文件里的来源会话）不注入自己
  · 笔记里的最后消息时间超过 ENGRAM_HANDOFF_MAX_AGE_DAYS（缺省 14）天就不注入
  · 注入前过 redact.scrub（密钥打码）
settings.json 片段见 TALK-grok 里的安装说明（本脚本不会自己注册）。
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import sys

if "#cron:" in os.environ.get("CC_SESSION_KEY", "") or os.environ.get("CLAUDE_WAKE_GATE"):
    raise SystemExit(0)

_ROOT = os.environ.get("ENGRAM_HOME") or str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, _ROOT)

MAX_CHARS_DEFAULT = 1200
MAX_AGE_DAYS_DEFAULT = 14
LINE_CAP = 110

# 标题行：「## 摘要」「【摘要】」「话题」「悬着的事」「她的状态」「近况」「一句话概要」……单独成行
_HEAD = re.compile(r"^\s*(?:#{1,6}\s*)?[【\[]?\s*(话题|这个窗口聊过什么|摘要（模型生成）|摘要|悬着的事|近况|她的状态|一句话概要)\s*[】\]]?\s*[:：]?\s*$")
_OTHER = re.compile(r"^\s*(?:#{1,6}\s+\S|【[^】]+】)")          # 不认识的标题（含「最后 N 轮」）：结束当前段
_KIND = {"话题": "summary", "这个窗口聊过什么": "summary", "摘要（模型生成）": None, "摘要": None,
         "悬着的事": "pending", "近况": "state", "她的状态": "state", "一句话概要": "brief"}


def _log(msg: str) -> None:
    try:
        with open(os.path.join(_ROOT, "hook.log"), "a", encoding="utf-8") as fh:
            fh.write(f"handoff_inject: {msg}\n")
    except OSError:
        pass


def candidates(root: pathlib.Path) -> list[pathlib.Path]:
    """按优先级排好的候选文件。"""
    out: list[pathlib.Path] = []
    win = root / "windows"
    try:
        for line in (win / "index.md").read_text(encoding="utf-8").splitlines():
            m = re.search(r"[—-]\s*([\w.\-]+\.md)\s*$", line.strip())
            if m:
                out.append(win / os.path.basename(m.group(1)))      # basename：index 里写什么都跳不出 windows/
                break                                               # index 最新的在最上面
    except OSError:
        pass
    try:
        names = sorted((p for p in win.glob("*.md") if p.name != "index.md"), key=lambda p: p.name, reverse=True)
        out += names[:3]
    except OSError:
        pass
    out.append(root / "handoff.txt")
    return out


def read_first(cands: list[pathlib.Path]) -> tuple[pathlib.Path, str] | None:
    for p in cands:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if text.strip():
            return p, text
    return None


def meta(text: str) -> dict:
    sid = re.search(r"(?:^session:\s*|来源会话(?:\s*id)?[：:]\s*)([0-9a-f-]{8,})", text, re.M)
    last = re.search(r"(?:^last_message:\s*|最后(?:一条消息时间|一条对话)[：:]\s*)([^\n（(]+)", text, re.M)
    return {"session": sid.group(1) if sid else "", "last": (last.group(1).strip() if last else "")}


def parse_sections(text: str) -> dict[str, list[str]]:
    sec: dict[str, list[str]] = {"summary": [], "pending": [], "state": [], "brief": []}
    cur = None
    for raw in text.splitlines():
        m = _HEAD.match(raw)
        if m:
            cur = _KIND[m.group(1)]
            continue
        if _OTHER.match(raw):
            cur = None
            continue
        if cur and raw.strip().startswith(("-", "•", "*")):
            item = raw.strip().lstrip("-•* ").strip()
            if item and item != "无" and not item.startswith("（以上是"):
                sec[cur].append(item)
    return sec


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def parse_date(s: str) -> datetime.date | None:
    m = re.match(r"\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s or "")
    try:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    except ValueError:
        return None


def build(sec: dict[str, list[str]], last: str, max_chars: int) -> str:
    head = f"【交接笔记｜上个窗口{('，最后消息 ' + last) if last else ''}｜这是交接文件里的摘要，不是她刚说的话，可能已过时，仅供接续】"
    pend = [_cut(x, LINE_CAP) for x in sec["pending"]]
    state = [_cut(x, LINE_CAP) for x in sec["state"]]
    summ = [_cut(x, LINE_CAP) for x in sec["summary"]]
    if not (pend or state or summ):
        return ""

    def render() -> str:
        parts = [head]
        if summ:
            parts.append("摘要：\n" + "\n".join("- " + x for x in summ))
        if pend:
            parts.append("悬着的事：\n" + "\n".join("- " + x for x in pend))
        if state:
            parts.append("她的状态：\n" + "\n".join("- " + x for x in state))
        return "\n".join(parts)

    out = render()
    # 超长：先砍摘要（从后往前），再砍状态，最后才砍悬着的事（那是最要紧的）
    for lst in (summ, state, pend):
        while len(out) > max_chars and lst:
            lst.pop()
            out = render()
    return _cut(out, max_chars)


def run(stdin_text: str = "") -> str:
    """返回要注入的文本（空串＝什么都不注入）。"""
    import flags
    if not flags.on("HANDOFF_INJECT"):
        return ""
    try:
        data = json.loads(stdin_text) if stdin_text.strip() else {}
    except ValueError:
        data = {}
    if str(data.get("source", "")) == "compact":
        return ""
    root = pathlib.Path(os.environ.get("ENGRAM_HANDOFF_DIR") or os.path.join(_ROOT, "handoff-ro"))
    got = read_first(candidates(root))
    if not got:
        return ""
    _p, text = got
    m = meta(text)
    sid = str(data.get("session_id", ""))
    if sid and m["session"] and sid == m["session"]:
        return ""
    d = parse_date(m["last"])
    max_age = int(os.environ.get("ENGRAM_HANDOFF_MAX_AGE_DAYS") or MAX_AGE_DAYS_DEFAULT)
    if d and (datetime.date.today() - d).days > max_age:
        return ""
    max_chars = int(os.environ.get("ENGRAM_HANDOFF_MAX_CHARS") or MAX_CHARS_DEFAULT)
    out = build(parse_sections(text), m["last"], max_chars)
    if not out:
        return ""
    import redact
    return redact.scrub(out, "hook")


def main() -> int:
    try:
        out = run(sys.stdin.read() if not sys.stdin.isatty() else "")
        if out:
            sys.stdout.write(out + "\n")
    except Exception as e:                                       # noqa: BLE001  交接出错不该卡住开窗
        _log(f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
