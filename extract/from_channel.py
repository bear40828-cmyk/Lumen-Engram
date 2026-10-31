#!/usr/bin/env python3
"""Lumen Engram · 直接从会话记录抽（补齐摘要抽不到的那一半）

摘要那条路只抽得到「被引号引出来的原话」，用户大量没被引用的发言全漏了。
这条路绕过摘要，直接扫 jsonl：

  她的话  ← role=user 记录里的 <channel source="...telegram..."> 块内文
  我的话  ← role=assistant 记录里 reply 工具的 text 入参

**两条硬规矩：**
  一、只有 role=user 里的 channel 块才算真实入站消息；assistant 侧的 channel 块
      是伪造的（2026-08-16、08-17 各出过一次），只会被记成「我的输出」。
  二、群里不只有她——按 user_id / user 认人，认不出的标 unknown，绝不默认算她。

2026-10-28 起（助手审定）：**助手自己的回复不再直接 promote 成 fact。**
整段回复原文照旧进 evidence（append-only、可回证），但不写 facts——否则「我：…」
这种回复流水账会被当成关于我的事实（当时清库发现 4288 条）。我这一侧的事交给
extract/from_evidence.py 的 LLM 蒸馏去抽「事」。要恢复旧行为加 --promote-claude。

用法：
    python3 extract/from_channel.py [jsonl...] [--dry-run] [--since 2026-08-01] [--promote-claude]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from classify import classify                                   # noqa: E402
from gate.gate import judge                                     # noqa: E402
from keys import match as match_key                             # noqa: E402
import unified                                                   # noqa: E402
from store.store import (add_evidence, connect, promote, record_class, record_key,  # noqa: E402
                         record_tier, stats)

WHO = {persona.USER: persona.N_USER, "claude": "我", "gpt": persona.N_GPT, persona.AGENT_C: persona.N_C}

CN = dt.timezone(dt.timedelta(hours=8))
PROJ = Path(persona.get("ENGRAM_TRANSCRIPT_DIR", str(Path.home() / ".claude/projects")))

# <channel ... user="..." user_id="..." ts="..."> 正文 </channel>
RE_CHANNEL = re.compile(
    r'<channel\s+(?P<attrs>[^>]*?source="[^"]*telegram[^"]*"[^>]*?)>(?P<body>.*?)</channel>',
    re.S)
# 不限来源的 channel 块：认不出的来源也不能掉进「普通 user 文本 = 她」（2026-10-28 补）
RE_CHANNEL_ANY = re.compile(r'<channel\s+(?P<attrs>[^>]*)>(?P<body>.*?)(?:</channel>|\Z)', re.S)
RE_ATTR = re.compile(r'(\w+)="([^"]*)"')
RE_TAGGY = re.compile(r"^\s*<[^>]+>\s*$")

# 群里不只有她。按 user_id / user 认人，认不出的宁可标 unknown，不许默认算她。
# 群里除了她还有别的 AI 的 bot。
# 名单统一放在 speakers.py（CHANNEL_BY_ID / CHANNEL_BY_NAME）；这里保留别名给老代码/老测试用。
import speakers                                                  # noqa: E402

BY_ID = speakers.CHANNEL_BY_ID
BY_NAME = speakers.CHANNEL_BY_NAME


def whose(attrs: str) -> str:
    """严格按名单认：user_id / 名字 ∈ 名单才有名字；名单外的 bot、id 和名字对不上的都是 unknown:*（不入库）。"""
    a = dict(RE_ATTR.findall(attrs))
    uid, uname = a.get("user_id", ""), a.get("user", "")
    return speakers.channel_actor(uname, uid) or f"unknown:{uname or uid or '?'}"


def beijing(iso_utc: str) -> tuple[str, str]:
    """(UTC ISO, 北京时间显示串)"""
    t = dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    return t.isoformat(), t.astimezone(CN).strftime("%Y-%m-%d %H:%M")


# cc-connect 时期她的消息就是普通 user 文本；下面这些开头的是系统／服务器塞进来的，不是她。
NOT_HER = ("【", "This session is being continued", "<command", "<local-command",
           "Caveat:", "[Request interrupted", "<system-reminder", "<task-notification")
RE_ATTACH = re.compile(r"\(Images? also saved locally[^)]*\)|\(Files saved locally[^)]*\)"
                       r"|\[Image: [^\]]*\]|\[\[" + re.escape(persona.ATTACH_TAG) + r"-[a-z]+\|[^\]]*\]\]|\[图片或附件\]", re.S)


def plain_user_text(text: str) -> str:
    t = text.strip()
    if not t or t.startswith(NOT_HER):
        return ""
    t = RE_ATTACH.sub("", t)
    t = re.sub(r"\n{2,}", "\n", t).strip()
    return t


def _is_background(path: Path) -> bool:
    """日记／体检／摘要抽查这类 new-per-run 会话：第一条 user 文本就是【后台…】任务。"""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message")
            if rec.get("type") != "user" or not isinstance(msg, dict):
                continue
            c = msg.get("content")
            t = c if isinstance(c, str) else "".join(
                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text") \
                if isinstance(c, list) else ""
            if t.strip():
                return t.lstrip().startswith("【后台")
    return False


def harvest(path: Path):
    """产出 (行号, actor, 说的话, UTC 时间或 None)。"""
    background = _is_background(path)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line_no, raw in enumerate(fh, start=1):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                continue
            role, content = msg.get("role"), msg.get("content")

            if role == "user":
                text = content if isinstance(content, str) else "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text") \
                    if isinstance(content, list) else ""
                found = False
                for m in RE_CHANNEL_ANY.finditer(text or ""):
                    found = True
                    body = m.group("body").strip()
                    if not body or RE_TAGGY.match(body):
                        continue
                    attrs = dict(RE_ATTR.findall(m.group("attrs")))
                    yield line_no, whose(m.group("attrs")), body, attrs.get("ts")
                # 2026-10-17 起走 cc-connect：她的话不再包 <channel>，就是一条普通 user 文本。
                # 之前这里只认 channel 块，结果 10/17 以后她说的话一条都没进库（10/23 查出）。
                # 服务器自己发的任务（唤醒、日记、体检）都以【…】开头，后台隔离会话整份跳过。
                if not found and not rec.get("isMeta") and not rec.get("isSidechain") \
                        and not background:
                    body = plain_user_text(text or "")
                    if body:
                        yield line_no, persona.USER, body, None

            elif role == "assistant" and isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    kind = blk.get("type")

                    # 我发给她的话——都在 reply 工具的入参里，不在 text 块里
                    if kind == "tool_use":
                        if "reply" not in (blk.get("name") or ""):
                            continue
                        inp = blk.get("input")
                        if isinstance(inp, dict) and isinstance(inp.get("text"), str):
                            body = inp["text"].strip()
                            if body:
                                yield line_no, "claude", body, None
                        continue

                    # 我自己写下的东西：总结、结论、给自己定的规矩。
                    #
                    # 我自己写下的总结、结论、给自己定的规矩也要收。
                    #
                    # 早先这一块整个没收，理由是我把「摘要是派生物、不当证据」
                    # 这条规矩套错了地方——那条说的是**模型生成的转述**，
                    # 而我亲手写下的不是转述，就是我说的话。
                    #
                    # 后果是实打实的：用户问起一条我自己定过的规则，答案就写在我自己的
                    # 交接总结里，库里一个字都没有，只能去 grep 原始文件。
                    # 关于我自己的决定，我反而想不起来。
                    #
                    # thinking 块仍然不收：那是想的不是说的，里面大量是推翻掉的推演。
                    if kind == "text":
                        body = (blk.get("text") or "").strip()
                        if len(body) >= 8:
                            yield line_no, "claude", body, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", help="只要这天及以后的（按 channel 时间戳，YYYY-MM-DD）")
    ap.add_argument("--promote-claude", action="store_true",
                    help="恢复旧行为：助手自己的回复也直接晋升成 fact（默认只存证据，走 LLM 蒸馏）")
    args = ap.parse_args()

    files = args.files or sorted(PROJ.glob("*.jsonl"))
    import flags
    raw_only = flags.on("RAW_ONLY")
    con = connect()
    tally: dict[str, int] = {}
    reasons: dict[str, int] = {}

    for path in files:
        for line_no, actor, body, ts in harvest(path):
            when_utc, when_cn = beijing(ts) if ts else (None, "")
            if args.since and when_utc and when_utc[:10] < args.since:
                continue

            who = WHO.get(actor, actor)
            head = f"{who}（{when_cn}）：" if when_cn else f"{who}："
            content = head + body.replace("\n", " ")

            if args.dry_run:
                tally[actor] = tally.get(actor, 0) + 1
                continue

            ev_id = add_evidence(con, source_kind="chat", source_actor=actor,
                                 source_path=str(path), source_locator=f"L{line_no}",
                                 content=body, occurred_at=when_utc,
                                 verification_status="verified")

            record_tier(con, ev_id, "primary", f"{path}:L{line_no}")

            # 任务三 RAW_ONLY：所有说话人一视同仁，只存证据（evidence + tier 上面已写），不进 facts。
            if raw_only:
                tally["_raw_only"] = tally.get("_raw_only", 0) + 1
                continue
            # 助手侧：只留证据，不直接晋升（交给 from_evidence 蒸馏）。
            if actor == "claude" and not args.promote_claude:
                tally["_claude_evidence_only"] = tally.get("_claude_evidence_only", 0) + 1
                continue

            # 防回声审计（task2-08）：subject = actor（证据说话人本人）；unified.decide 拦 LLM 越权 subject，
            # store.promote 按库里证据兜底。助手侧默认只存证据（上面 continue）。
            # gate + classify + keys：默认逐字走原三套规则；ENGRAM_UNIFIED_GATE=1 时一次 LLM 调用，失败退回规则
            v = unified.decide(con, subject=actor, content=content,
                               evidence_rows=[{"verification_status": "verified"}])
            if not v.promote:
                tally["_gate_no"] = tally.get("_gate_no", 0) + 1
                reasons[v.reason] = reasons.get(v.reason, 0) + 1
                continue

            fact_id = promote(con, subject=v.subject, fact_type=v.fact_type, content=content,
                              evidence_ids=[ev_id], valid_from=when_utc,
                              importance=v.importance, confidence=v.confidence)
            if fact_id is not None:
                record_class(con, fact_id, v.fact_type, v.type_conf, v.type_why)
                if v.fact_key:
                    record_key(con, fact_id, v.fact_key, v.value, v.polarity, v.eligible)
            key = "_dup" if fact_id is None else actor
            tally[key] = tally.get(key, 0) + 1

    if not args.dry_run:
        con.commit()

    print("入库：")
    for k in sorted(tally, key=lambda k: -tally[k]):
        if k.startswith("_"):
            continue
        print(f"  {WHO.get(k, k):<10} {tally[k]:>6} 条")
    print(f"  门控拒 {tally.get('_gate_no', 0)}，重复 {tally.get('_dup', 0)}，"
          f"{persona.N_CLAUDE}侧只存证据 {tally.get('_claude_evidence_only', 0)}，RAW_ONLY 只存证据 {tally.get('_raw_only', 0)}")
    if reasons:
        print("门控拒的理由：")
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {r}")
    if not args.dry_run:
        print("库：", stats(con))
    return 0


if __name__ == "__main__":
    sys.exit(main())
