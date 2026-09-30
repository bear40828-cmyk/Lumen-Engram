#!/usr/bin/env python3
"""UserPromptSubmit hook: recall before replying.

每轮用户消息进来、模型开口之前，按这句话去记忆库检索，把命中的少量记录
带状态标注注入上下文；检索不到时明确注入「未检索到」，让模型不去编造。

规则：
  · 寒暄、撒娇等没有检索意图的输入不查（交给路由层判断）
  · 只读不写：不调 touch()，避免误召回反过来给错误记录加权
  · 输出有字数上限，每轮都进上下文，不能占太多
  · 同一条记录在冷却窗口内不重复注入
  · 任何异常静默 exit 0，召回失败不能卡住对话
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

ENGRAM = os.environ.get("ENGRAM_HOME", str(pathlib.Path(__file__).resolve().parent))
LOG = os.path.join(ENGRAM, "hook.log")
MAX_CHARS = 420
MAX_HITS = 2
PER_HIT = 140
MAX_AUTO_INPUT = 700

STATUS_MARK = {
    "candidate": "待核",
    "confirmed": "确认",
    "uncertain": "存疑",
    "superseded": "已被取代",
    "invalid": "已作废",
}

RE_CHANNEL = re.compile(r"<channel[^>]*>(?P<body>.*?)</channel>", re.S)
RE_TELEGRAM = re.compile(
    r"\[TELEGRAM_EXTERNAL_MESSAGE\].*?original_body=(?P<q>[\"'])(?P<body>.*?)"
    r"(?P=q).*?\[/TELEGRAM_EXTERNAL_MESSAGE\]", re.S)
SYNTHETIC = re.compile(
    r"^\s*(?:<task-notification>|<system-reminder>|\[tool|\[document|\(document:)", re.I)
EXPLICIT_RECALL = re.compile(
    r"(上次|上回|之前|以前|当时|那天|昨天|前天|还记得|"
    r"记不记得|你说过|我说过|我跟你说过|原话|查一下|翻一下|翻翻)")

REINJECT_COOLDOWN_MIN = 30


def recently_injected(con, minutes: int = REINJECT_COOLDOWN_MIN) -> set[str]:
    """窗口期内真的注入过、并且到现在还有效的那些记录。

    数据来自遥测表——这是今天刚把钩子接上去的那套，记的是「实际注入」
    而不是「查到了」，所以这里能拿到的是真账。
    失效、被取代、我已表态不认的，不算数：那些本来就该重新审视。
    """
    try:
        rows = con.execute(
            """SELECT DISTINCT f.ext_id
                 FROM recall_hits h
                 JOIN recall_runs r ON r.id = h.run_id
                 JOIN facts f ON f.id = h.fact_id
                WHERE h.selected_for_context = 1
                  AND r.router_decision = 'recall'
                  AND datetime(r.created_at) >= datetime('now', ?)
                  AND f.status IN ('candidate','confirmed')
                  AND f.valid_to IS NULL""",
            (f"-{int(minutes)} minutes",))
        return {row[0] for row in rows}
    except Exception as e:
        log(f"去重查询失败（按不去重处理）：{e}")
        return set()


def log(msg: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {msg}\n")
    except Exception:
        pass


def incoming(data: dict) -> str:
    """取这一轮用户真正说的话。通道消息包在 <channel> 里，取里面的正文。"""
    text = data.get("prompt") or ""
    bodies = [m.group("body").strip() for m in RE_CHANNEL.finditer(text)]
    if bodies:
        return bodies[-1]
    telegram = list(RE_TELEGRAM.finditer(text))
    return (telegram[-1].group("body") if telegram else text).strip()


def skip_input(text: str) -> str | None:
    if SYNTHETIC.search(text):
        return "系统／工具消息"
    if len(text) > MAX_AUTO_INPUT and not EXPLICIT_RECALL.search(text):
        return "长篇粘贴且未要求回忆"
    return None


def _alt(var: str) -> str:
    terms = [re.escape(t.strip()) for t in os.environ.get(var, "").split(",") if t.strip()]
    return ("|".join(terms) + "|") if terms else ""


ENTITY_ALT = _alt("ENGRAM_ENTITY_TERMS")
TOPIC_ALT = _alt("ENGRAM_TOPIC_TERMS")

DISTINCTIVE = re.compile(
    r"("
    r"[0-9]{1,2}\s*月\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}|"
    + ENTITY_ALT +
    r"[A-Za-z][A-Za-z0-9_.\-]{3,}|"
    + TOPIC_ALT +
    r"记忆库|交接|日记|看板|唤醒|额度|封号"
    r")"
)

STRONG = re.compile(
    r"[0-9]{1,2}\s*月\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}"
    + ("|" + ENTITY_ALT.rstrip("|") if ENTITY_ALT else ""))

TOO_COMMON = {"claude", "code", "http", "https", "json", "text", "post", "photo", "document"}


QUESTION = re.compile(r"[？?]|吗|来着|到底|是不是|谁|哪个|哪里|什么时候|第一次|最开始|怎么回事")


def pinned_candidates(con, text: str, k: int = 3) -> list[dict]:
    try:
        rows = con.execute(
            "SELECT id, ext_id, subject, content, status, user_pinned FROM facts"
            " WHERE user_pinned=1 AND status='confirmed' AND valid_to IS NULL").fetchall()
    except Exception:
        return []
    q = {text[i:i + 2] for i in range(len(text) - 1)} - {"其实", "我以", "以为"}
    scored = []
    for row in rows:
        c = row["content"] or ""
        overlap = sum(1 for g in q if g in c)
        if overlap >= 2:
            scored.append((overlap, dict(row)))
    scored.sort(key=lambda x: -x[0])
    return [d | {"fact_id": d["id"]} for _, d in scored[:k]]


def distinctive_terms(text: str) -> list[str]:
    found = []
    for m in DISTINCTIVE.finditer(text):
        w = m.group(1)
        if w.lower() in TOO_COMMON:
            continue
        if w not in found:
            found.append(w)
    return found


def main() -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0

    text = incoming(data)
    if not text or len(text) > 2000:
        return 0
    skipped = skip_input(text)
    if skipped:
        log(f"不查 | {text[:40]} | {skipped}")
        return 0

    marks = distinctive_terms(text)
    if not marks and not EXPLICIT_RECALL.search(text) and not QUESTION.search(text):
        log(f"不查 | {text[:40]} | 无指向性词")
        return 0

    sys.path.insert(0, ENGRAM)
    try:
        from retrieve import log_recall, retrieve
        from store.store import connect
    except Exception as e:
        log(f"import 失败：{e}")
        return 0

    def tel(r, decision, reason, selected=None):
        try:
            log_recall(con, text, r, decision=decision, reason=reason,
                       selected=selected)
        except Exception as e:
            log(f"遥测失败：{e}")

    try:
        con = connect()
        r = retrieve(con, text, who=None)
    except Exception as e:
        log(f"召回失败：{e}")
        return 0

    pinned = pinned_candidates(con, text)
    if not r.should_recall and (not pinned or "软话" in (r.reason or "")):
        tel(r, "skip", f"router:{r.reason}")
        log(f"不查 | {text[:40]} | {r.reason}")
        return 0

    seen_ids = {p["ext_id"] for p in pinned}
    r.hits = pinned + [h for h in (r.hits if r.should_recall else []) if h.get("ext_id") not in seen_ids]
    if not r.hits:
        if EXPLICIT_RECALL.search(text):
            print("【记忆库】已查，没有可靠记录；别猜，直说想不起来或问用户。")
        tel(r, "skip", "no_hits")
        log(f"查了没有 | {text[:40]} | 词{r.terms}")
        return 0

    strong = [m for m in marks if STRONG.fullmatch(m)]
    def relevant(h):
        body = (h.get("content") or "").lower()
        hit = [m for m in marks if m.lower() in body]
        return any(m in strong for m in hit) or len(hit) >= 2
    how = "关键词闸"
    try:
        from rerank import judge
        j = judge(text, [h.get("content") or "" for h in r.hits])
    except Exception:
        j = None
    if j is not None:
        how = "模型精排"
        if not j["need"]:
            tel(r, "skip", f"rerank_no_need:{len(r.hits)}")
            log(f"模型判不用翻旧账 | {text[:40]} | {len(r.hits)}条")
            return 0
        scored = sorted(zip(j["scores"], range(len(r.hits))), reverse=True)
        on_topic = [r.hits[i] for sc, i in scored if sc >= 7]
    else:
        on_topic = [h for h in r.hits if relevant(h)] if marks else []
    if not on_topic:
        if EXPLICIT_RECALL.search(text):
            print("【记忆库】已查，没有跟这句相关的记录；别猜，直说想不起来或问用户。")
        tel(r, "skip", f"off_topic:{len(r.hits)}")
        log(f"查到但不相关，不注（{how}） | {text[:40]} | 指向词{marks} | {len(r.hits)}条")
        return 0

    fresh = recently_injected(con)
    hits = [h for h in on_topic if h.get("ext_id") not in fresh]
    if not hits:
        tel(r, "skip", f"recently_injected:{len(r.hits)}")
        log(f"刚注过，不重复 | {text[:40]} | 词{r.terms} | {len(r.hits)}条")
        return 0

    lines = [f"【记忆库·{r.task_type}】线索，非结论；当规则执行前先核对来源与适用范围。"]
    if r.conflicts:
        lines.append("⚠️ 有打架的记录：" + "；".join(r.conflicts)
                     + "。按时间新的那条为准，并且要说清有两条。")
    used = len(lines[0])
    injected = set()
    for h in hits[:MAX_HITS]:
        mark = STATUS_MARK.get(h.get("status") or "", h.get("status") or "?")
        if h.get("user_pinned"):
            mark += "·钉"
        if h.get("my_stance") == "disown":
            mark += "·我已不认"
        sid = (h.get("ext_id") or "")[5:13]
        piece = f"· [{h['subject']}·{mark} {sid}] {(h['content'] or '')[:PER_HIT]}"
        if used + len(piece) > MAX_CHARS:
            break
        lines.append(piece)
        used += len(piece)
        injected.add(h["ext_id"])
    lines.append("（库内记录；冲突时核对原文。）")
    print("\n".join(lines)[:MAX_CHARS])
    tel(r, "recall", f"injected:{len(injected)}/{len(r.hits)}", selected=injected)
    log(f"召回 {len(r.hits)} 条、注入 {len(injected)} 条（{how}）"
        f"（跳过刚注过的 {len(r.hits) - len(hits)} 条）"
        f" | {text[:40]} | 词{r.terms} | 层{r.layers_read}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
