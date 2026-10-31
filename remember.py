#!/usr/bin/env python3
"""Lumen Engram · engram_remember：助手自记（任务三，2026-10-28）

只是补充，不是主路径：每次会话安静后有自动蒸馏兜底，日常的事不用手记。

硬规矩（没有开关能放松它们）：
  · 记成 candidate、subject 永远是 claude（没有 about / subject / status / pinned / confirmed 入口；
    传了也被忽略）。永远不是钉住、不是确认；本模块不 import pin / confirm。
  · 标记**不写进事实正文**：「自记」只存在 fact_origin.origin='self-note' 一列里；
    召回显示时才临时加〔助手自记〕（echo.tag，和〔转述〕同一个渲染层）。FTS / 向量 / 卡片用的都是干净正文。
  · 严格上限：每个会话进程 3 条（PER_SESSION）、每个北京日 5 条（PER_DAY，查库，重启 / 换窗口不清零）。
  · 防回声：自己写一条 tool_result 证据（source_actor=claude），promote 兜底按证据判；quote 命中的原话只当附加出处。
  · 数字校验：正文里有数字 / 日期，必须附 quote 且数字出自 quote（找不到原话的数字不记）。
  · 去重：同 subject 同 sha 返回已有；与近 14 天活着的 claude 事实 Jaccard ≥ 0.65 也提示已有相近的。
kind='correction'（2026-10-28，任务四·步骤1）：她当面纠正我时当场记下，直接进「她纠正过我」那一层
（confirmed + 钉住，召回时排在最前面，和 corrections.py promote 写的是同一种事实）。硬规矩：
  · 内容只能出自**她的话**：必须附 quote（逐字，4–80 字），而且必须在她本人最近 24 小时说过的话里找得到——
    先查库里已采集的她的原话，再查会话文件（collect 同一套认人规矩，只认她本人，不认我、不认别的 AI / 转发包装）；
    找不到就拒绝。我自己的推断写不进来。
  · right（实际情况）必须跟 quote 有 ≥2 个共同二元组，数字必须出自 quote；wrong 是我自己以为的（说的是我自己）。
  · 证据是她那句原话（chat / 她 / primary），不是 tool_result；不占自记的每会话 3 条、每日 5 条，另限每个北京日 10 条。
  · 同一件事她说五六遍也只记一条（与已有纠正 Jaccard ≥ 0.5 就提示已有）；同时追加一行到 corrections.jsonl（这一层的真源）。
派生层（class / 卡片 / 向量）和别的新事实一样存完再补（derive_repair），reason='self-note-v1'。
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from common import sha256                                          # noqa: E402
from extract.from_evidence import numbers_unsupported              # noqa: E402
from store.store import add_evidence, now, promote, record_tier   # noqa: E402

SUBJECT = "claude"
PER_SESSION = int(persona.get("ENGRAM_REMEMBER_PER_SESSION", "3"))
PER_DAY = int(persona.get("ENGRAM_REMEMBER_PER_DAY", "5"))
MIN_LEN, MAX_LEN = 8, 120
KINDS = ("episodic", "semantic", "preference", "procedural")
CORR_PER_DAY = 10
CORR_TAG = "【她纠正过我·"
FRESH_HOURS = 24
CORRECTIONS_JSONL = ROOT / "corrections.jsonl"
RUN_ID = "mcp_" + uuid.uuid4().hex[:10]            # 这个 MCP 进程 = 一个会话
RE_MARKUP = re.compile(r'(<channel\b|</channel>|message_id=|chat_id=|\[TELEGRAM_EXTERNAL|<task-notification'
                       r'|<system-reminder|plugin:telegram)')
CN = dt.timezone(dt.timedelta(hours=8))

TOOL = {
    "name": "engram_remember",
    "description": persona.fill("记下一条你自己判断「以后一定用得上」的事。记成「待核」候选，标明是@@CLAUDE@@自记。"
                    "⚠️ 这不是钉住：钉住（engram_pin）只属于@@USER@@亲口要求钉的内容；自记的永远是候选，"
                    "不会变成 confirmed / 钉住，你也不能改、不能删它。"
                    "这只是补充，不是主路径：会话安静后有自动蒸馏兜底，日常的事不用手记。"
                    "该记：① @@USER@@刚明确说出、以后会反复用到的偏好/规矩/约定；② 你刚做完、会改变以后做法的决定或改动；"
                    "③ 你答应了她、还没做的事。不要记：安慰、建议、猜测、进度流水、情绪渲染、一次性的小指令；"
                    "亲密内容只能记「某时有过互动」，不记动作和身体细节；已经能从原话里查到的。"
                    "写法：一句话、第三人称、写清谁/什么事/哪天，8–120 字，**不要自己加「自记」二字**（显示时会自动标）。"
                    "正文里有数字或日期的，必须附 quote（从最近对话逐字复制 4–40 字，数字要出自它）。"
                    "上限：每个会话 3 条、每个北京日 5 条，超了会被拒绝。"
                    "【纠正】@@USER@@当面纠正你（「不是X，是Y」「我不在家」）时，马上用 kind=correction 记：content＝实际情况（只能写她说的，别加你的推断），"
                    "wrong＝你原来以为的，topic＝3–8 字主题，quote＝她那句话逐字复制（必须是她本人最近说的，找不到会被拒绝）。"
                    "纠正直接进「她纠正过我」那一层（钉住、召回优先），不占自记名额；同一件事只记一次。"),
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "一句话，8–120 字"},
            "kind": {"type": "string", "enum": list(KINDS) + ["correction"], "description": "默认 episodic；她当面纠正你用 correction"},
            "quote": {"type": "string", "description": "选填，4–40 字，逐字；有数字/日期时必填。correction 必填，4–80 字，必须是她本人最近说的原话"},
            "wrong": {"type": "string", "description": "仅 correction：你原来以为的（一句话）"},
            "topic": {"type": "string", "description": "仅 correction：这件事的主题，3–8 字"},
        },
        "required": ["content"],
    },
}
PIN_EXTRA = persona.fill("只用于@@USER@@明确说要钉住的内容。你自己想留的，用 engram_remember。")


def _day_start_utc(now_dt: dt.datetime) -> str:
    d = now_dt.astimezone(CN).replace(hour=0, minute=0, second=0, microsecond=0)
    return d.astimezone(dt.timezone.utc).isoformat()


def _bigr(s: str) -> set:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def counts(con, run_id: str = RUN_ID, now_dt: dt.datetime | None = None) -> tuple[int, int]:
    now_dt = now_dt or dt.datetime.now(dt.timezone.utc)
    s = con.execute("SELECT COUNT(*) FROM fact_origin WHERE origin='self-note' AND run_id=?", (run_id,)).fetchone()[0]
    d = con.execute("SELECT COUNT(*) FROM fact_origin WHERE origin='self-note' AND created_at>=?",
                    (_day_start_utc(now_dt),)).fetchone()[0]
    return s, d


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _her_message(con, quote: str, now_dt: dt.datetime, transcript_dir=None):
    """她本人最近 FRESH_HOURS 小时里说的、含 quote 的那句。先查库里已采集的，再查会话文件。
    返回 dict(evidence_id|None, body, path, loc, occurred_at) 或 None。别人 / 我 / 转发包装的话一律不算。"""
    q = _norm(quote)
    since = (now_dt - dt.timedelta(hours=FRESH_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    for r in con.execute("SELECT id, content, occurred_at, source_path, source_locator FROM evidence"
                         " WHERE source_kind='chat' AND source_actor=?"
                         " AND substr(COALESCE(occurred_at, captured_at),1,19) >= ? ORDER BY id DESC LIMIT 400",
                         (persona.USER, since)):
        if q in _norm(r["content"]):
            return {"evidence_id": r["id"], "body": r["content"], "path": r["source_path"],
                    "loc": r["source_locator"], "occurred_at": r["occurred_at"]}
    tdir = transcript_dir or persona.get("ENGRAM_TRANSCRIPT_DIR", "")
    if not tdir or not Path(tdir).is_dir():
        return None
    import time
    import speakers
    from extract.from_channel import beijing, harvest
    files = sorted((p for p in Path(tdir).glob("*.jsonl")
                    if time.time() - p.stat().st_mtime <= FRESH_HOURS * 3600),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    for p in files:
        items = list(harvest(p))[-200:]
        for line_no, actor, body, ts in reversed(items):
            actor = speakers.normalize_actor(actor)
            if actor is not None:
                actor, _note = speakers.reconcile(actor, body)
            if actor != persona.USER or q not in _norm(body):
                continue
            when = beijing(ts)[0] if ts else None
            if when and when[:19] < since:
                continue
            return {"evidence_id": None, "body": body, "path": str(p), "loc": f"L{line_no}",
                    "occurred_at": when}
    return None


def correct(con, content, quote, wrong, topic, *, now_dt: dt.datetime | None = None,
            transcript_dir=None, corrections_path=None) -> tuple[bool, str]:
    """她当面纠正我 → 当场记进「她纠正过我」层（confirmed + 钉住）。内容只能出自她的话。"""
    now_dt = now_dt or dt.datetime.now(dt.timezone.utc)
    content, quote = str(content or "").strip(), str(quote or "").strip()
    wrong, topic = str(wrong or "").strip(), str(topic or "").strip()
    if not (8 <= len(content) <= MAX_LEN):
        return False, f"content（实际情况）要 8–{MAX_LEN} 字（现在 {len(content)}）。"
    if not (4 <= len(quote) <= 80):
        return False, "纠正必须附 quote：她那句话逐字复制，4–80 字。"
    if not (4 <= len(wrong) <= 100) or not (2 <= len(topic) <= 12):
        return False, "wrong（你原来以为的，4–100 字）和 topic（主题，2–12 字）都要填。"
    if any(RE_MARKUP.search(x) or "自记" in x for x in (content, wrong, topic, quote)):
        return False, "内容里带通道标记或「自记」，不能记。"
    q = _bigr(quote)
    if len(_bigr(content) & q) < 2:
        return False, "content 跟 quote 对不上（至少要有 2 个共同的词组）——纠正只能写她说的，不能写你的推断。"
    if re.search(r"\d", content) and numbers_unsupported(content, quote):
        return False, "content 里的数字/日期在 quote 里找不到——原话没写的数字不记。"
    today = _day_start_utc(now_dt)[:19]
    n = con.execute("SELECT COUNT(*) FROM facts WHERE user_pinned=1 AND content LIKE ? AND created_at>=?",
                    (CORR_TAG + "%", today)).fetchone()[0]
    if n >= CORR_PER_DAY:
        return False, f"今天已经记了 {CORR_PER_DAY} 条纠正，不再收。"
    her = _her_message(con, quote, now_dt, transcript_dir)
    if her is None:
        return False, ("在她本人最近 24 小时说过的话里找不到这句 quote——纠正只能记她亲口说的；"
                       "quote 要逐字从她那句话里复制（不是你的复述，也不是别的 AI 转发的）。")
    when = her["occurred_at"] or now_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    day = when[:10]
    wrong_t = wrong if wrong.startswith("我") else "我以为" + wrong
    fact_text = f"{CORR_TAG}{day}·{topic}】{wrong_t}；其实{re.sub(r'^(其实|实际上)[，, ]*', '', content)}"
    mine = _bigr(fact_text.split("】", 1)[-1])                 # 比正文，不比「【她纠正过我·日期·」那套模板
    for r in con.execute("SELECT ext_id, content FROM facts WHERE user_pinned=1 AND status='confirmed'"
                         " AND valid_to IS NULL AND content LIKE ?", (CORR_TAG + "%",)):
        t = _bigr(r["content"].split("】", 1)[-1])
        if mine and t and len(mine & t) / len(mine | t) >= 0.5:
            return False, f"已有相近的纠正：{r['ext_id']}（同一件事不用重复记）。"
    ev = her["evidence_id"]
    if ev is None:
        import redact
        rbody, _cnt = redact.redact(her["body"])
        ev = add_evidence(con, source_kind="chat", source_actor=persona.USER, source_path=her["path"],
                          source_locator=her["loc"], content=rbody, occurred_at=her["occurred_at"],
                          verification_status="verified")
        record_tier(con, ev, "primary", f"{her['path']}:{her['loc']}")
    fid = promote(con, subject=persona.USER, fact_type="semantic", content=fact_text, evidence_ids=[ev],
                  status="confirmed", valid_from=when, importance=1.0, confidence=1.0, user_pinned=True)
    if fid is None:
        con.rollback()
        return False, "没记成（重复或被防回声拦下）。"
    con.commit()
    import json
    row = {"ts": when, "source_path": her["path"], "line": her["loc"], "her": her["body"][:600],
           "my_prev": "", "her_prev": "", "wrong": wrong_t, "right": content, "topic": topic,
           "ask": topic, "keep": True, "via": "engram_remember"}
    try:
        with open(corrections_path or CORRECTIONS_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass                                    # 真源文件写不了不影响库里这条（库里的也能导出）
    ext = con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]
    return True, f"已记入「她纠正过我」层（钉住）：{ext}。今天第 {n + 1}/{CORR_PER_DAY} 条纠正。"


def remember(con, content, kind="episodic", quote=None, *, run_id: str = RUN_ID,
             now_dt: dt.datetime | None = None, wrong=None, topic=None, **_ignored) -> tuple[bool, str]:
    """**_ignored：status / pinned / subject / about / ext_id 等多余参数一律忽略。
    kind='correction' 走 correct()（只在她的话里取材，进纠正层）；其余都是自记。"""
    now_dt = now_dt or dt.datetime.now(dt.timezone.utc)
    if kind == "correction":
        return correct(con, content, quote, wrong, topic, now_dt=now_dt,
                       transcript_dir=_ignored.get("transcript_dir"),
                       corrections_path=_ignored.get("corrections_path"))
    content = str(content or "").strip()
    quote = str(quote or "").strip() or None
    kind = kind if kind in KINDS else "episodic"
    if not (MIN_LEN <= len(content) <= MAX_LEN):
        return False, f"内容要 {MIN_LEN}–{MAX_LEN} 字（现在 {len(content)}）。"
    if RE_MARKUP.search(content):
        return False, "内容里带通道标记，不能记。"
    if "自记" in content:
        return False, "别在正文里写「自记」——标记是系统显示时加的。"
    s, d = counts(con, run_id, now_dt)
    if s >= PER_SESSION:
        return False, f"本会话已经记满 {PER_SESSION} 条了，不再收。这只是补充，自动蒸馏会兜底。"
    if d >= PER_DAY:
        return False, f"今天已经记满 {PER_DAY} 条了，不再收。这只是补充，自动蒸馏会兜底。"
    if re.search(r"\d", content):
        if not quote or len(quote) < 4:
            return False, "正文里有数字/日期，必须附 quote（逐字，4–40 字）。"
        if numbers_unsupported(content, quote):
            return False, "正文里的数字/日期在 quote 里找不到——原话没写的数字不记。"
    sha = sha256(content)
    dup = con.execute("SELECT ext_id FROM facts WHERE subject=? AND content_sha256=? AND status IN"
                      " ('candidate','confirmed') AND valid_to IS NULL", (SUBJECT, sha)).fetchone()
    if dup:
        return False, f"库里已有一模一样的：{dup[0]}"
    mine = _bigr(content)
    since = (now_dt - dt.timedelta(days=14)).isoformat()
    for r in con.execute("SELECT ext_id, content FROM facts WHERE subject=? AND valid_to IS NULL AND status IN"
                         " ('candidate','confirmed') AND COALESCE(valid_from, created_at)>=?", (SUBJECT, since)):
        t = _bigr(r["content"])
        if mine and t and len(mine & t) / len(mine | t) >= 0.65:
            return False, f"已有相近的：{r['ext_id']}"
    n = s + 1
    loc = f"remember:{run_id}#{n}"
    ts = now_dt.isoformat()
    ev = add_evidence(con, source_kind="tool_result", source_actor="claude", source_path="mcp:engram_remember",
                      source_locator=loc, content=content, occurred_at=ts, verification_status="unverified")
    record_tier(con, ev, "secondary", f"self:{run_id}#{n}")
    ev_ids = [ev]
    if quote:
        since48 = (now_dt - dt.timedelta(hours=48)).isoformat()
        hit = con.execute("SELECT id FROM evidence WHERE source_kind='chat' AND instr(content, ?)>0"
                          " AND COALESCE(occurred_at, captured_at)>=? ORDER BY id DESC LIMIT 1",
                          (quote, since48)).fetchone()
        if hit:
            ev_ids.append(hit[0])
    fid = promote(con, subject=SUBJECT, fact_type=kind, content=content, evidence_ids=ev_ids,
                  status="candidate", valid_from=ts, confidence=0.5)
    if fid is None:
        con.rollback()
        return False, "没记成（重复或被防回声拦下）。"
    con.execute("INSERT INTO fact_origin (fact_id, origin, run_id, session_key, quote, prompt_version, created_at)"
                " VALUES (?,?,?,?,?,?,?)", (fid, "self-note", run_id, "mcp:engram_remember", quote, "self-note-v1", now_dt.isoformat()))
    con.commit()
    ext = con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]
    return True, (f"已记（待核，{persona.N_CLAUDE}自记）：{ext}。它不是钉住，也不会自动变成确认。"
                  f"今天第 {d + 1}/{PER_DAY} 条，本会话第 {s + 1}/{PER_SESSION} 条。")
