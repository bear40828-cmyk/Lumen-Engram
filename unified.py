#!/usr/bin/env python3
"""Lumen Engram · 一次判定（gate + classify + keys 合成一次 LLM 调用）——任务二·步骤5，2026-10-28

之前一条候选要过三套手写规则：gate（有没有事实槽位）→ classify（四类里哪一类）→ keys.match（占哪个槽位）。
规则各自一堆正则，互相不知道对方怎么判的，专名白名单永远追不上她会聊到的东西。
现在可以让模型一次说完，输出固定 JSON：

    {"promote": bool, "fact_type": "episodic|semantic|preference|procedural",
     "subject": "user|claude|gpt|agent_c|us", "fact_key": "<keys.yaml 里的名字>或 null",
     "slots": ["数值","专名","人物","偏好","变化","规矩","纠错" 中的几个],
     "confidence": 0~1, "quote": "从原话里逐字复制的 4~40 字（promote 为 true 时必须）"}

开关：ENGRAM_UNIFIED_GATE=1（默认关，见 flags.py）。**默认关＝逐字走原来的三套规则，行为不变。**

硬护栏（不管开不开，LLM 都不能越过）：
  · 她钉住的（user_pinned）一律晋升；纯表情／标点一律不晋升；
  · 没有证据 / 证据没核对通过 一律不晋升（evidence 层的规矩）；库里已有一模一样的一律不晋升。
  · fact_key 只能出自 keys.yaml，「不许自己造 key」——不在表里的当 null；
  · quote 必须逐字在原话里，对不上整条输出作废。
LLM 失败（没配 key、超时、返回体不合规、quote 对不上）→ 自动退回规则，并在 reason 里标明。
规则代码（gate/gate.py、classify.py、keys.py）原样保留，是兜底。

用法（三个入库脚本都是这一个调用）：
    u = unified.decide(con, subject=actor, content=content, evidence_rows=[...])
    if not u.promote: ...
    promote(con, subject=u.subject, fact_type=u.fact_type, ..., importance=u.importance, confidence=u.confidence)
    record_class(con, fid, u.fact_type, u.type_conf, u.type_why)
    if u.fact_key: record_key(con, fid, u.fact_key, u.value, u.polarity, u.eligible)
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

import echo                                                   # noqa: E402
import flags                                                  # noqa: E402
import keys as K                                              # noqa: E402
from classify import classify                                 # noqa: E402
from common import sha256                                     # noqa: E402
from gate.gate import NO_TEXT, SLOTS, Verdict, judge          # noqa: E402

SUBJECTS = {*persona.ACTORS, "us"}
TYPES = {"episodic", "semantic", "preference", "procedural"}
SLOT_NAMES = tuple(SLOTS)
MIN_QUOTE = 4
SCHEMA = persona.fill('{"promote": true或false, "fact_type": "episodic|semantic|preference|procedural", '
          '"subject": "@@A_USER@@|claude|gpt|@@A_C@@|us", "fact_key": "上面表里的名字或null", '
          '"slots": ["数值|专名|人物|偏好|变化|规矩|纠错 中的几个"], "confidence": 0到1, '
          '"quote": "从原话逐字复制的4到40字"}')

PROMPT = persona.fill("""你在替一个长期记忆库做「要不要记」的一次性判定。下面是一条候选（说话人 + 原话）。

要记（promote=true）：过一个月还可能想得起来的事——有来龙去脉的小事、她说出口的心意/承诺/在意的话、
她的生活（买了什么、去了哪、身体、在忙什么）、定下的做法/规矩/踩过的坑、稳定事实（身份、关系、经历、数字、日期、专名）。
不记（promote=false）：打招呼、语气词、表情、单纯的「嗯」「好」「晚安」、对眼前小事的一句指令、
AI 自己的安慰/建议/推测（除非被接受成了约定）、网上看到的别人的事。短不是理由——@@EX_SHORT@@都要记。

fact_type：episodic=某时发生的一件具体的事；semantic=稳定事实；preference=喜欢/讨厌/希望怎样（偏好主体必须就是 subject 本人，别人的偏好不算）；
procedural=定下的做法、规矩、以后该怎么办。拿不准一律 episodic。
subject：这件事主要关于谁（@@A_USER@@/claude/gpt/@@A_C@@/us，us=@@USER@@和@@CLAUDE@@两人之间的事）。**只能是说话人本人**（us 需要@@USER@@和对方双方各有原话，这里只有一方，所以不要写 us）——
别人说的话不能记成另一个人的事（比如@@CLAUDE@@说「@@USER@@怎样」不能记成@@USER@@的事）。
fact_key：只能从下面的表里选，选不到就 null，不许自己造：
{keys}
slots：这条里有哪几类事实槽位（可空）。
quote：从原话里**逐字**复制一小段（4～40 字），能证明这条值得记；promote=false 时给空字符串。

说话人：{speaker}
原话：{body}

只输出 JSON：{schema}""")


@dataclass
class Unified:
    promote: bool
    reason: str
    code: str
    subject: str
    fact_type: str = "episodic"
    type_conf: float = 0.5
    type_why: str = ""
    importance: float | None = None
    confidence: float | None = None
    slots: tuple[str, ...] = ()
    fact_key: str | None = None
    value: str | None = None
    polarity: str = "pos"
    eligible: bool = False
    source: str = "rules"            # rules / llm


def _rules(con, subject, content, evidence_rows, user_pinned, note="") -> Unified:
    """原来的三套规则，一字不改地串起来。"""
    v: Verdict = judge(con, subject=subject, content=content, evidence_rows=evidence_rows,
                       user_pinned=user_pinned)
    if not v.promote:
        return Unified(False, v.reason + note, v.code, subject, slots=v.slots)
    kind, kconf, kwhy = classify(content, subject)
    fkey, fval, fpol, fok = K.match(content)
    return Unified(True, v.reason + note, v.code, subject, kind, kconf, kwhy,
                   v.importance, v.confidence, v.slots, fkey, fval, fpol, fok, "rules")


def _keys_table() -> str:
    return "\n".join(f"- {k.name}（{k.cardinality}，{k.value_type}）" for k in K.registry())


def parse_llm(text: str, body: str) -> dict | None:
    """校验模型输出。不合规返回 None（调用方退回规则）。"""
    try:
        m = re.search(r"\{.*\}", text, re.S)
        got = json.loads(m.group(0) if m else text)
    except Exception:                                         # noqa: BLE001
        return None
    if not isinstance(got, dict) or not isinstance(got.get("promote"), bool):
        return None
    if not got["promote"]:
        return {"promote": False}
    typ, subj = str(got.get("fact_type", "")).strip(), str(got.get("subject", "")).strip()
    quote = str(got.get("quote", "")).strip()
    if typ not in TYPES or subj not in SUBJECTS:
        return None
    if len(quote) < min(MIN_QUOTE, len(body)) or quote not in body:   # 逐字核对
        return None
    key = got.get("fact_key")
    key = key if isinstance(key, str) and K.by_name(key) else None    # 不许造 key
    slots = tuple(s for s in (got.get("slots") or []) if s in SLOT_NAMES) \
        if isinstance(got.get("slots"), list) else ()
    try:
        conf = min(max(float(got.get("confidence", 0.6)), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.6
    return {"promote": True, "fact_type": typ, "subject": subj, "fact_key": key,
            "slots": slots, "confidence": conf, "quote": quote}


def _key_value(key_name: str, body: str) -> tuple[str | None, str, bool]:
    """LLM 选了 key 之后，值/极性/可巩固仍用 keys.py 的规则去抠（值要能在原话里对上）。"""
    spec = K.by_name(key_name)
    value, pol, eligible = None, "pos", True
    if spec and spec.value_pattern and (m := spec.value_pattern.search(body)):
        value = m.group(0)
        pol = K._polarity(body, m.start())
        if spec.requires_self and not K._is_self(body, m.start()):
            eligible = False
        if spec.digit_boundary and value.isdigit():
            before = body[m.start() - 1] if m.start() > 0 else ""
            after = body[m.end()] if m.end() < len(body) else ""
            if before.isdigit() or after.isdigit():
                eligible = False
    elif spec and spec.value_pattern:
        eligible = False                                     # 值抠不出来，不拿去自动巩固
    return value, pol, eligible


def decide(con, *, subject: str, content: str, evidence_rows: list[dict],
           user_pinned: bool = False) -> Unified:
    if flags.on("RAW_ONLY") and not user_pinned:
        # 任务三：只存原话。咽喉口——任何调用方（含将来新写的）都不可能再逐条晋升；事实由 distill_slice 整理。
        return Unified(False, "raw-only：只存证据，不逐条判断", "raw_only", subject)
    if not flags.on("UNIFIED_GATE"):
        return _rules(con, subject, content, evidence_rows, user_pinned)

    body = content.split("：", 1)[-1].strip()
    # ── 硬护栏：这几条不让 LLM 参与
    if user_pinned:
        return _rules(con, subject, content, evidence_rows, user_pinned)
    if NO_TEXT.match(body):
        return Unified(False, "只有表情或标点", "no_text", subject)
    if not evidence_rows:
        return Unified(False, "没有可核对的证据，留在证据层", "no_evidence", subject)
    verified = [e for e in evidence_rows if e.get("verification_status") == "verified"]
    if not verified:
        return _rules(con, subject, content, evidence_rows, user_pinned)   # 给出同样的拒绝理由
    if con.execute("SELECT 1 FROM facts WHERE subject=? AND content_sha256=?",
                   (subject, sha256(content))).fetchone():
        return Unified(False, "库里已有一模一样的", "duplicate", subject)

    # ── 一次 LLM 调用
    try:
        from llm import chat
        text = chat([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": PROMPT.format(keys=_keys_table(), speaker=subject,
                                                       body=body[:800], schema=SCHEMA)},
        ], purpose="extract", max_tokens=300, timeout=30)
    except Exception as e:                                    # noqa: BLE001
        return _rules(con, subject, content, evidence_rows, user_pinned,
                      note=f"（LLM 不可用，退回规则：{type(e).__name__}）")
    got = parse_llm(text, body)
    if got is None:
        return _rules(con, subject, content, evidence_rows, user_pinned,
                      note="（LLM 输出不合规或 quote 对不上，退回规则）")
    if not got["promote"]:
        return Unified(False, "LLM 判不用记", "llm_no", subject, source="llm")
    if got["subject"] == "us":
        actors = {e.get("source_actor") for e in evidence_rows if e.get("source_actor")} or {subject}
        if not echo.us_ok(actors):
            # task2-09：us 要双方原话；单方证据（尤其只有助手）产不出 us → 不信这次输出，退回规则（subject=说话人）
            return _rules(con, subject, content, evidence_rows, user_pinned,
                          note="（防回声：LLM 给了 us 但只有一方的原话，退回规则）")
    elif not echo.owner_ok(subject, got["subject"]):
        # 防回声：说话人是 subject，LLM 却把它记到别人名下 → 不信这次输出，退回规则（规则里 subject 就是说话人本人）
        return _rules(con, subject, content, evidence_rows, user_pinned,
                      note=f"（防回声：LLM 想把 {subject} 的话记成 {got['subject']} 的事，退回规则）")

    slots = got["slots"]
    importance = round(min(0.95, 0.3 + 0.12 * len(slots) + (0.1 if len(body) >= 30 else 0.0)), 2)
    confidence = round(min(1.0, 0.6 + 0.1 * len(verified)) * 0.5 + got["confidence"] * 0.5, 2)
    fkey = got["fact_key"]
    value, pol, eligible = _key_value(fkey, body) if fkey else (None, "pos", False)
    return Unified(True, "LLM：" + (("＋".join(slots)) or "无槽位"), "llm_yes", got["subject"],
                   got["fact_type"], got["confidence"], "unified-llm-v1", importance, confidence,
                   tuple(slots), fkey, value, pol, eligible, "llm")
