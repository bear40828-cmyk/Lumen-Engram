#!/usr/bin/env python3
"""Lumen Engram · 防回声（task2-08，2026-10-28）——入库和召回两处共用的一套规矩

回声 = 助手（或别的 AI）说的话，被记成「用户自己的事实」，召回两三次后长成长期事实，
或被我误当成她说过的话。这是最重要的一条护栏，**入库和召回两处都要有**，所以放在一个模块里：

入库（硬规矩，默认开，没有开关）
    证据 source_actor 是 X，从它产出的事实 subject 只能是 X 或 us。
    **task2-09 收紧 us**：us 只有在支持证据里**同时有她（user）和另一方**各自的原话时才成立；
    只有助手（或只有她）一方的证据产不出 us——它要么落回 claude（内容确实是助手自己的事），要么丢弃。
    · owner_ok(actor, subject)         单点判断（抽取器 check / unified / distill_self 用）
    · evidence_allows(con, subject, evidence_ids)
                                       按库里实际挂的证据判（store.promote 兜底；任何路径都绕不过）
    例外：确认过 / 钉住的行（她自己的纠正、她自己钉的）不拦。

召回（降权 + 标注 + 不强化）
    owner-support = 这条事实挂的 support 证据里，有没有一条是主体本人说的。
    主体是人（user / gpt / agent_c）、挂着证据、却一条本人的都没有 = 「只有转述」(echo-only)：
    · 排序 × ECHO_WEIGHT（1/3，与 consolidate.ECHO_WEIGHT 同一个数）
    · 注入时标〔转述·来自助手〕，让模型不把它当成本人的话
    · store.reinforce 不给它加权（不让转述被召回几次就越排越前）
    confirmed / 钉住的豁免；没挂证据的无从判断，不降权。
"""
from __future__ import annotations

import re
import sys

import persona

ECHO_WEIGHT = 1 / 3                      # 与 consolidate.ECHO_WEIGHT 一致（测试里固定）
INGEST_PERSONS = frozenset({persona.USER, "claude", "gpt", persona.AGENT_C})   # 受「X 只产 X/us」约束的 subject
# 召回侧判「转述」的主体：人类/外部 AI + us（task2-09）+ claude（task2-09：她说的话被记在助手名下的那类）
RECALL_PERSONS = frozenset({persona.USER, "gpt", persona.AGENT_C, "us", "claude"})
US = "us"
WHO_CN = {persona.USER: persona.N_USER, "claude": persona.N_CLAUDE, "gpt": persona.N_GPT, persona.AGENT_C: persona.N_C, "system": "系统"}

BLOCKED: list[tuple[str, str]] = []      # 本进程里被拦下的 (subject, 证据说话人)，只记聚合信息，不记内容


def us_ok(actors) -> bool:
    """us 的成立条件：支持证据里同时有她（user）和另一方的原话。只有一方 = 回声，不是共同的事。"""
    a = {x for x in (actors or ()) if x}
    return persona.USER in a and len(a) >= 2


# 单方证据落回去的主体：内容必须以该说话人为主语（助手…/用户…），否则丢弃
_LEAD = {"claude": (persona.N_CLAUDE, *persona.CLAUDE_ALIASES),
         persona.USER: (persona.N_USER, *persona.USER_ALIASES),
         "gpt": (persona.N_GPT, *persona.GPT_ALIASES), persona.AGENT_C: (persona.N_C,)}


def resolve_us(actor, content) -> str | None:
    """us 没有双方证据时：内容以说话人本人为主语 → 落回该说话人；否则 None（丢弃）。"""
    leads = _LEAD.get(actor or "")
    if leads and str(content or "").lstrip().startswith(leads):
        return actor
    return None


def owner_ok(actor, subject) -> bool:
    """证据说话人 actor，能不能产出这个 subject 的事实。actor 未知（None）时不判，交给 promote 兜底。
    us：单个说话人的证据证明不了「两个人的事」→ 一律 False（us 要走 us_ok，用双方证据）。"""
    if not actor:
        return True
    if subject == US:
        return False
    if subject not in INGEST_PERSONS:
        return True
    return subject == actor


def _fixes(con, ids) -> dict:
    try:
        import actor_fix
        return actor_fix.lookup(con, ids)
    except Exception:                                        # noqa: BLE001  修正层出错＝当没有，不影响入库/召回
        return {}


def _actors_of(fix, original):
    if not fix:
        return {original} if original else set()
    import actor_fix
    return actor_fix.actors_of(fix, original)


def evidence_actors(con, evidence_ids) -> set[str]:
    ids = [i for i in evidence_ids or () if i is not None]
    if not ids:
        return set()
    q = ",".join("?" * len(ids))
    rows = con.execute(f"SELECT id, source_actor FROM evidence WHERE id IN ({q})", ids).fetchall()
    fixes = _fixes(con, [r[0] for r in rows])                 # 说话人标注修正层（overlay）：没有记录用原标注
    out: set = set()
    for r in rows:
        out |= _actors_of(fixes.get(r[0]), r[1])
    return out


def evidence_allows(con, subject, evidence_ids) -> tuple[bool, str]:
    """按库里实际挂的证据判：主体是人时，至少一条证据要出自主体本人。没挂证据不判（放行）。"""
    if subject not in INGEST_PERSONS and subject != US:
        return True, ""
    actors = evidence_actors(con, evidence_ids)
    if not actors:
        return True, ""
    if subject == US:
        if us_ok(actors):
            return True, ""
        return False, f"subject-us：证据只来自 {'/'.join(sorted(actors))}，没有双方原话，不能记成 us 的事"
    if subject in actors:
        return True, ""
    return False, f"subject-echo：证据只来自 {'/'.join(sorted(actors))}，不能记成 {subject} 的事"


def note_block(subject, why: str) -> None:
    BLOCKED.append((str(subject), why))
    print(f"[anti-echo] 拒绝入库：subject={subject}（{why}）", file=sys.stderr)


# ── 召回侧：owner-support 信号 ─────────────────────────────────────────────
def provenance(con, ext_ids) -> dict[str, dict]:
    """{ext_id: {subject, status, pinned, n_support, n_owner, actors}}。只看 role='support' 的证据。"""
    ids = list(dict.fromkeys(i for i in ext_ids if i))
    out: dict[str, dict] = {}
    for k in range(0, len(ids), 500):
        part = ids[k:k + 500]
        q = ",".join("?" * len(part))
        rows = con.execute(
                f"""SELECT f.ext_id, f.subject, f.status, f.user_pinned, e.source_actor, e.id
                    FROM facts f
                    LEFT JOIN fact_evidence fe ON fe.fact_id = f.id AND fe.role = 'support'
                    LEFT JOIN evidence e ON e.id = fe.evidence_id
                    WHERE f.ext_id IN ({q})""", part).fetchall()
        fixes = _fixes(con, [r[5] for r in rows if r[5] is not None])
        for r in rows:
            ext, subj, status, pinned, actor = r[0], r[1], r[2], r[3], r[4]
            p = out.setdefault(ext, {"subject": subj, "status": status, "pinned": bool(pinned),
                                     "n_support": 0, "n_owner": 0, "actors": set()})
            if r[5] is not None and r[5] in fixes:           # 说话人标注修正层：这条证据实际算谁的话
                acts = _actors_of(fixes[r[5]], actor)
                if acts:
                    p["n_support"] += 1
                    p["actors"] |= acts
                    if subj in acts:
                        p["n_owner"] += 1
                continue
            if actor is not None:
                p["n_support"] += 1
                p["actors"].add(actor)
                if actor == subj:
                    p["n_owner"] += 1
    return out


def is_echo_only(p: dict | None) -> bool:
    """只有转述：主体是人、挂着证据、没有一条出自本人；confirmed / 钉住豁免。"""
    if not p:
        return False
    if p["pinned"] or p["status"] == "confirmed":
        return False
    if p["subject"] not in RECALL_PERSONS or p["n_support"] <= 0:
        return False
    if p["subject"] == US:                       # us：只有一方的证据 = 转述（task2-09）
        return not us_ok(p["actors"])
    return p["n_owner"] == 0


def echo_sources(con, ext_ids) -> dict[str, list[str]]:
    """只返回「只有转述」的那些：{ext_id: [证据说话人...]}（已排序）。"""
    return {ext: sorted(p["actors"]) for ext, p in provenance(con, ext_ids).items() if is_echo_only(p)}


def label(sources) -> str:
    """〔转述·来自助手〕。sources 为空返回空串。"""
    if not sources:
        return ""
    return "〔转述·来自" + "、".join(WHO_CN.get(s, str(s)) for s in sources) + "〕"


SELF_TAG = f"〔{persona.N_CLAUDE}自记〕"


def self_note_ids(con, ext_ids) -> set:
    """这些 ext_id 里哪些是助手自记的（fact_origin.origin='self-note'）。标记只存旁表，不在正文里；
    显示时才加 SELF_TAG（和〔转述〕同一个渲染层）。没表 / 出错当没有。"""
    ids = [i for i in ext_ids or () if i]
    if not ids:
        return set()
    try:
        q = ",".join("?" * len(ids))
        return {r[0] for r in con.execute(
            f"SELECT f.ext_id FROM facts f JOIN fact_origin o ON o.fact_id=f.id"
            f" WHERE o.origin='self-note' AND f.ext_id IN ({q})", ids)}
    except Exception:                                        # noqa: BLE001
        return set()


def annotate_origin(con, hits: list[dict]) -> list[dict]:
    """给召回结果就地打上 self_note（True/False）。幂等，失败不影响召回。"""
    try:
        sn = self_note_ids(con, [h.get("ext_id") for h in hits])
        for h in hits:
            h["self_note"] = h.get("ext_id") in sn
    except Exception:                                        # noqa: BLE001
        for h in hits:
            h.setdefault("self_note", False)
    return hits


def tag(h: dict) -> str:
    """一条命中前面的全部标注：〔转述·来自某某〕 + 〔助手自记〕（都没有就是空串）。"""
    return (label(h.get("echo_from")) + (SELF_TAG if h.get("self_note") else "")
            + (SAY_TAG if h.get("assistant_say") else ""))


# 助手原话：subject=claude 且正文是「我：…」或「某某（日期 时间）：…」这种引他自己说过的话。
# 它是他说过什么，不是关于她的事实；尤其是对她眼下状态的推断（「回宿舍再说」「回宿舍慢慢喝」）。
# 2026-10-28（任务四·步骤1B）：打 assistant_say 标记；召回时排序 × ECHO_WEIGHT、前面标〔我说的〕。
# 不删、不隐藏——他自己的承诺和决定还要能查到，只是不让它压过她本人的话和她的纠正。
RE_SAY = re.compile(r"^\s*(?:我：|[^（：\n]{1,10}（\d{4}-\d{2}-\d{2}[^）]{0,12}）：)")
SAY_TAG = "〔我说的〕"


def is_assistant_say(h: dict) -> bool:
    return h.get("subject") == persona.CLAUDE and bool(RE_SAY.match(h.get("content") or ""))


def annotate_say(hits: list[dict]) -> list[dict]:
    for h in hits:
        h["assistant_say"] = is_assistant_say(h)
    return hits


def demote_say(hits: list[dict]) -> list[dict]:
    """稳定地把助手原话排到她本人的话后面（原地；其余相对次序不变）。"""
    hits.sort(key=lambda h: bool(h.get("assistant_say")))
    return hits


def annotate(con, hits: list[dict]) -> list[dict]:
    """给召回结果就地打上 echo_from（只有转述才有）。已经打过的不重算。幂等，失败不影响召回。"""
    try:
        todo = [h["ext_id"] for h in hits if h.get("ext_id") and "echo_from" not in h]
        found = echo_sources(con, todo)
        for h in hits:
            if h.get("ext_id") in found:
                h["echo_from"] = found[h["ext_id"]]
            elif "echo_from" not in h:
                h["echo_from"] = []
    except Exception:                                        # noqa: BLE001
        for h in hits:
            h.setdefault("echo_from", [])
    annotate_say(hits)
    return hits


def tag_for(con, ext_id: str) -> str:
    """单条用：返回标注文字（不是转述返回空串）。"""
    try:
        return label(echo_sources(con, [ext_id]).get(ext_id))
    except Exception:                                        # noqa: BLE001
        return ""
