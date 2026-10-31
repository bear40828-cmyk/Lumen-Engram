"""说话人归属：cc-connect 会话记录和信箱两条入库路径共用同一套名单和映射（任务三，2026-10-28）。

防回声（I-1）整个靠「这句话是谁说的」：证据 source_actor 错了，subject 就跟着错。
原来两条路各写各的：会话记录（extract/from_channel.harvest）按 user_id / 工具名认人，认不出标 unknown:*；
信箱（ingest_mailbox.py）却是 from=='me'→user，**其余一律 claude**——任何不是她的人（GPT、GLM 助手、
未来的别的 bot）发的话都会被记成助手说的。这里统一成：认得出才有名字，认不出就是 None（不入库、计数），
绝不猜成助手。

本模块没有开关：只是名单和纯函数，不碰库。
"""
from __future__ import annotations

import persona

VALID_ACTORS = persona.ACTORS
WHO = {persona.USER: persona.N_USER, "claude": "我", "gpt": persona.N_GPT, persona.AGENT_C: persona.N_C}

# 信箱 messages.jsonl 里的 from 字段。'me' 是她本人（原脚本就这么认）；其余只认明确写着是谁的。
# 部署特有的写法用 ENGRAM_MAILBOX_ALIASES（「值:标签,…」）补。
# 认不出的值一律 None，由调用方计数后跳过——不再默认算助手。
_MAILBOX = {
    "me": persona.USER, persona.USER: persona.USER, "user": persona.USER, persona.N_USER: persona.USER,
    "claude": "claude", persona.N_CLAUDE: "claude", "assistant": "claude", "ai": "claude",
    "gpt": "gpt", persona.N_GPT: "gpt",
    persona.AGENT_C: persona.AGENT_C, persona.N_C: persona.AGENT_C,
}
_MAILBOX.update(persona.pairs("ENGRAM_MAILBOX_ALIASES"))


def normalize_actor(actor) -> str | None:
    """会话记录路径（harvest）给出的 actor：在名单里原样返回；unknown:* 或别的返回 None。"""
    a = str(actor or "").strip()
    return a if a in VALID_ACTORS else None


def mailbox_actor(msg: dict) -> str | None:
    """信箱一条消息的说话人。认不出返回 None（不要猜）。"""
    v = str((msg or {}).get("from") or "").strip().lower()
    return _MAILBOX.get(v) or _MAILBOX.get(str((msg or {}).get("from") or "").strip())


# ── Telegram channel 块的说话人（bot 转发被标成她的话）──────────────────────────
# <channel source=… user="某_bot" user_id="<数字>" …>：user_id 是平台给的、不会重名，优先；
# 只有名字时才认名字。名单之外的（包括任何别的 *_bot）一律 None——不入库、计数，绝不默认算她。
# 数据里 user 属性有时直接是她的数字 id，当 id 用。
# 名单来自配置：ENGRAM_CHANNEL_IDS=「id:标签,…」、ENGRAM_CHANNEL_NAMES=「名字:标签,…」（见 persona.py）。
CHANNEL_BY_ID = persona.CHANNEL_BY_ID
CHANNEL_BY_NAME = persona.CHANNEL_BY_NAME


def channel_actor(user, user_id=None) -> str | None:
    """channel 块的 user / user_id 属性 → 说话人。id 和名字都给且对不上 → None（宁可不认）。"""
    u, i = str(user or "").strip(), str(user_id or "").strip()
    by_id = CHANNEL_BY_ID.get(i) or (CHANNEL_BY_ID.get(u) if u.isdigit() else None)
    by_name = CHANNEL_BY_NAME.get(u)
    if by_id and by_name and by_id != by_name:
        return None
    return by_id or by_name


import re as _re

_CH_ATTR = _re.compile(r'(\w+)="([^"]*)"')
_CH_LEAD = _re.compile(r'\A\s*<channel\s+(?P<attrs>[^>]*)>', _re.S)


def leading_channel(body: str):
    """正文以 <channel …> 开头（整条是转发包装）→ (user, user_id)，否则 None。"""
    m = _CH_LEAD.match(body or "")
    if not m:
        return None
    a = dict(_CH_ATTR.findall(m.group("attrs")))
    return a.get("user", ""), a.get("user_id", "")


def reconcile(actor, body: str):
    """入库时的最后一道：正文本身是 channel 包装时，说话人以包装里的 user/user_id 为准。
    返回 (actor 或 None, 备注)。备注：None＝没动；'relabeled'＝改标了；'unknown-channel-user'＝认不出（调用方计数后跳过）。"""
    lead = leading_channel(body)
    if lead is None:
        return actor, None
    mapped = channel_actor(*lead)
    if mapped is None:
        return None, "unknown-channel-user"
    if mapped != actor:
        return mapped, "relabeled"
    return actor, None
