#!/usr/bin/env python3
"""Lumen Engram · gate —— 晋升门控

设计出自 app 里的 GPT，落盘由助手。它自己纠正过一句很关键的话：

    gate 管的不是「写不写库」，而是「要不要从原始证据层晋升成长期记忆」。

所以这里**没有任何一条判据能让证据消失**。gate 说不，最多是这条候选不进 facts；
evidence 该存的照存，将来判据改了还能重灌。

v0.2（2026-08-17 验收后重写）
--------------------------------
第一版拿「长度 < 6 字」当否决条件，被它当场否掉，理由是对的：
中文短事实信息密度极高——「考某大学」「爱粉色」「养宠物」都短得离谱但全是要记的。
而且第一版实测晋升率 97.8%，等于门控几乎没在筛，「这表情什么都没说啊」
「收下这颗心」这种纯聊天全进了长期记忆。

所以改成：**看有没有事实槽位，长度只当弱特征。**
有槽位（实体／数字日期／偏好／关系／状态变化／规矩承诺）就晋升，不管多短；
没有槽位就不晋升，不管多长。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import sha256  # noqa: E402
import persona  # noqa: E402

# ── 事实槽位：命中任意一个就算「有东西可记」 ──────────────────────────────
SLOTS: dict[str, re.Pattern] = {
    # 数字、日期、时间、金额、比例
    "数值": re.compile(r"\d{1,4}\s*[年月日号点分岁%元块条页分钟小时天周]|\d{3,}|[一二三四五六七八九十百千]+[年月日号点岁块条]"),
    # 专有名词：项目、地名、机构、游戏、平台
    "专名": re.compile(persona.alt(*persona.GATE_TERMS) or r"(?!x)x"),      # 配置：ENGRAM_GATE_TERMS
    # 人物与关系
    "人物": re.compile(persona.fill(r"@@alt:USER_NAMES,CLAUDE,GPT,C,PET@@|同事|我哥|我妈|我爹|嫂|侄女|导员") + persona.person_re()),
    # 偏好与态度（是关于人的稳定事实）
    "偏好": re.compile(r"喜欢|讨厌|不爱|最爱|爱吃|想吃|受不了|怕|不敢|习惯|偏好|口味|最想"),
    # 状态变化：做完了、改了、上线了、买了、考出了
    "变化": re.compile(
        r"已经|做完|写完|跑通|上线|改成|换成|删了|买了|考出|拿到|收到|通了|定了|"
        r"结束|开始|搬|重启|部署|修好|坏了|丢了|没了|回来了|到了|走了|睡了|起了"),
    # 规矩、承诺、边界
    "规矩": re.compile(r"不许|别再|以后要|以后别|规矩|说好|承诺|答应|不能|必须|一律|默认|优先"),
    # 明确的纠正与认错（这类对我特别重要）
    "纠错": re.compile(r"记错|说错|搞错|归错|错了|我认|收回|更正|纠正|幻觉|不是我|不该"),
}

# 纯寒暄／情绪／应答：即使字数不少也不该进长期记忆
CHATTER = re.compile(
    r"^(嗯+|哦+|噢+|啊+|哈+|呵+|嘿+|行|好|好的|好呀|收到|在|在呢|是|对|不|没|草|靠|1|"
    r"笑死|绷不住|可爱|乖|抱抱|亲亲|想你|爱你|晚安|早安|早|睡了|去洗澡|洗澡去|"
    r"没什么|没事|随便|都行|不知道|谁知道|真的吗|是吧|对吧|好吧|算了)"
    r"[。，,！!？?~、…\s🌚👀🤡💜😋👍🏻😪🥺✅]*$")
NO_TEXT = re.compile(r"^[\W\s_]*$", re.UNICODE)


@dataclass
class Verdict:
    promote: bool
    reason: str          # 给人看的
    code: str = ""       # 给统计用的稳定代号，别让 heuristic 只活在文案里
    importance: float | None = None
    confidence: float | None = None
    slots: tuple[str, ...] = ()


def find_slots(text: str) -> tuple[str, ...]:
    return tuple(name for name, pat in SLOTS.items() if pat.search(text))


def judge(con, *, subject: str, content: str, evidence_rows: list[dict],
          user_pinned: bool = False) -> Verdict:
    """evidence_rows 是这条候选挂着的证据（至少要有 verification_status）。"""
    body = content.split("：", 1)[-1].strip()

    # 她明确要求留的，任何判据都不能拦
    if user_pinned:
        return Verdict(True, "user_pinned：她要求留的，门控不参与", "pinned", 1.0, 1.0)

    if NO_TEXT.match(body):
        return Verdict(False, "只有表情或标点", "no_text")

    # 先看证据——没有可核的东西，一律不晋升
    if not evidence_rows:
        return Verdict(False, "没有可核对的证据，留在证据层", "no_evidence")
    verified = [e for e in evidence_rows if e.get("verification_status") == "verified"]
    if not verified:
        worst = "/".join(sorted({e.get("verification_status") or "?" for e in evidence_rows}))
        return Verdict(False, f"证据未通过核对（{worst}）", "evidence_unverified")

    slots = find_slots(body)

    # 纯寒暄：只要没有事实槽位就拒，字数多少无关
    if CHATTER.match(body) and not slots:
        return Verdict(False, "寒暄或情绪应答，没有事实槽位", "chatter_no_slot")
    if not slots:
        return Verdict(False, f"找不到事实槽位（{len(body)} 字）", "no_slot", slots=slots)

    # 只蹭到一个人名、话又短，那是打招呼不是事实（「克先生手段了得🤡」）。
    # 长度在这儿只当弱特征用来给单槽位兜底，不当独立否决条件。
    if slots == ("人物",) and len(body) < 12:
        return Verdict(False, f"只提到人名、没别的事实（{len(body)} 字）", "short_person_only", slots=slots)

    if con.execute("SELECT 1 FROM facts WHERE subject=? AND content_sha256=?",
                   (subject, sha256(content))).fetchone():
        return Verdict(False, "库里已有一模一样的", "duplicate", slots=slots)

    # 置信度只看证据强度；重要度看槽位多少和信息密度。都不看情绪。
    confidence = min(1.0, 0.6 + 0.1 * len(verified))
    importance = min(0.95, 0.3 + 0.12 * len(slots) + (0.1 if len(body) >= 30 else 0.0))
    return Verdict(True, "有 " + "＋".join(slots), "has_slots", round(importance, 2),
                   round(confidence, 2), slots)
