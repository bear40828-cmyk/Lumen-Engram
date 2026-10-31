#!/usr/bin/env python3
"""Lumen Engram · 记忆类型分类 v0.2

四类：preference（偏好）／semantic（稳定事实）／procedural（规矩做法）／episodic（一次性事件）

v0.1 的毛病被 GPT 一句话点破：「别人喜欢说你坏话 → preference」不是关键词的问题，
是 **experiencer 没判**——这个偏好到底属于谁。所以 v0.2 改成：

  1. 先把长文本切成分句（标点 + 转折词），逐句判，不再拿整段贴一个类型
  2. preference 要求偏好主体 == 这条事实的 subject；第三方的偏好不算她的偏好
  3. 报道／引用框架（听说、她说、据说、好像、原话是）默认按事件走
  4. 给出 type_confidence，低于阈值一律退回 episodic

最关键的一条也是它说的：**不确定就落 episodic。**
事件淡快一点最多以后不好想起来；错贴成偏好会让一条偶然八卦活一年，代价大得多。

半衰期是 **recency 半衰期**（只作用于激活度里的新近度那一项），不是总激活度的半衰期
——总分里还有不衰减的底座（重要度、置信度、被召回次数）。这个命名是 GPT 第二轮要求澄清的：
不然以后调参会误以为「事件一年后还剩 48%，半衰期是不是坏了」。
"""
from __future__ import annotations

import re
import persona

# ── 分句 ──────────────────────────────────────────────────────────────────
RE_CLAUSE = re.compile(
    r"[。！？；;!?\n]+|，(?=\S)|,(?=\S)|(?<=\S)(?:但是|但|不过|然后|而且|另外)")

# ── 报道／引用框架：命中就默认事件 ────────────────────────────────────────
RE_REPORT = re.compile(r"(听说|据说|他说|她说|别人说|人家说|转述|原话是|好像|似乎|传闻)")

# ── 偏好 ─────────────────────────────────────────────────────────────────
RE_PREF = re.compile(r"(喜欢|不喜欢|讨厌|最爱|爱吃|爱喝|想吃|想喝|受不了|不爱|爱看|爱玩)")
RE_PREF_NOUN = re.compile(r"(口味|偏好|最想要|吃不了|玩不了)")

FIRST = ("我", "咱", "俺")
SECOND = ("你",)
THIRD = ("她", "他", "它", "同事", "组员", "别人", "人家", "同学", "老师", "妈", "哥", "嫂",
         "侄女", "导员", "主播", "网友", "那位", persona.N_USER, persona.N_CLAUDE, persona.N_GPT, persona.N_C)

# ── 规矩做法 ──────────────────────────────────────────────────────────────
RE_PROC = re.compile(
    r"(一律|必须|不许|别再|以后要|以后别|规矩|说好|约定|默认|优先|步骤|流程|"
    r"先.{0,6}再|怎么(做|弄|起|跑|装|改)|命令|脚本|重启|部署|挂载|"
    r"记得|注意|千万|不能忘|下次|(之前|前)先|先(查|问|确认|摸|看)|别拿|不要拿)")

# ── 稳定事实 ──────────────────────────────────────────────────────────────
RE_SEM = re.compile(
    r"(老家|籍贯|本科|专业|毕业|生日|属于|叫做|名字是|是我的|我的.{0,4}是|"
    r"住在|在.{0,6}读|考(的是|了)?[^。]{0,6}(学校" + "".join("|" + persona.alt(t) for t in persona.CLASSIFY_TERMS) + r")|"
    r"\d+\s*%|满分|分值|构成|由.{0,8}组成|一共.{0,6}(科|门|条|人)|考.{0,3}科|"
    r"性别|年龄|多大|几岁)")

# ── 一次性事件标记 ────────────────────────────────────────────────────────
RE_EPI = re.compile(
    r"(今天|昨天|刚才|刚刚|方才|早上|中午|下午|晚上|凌晨|\d{1,2}[:：]\d{2}|"
    r"这次|那次|当时|然后|结果|一会儿|等会)")

CONF_FLOOR = 0.55          # 低于这个一律退回 episodic
PRIORITY = {"preference": 3, "semantic": 2, "procedural": 1, "episodic": 0}


def clauses(text: str) -> list[str]:
    return [c.strip() for c in RE_CLAUSE.split(text) if c and c.strip()]


# 「他妈」「他吗」「她妈」是脏话不是第三人称，先抹掉再判人，
# 否则「我真喜欢粉色」会被判成别人的偏好（v0.2 实测踩到）。
RE_SWEAR = re.compile(r"(他妈|他吗|她妈|妈的|尼玛)")


def experiencer(clause: str, verb_pos: int) -> str:
    """偏好动词前面那一小段里出现的是谁：self / second / other / unknown。

    取**离动词最近**的那个人称，而不是按类别优先——「我同事喜欢」里最近的是同事，
    「我他吗喜欢」抹掉脏话后最近的是我。
    """
    left = RE_SWEAR.sub("", clause[max(0, verb_pos - 8):verb_pos])
    best_pos, best_kind = -1, "unknown"
    for kind, words in (("other", THIRD), ("self", FIRST), ("second", SECOND)):
        for w in words:
            pos = left.rfind(w)
            if pos > best_pos:
                best_pos, best_kind = pos, kind
    return best_kind


def classify_clause(clause: str) -> tuple[str, float]:
    """判一个分句。返回 (类型, 置信度)。"""
    if RE_REPORT.search(clause):
        return "episodic", 0.7                     # 转述框架，先当事件

    m = RE_PREF.search(clause)
    if m:
        who = experiencer(clause, m.start())
        if who == "self":
            return "preference", 0.9
        if who == "second":
            return "preference", 0.6               # 说的是对方的偏好，置信度压低
        if who == "other":
            return "episodic", 0.8                 # 第三方的偏好，对这条只是见闻
        return "preference", 0.5                   # 主体不明
    if RE_PREF_NOUN.search(clause):
        return "preference", 0.7

    proc = bool(RE_PROC.search(clause))
    sem = bool(RE_SEM.search(clause))
    epi = bool(RE_EPI.search(clause))
    if proc and not sem:
        return "procedural", 0.75
    if sem and not epi:
        return "semantic", 0.8
    if sem and epi:
        return "semantic", 0.6
    if proc:
        return "procedural", 0.6
    return "episodic", 0.6


def classify(content: str, subject: str = persona.USER) -> tuple[str, float, str]:
    """整条事实的类型。返回 (类型, 置信度, 理由)。

    逐分句判，取置信度最高（同分看优先级）的非事件结论；
    只有一句支撑就打折；低于 CONF_FLOOR 退回 episodic。
    """
    body = content.split("：", 1)[-1].strip()
    if not body:
        return "episodic", 0.5, "空正文"

    votes = [classify_clause(c) for c in clauses(body)] or [("episodic", 0.5)]
    non_epi = [v for v in votes if v[0] != "episodic"]

    if not non_epi:
        return "episodic", max(v[1] for v in votes), f"{len(votes)} 个分句都判事件"

    kind, raw = max(non_epi, key=lambda v: (v[1], PRIORITY[v[0]]))
    share = sum(1 for v in votes if v[0] == kind) / len(votes)
    conf = round(min(0.98, raw * (0.75 + 0.25 * share)), 2)

    if conf < CONF_FLOOR:
        return "episodic", conf, f"想判 {kind}，置信度只有 {conf}，按规矩退回事件"
    return kind, conf, f"{len(votes)} 分句中 {kind} 占 {share:.0%}"


# recency 半衰期（天）。只作用于激活度里的新近度那一项，不是总分的半衰期。
RECENCY_HALF_LIFE = {
    "preference": 365.0,
    "semantic": 240.0,
    "procedural": 120.0,
    "episodic": 45.0,
}


def half_life(fact_type: str | None) -> float:
    """recency 半衰期，不是总激活度的半衰期。"""
    return RECENCY_HALF_LIFE.get(fact_type or "episodic", 45.0)


if __name__ == "__main__":
    U, A, B = persona.N_USER, persona.N_CLAUDE, persona.N_GPT
    samples = [
        (f"{U}：我比较喜欢红色，不太喜欢绿色", persona.USER),
        (f"{B}：他昨天在会上反驳了你的方案", "gpt"),
        (f"{U}：我同事习惯把快递堆在工位旁边", persona.USER),
        (f"{U}：听说她喜欢红色", persona.USER),
        (f"{U}：项目下个季度结束", persona.USER),
        (f"{A}：这门课考三部分，每部分占比相同", "claude"),
        (f"{U}：回复一律用纯文本", persona.USER),
        (f"{U}：今天出门忘带钥匙又折回去了", persona.USER),
    ]
    for text, subj in samples:
        kind, conf, why = classify(text, subj)
        print(f"{kind:<11}{conf:>5.2f}{half_life(kind):>6.0f}天  {text[:34]:<36}{why}")
