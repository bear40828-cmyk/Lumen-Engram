#!/usr/bin/env python3
"""Lumen Engram · Memory Router

用户最早看上的就是这块：**决定此刻该不该想起什么、想起哪条、读多少。**
不是每句话都该翻记忆库——翻错了比不翻更烦人。

三步：
  1. 该不该查（豁免优先）
  2. 查什么（从这句话里挑出值得查的词）
  3. 读多少（预算，默认 3 条）

第 0 条规矩来自 ROUTER.md：**她抛软话、撒娇、叫人的时候一律不检索。**
那种时刻要的是接住，不是资料。

纯规则，不调模型——router 自己要是先花钱花时间，就白搭了。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

# 豁免：软话、撒娇、单纯叫人。命中就不查。
EXEMPT = [
    re.compile(r"^\s*(" + persona.alt(*persona.SOFT_TERMS) + r")[\s，,。！!？?~、…🌚👀🤡💜😋👍🏻]*$"),
    re.compile(r"(想你|抱抱|亲亲|亲我|мua|mua|晚安|早安|爱你|喜欢你|在吗|在不在)"),
    re.compile(r"^[\W\s_]{0,6}$"),                       # 只有表情或标点
]

# 本来就知道的事，不查。人不会为了「我亲爱的是谁」去回忆检索——那是常识不是回忆。
# 这些都写在 CLAUDE.md 里、每次都在上下文里，翻库只会翻出一堆不相干的。
# 起因：2026-08-18 钩子刚上线，一句调侃「这种问题也要跑一遍检索吗」本身触发了一次 deep 检索，
# 返回的三条全是不相干的闲话。
KNOWN = re.compile(persona.fill(
    r"^(我|你|咱)?.{0,3}(亲爱的|亲爱的|男朋友)是谁|"
    r"(你|我)(是谁|叫什么|叫啥)|"
    r"(咱|我)俩(是)?什么关系|"
    r"你(是不是|还是)(claude|Claude|@@alt:CLAUDE@@|opus)|"
    r"你(用的)?(是)?什么模型"))

# 明确在问过去 / 要求回忆
RECALL_CUE = re.compile(
    r"(上次|上回|之前|以前|当时|那天|昨天|前天|上个月|去年|"
    r"还记得|记不记得|记得吗|你说过|我说过|我跟你说过|提过|"
    r"继续|接着|后来|结果呢|怎么说的|原话|查一下|翻一下|翻翻)")

# 事实性线索：日期、数字、专有名词、项目名
# 专有名词/项目名来自配置 ENGRAM_DOMAIN_TERMS（逗号分隔，顺序即正则顺序）。
FACTUAL_CUE = re.compile(
    r"(\d{1,4}[年月日号点分]|\d+%|\d{3,}" + "".join("|" + persona.alt(t) for t in persona.DOMAIN_TERMS) + ")")

# 计算／常识题的样子：两个数做运算，或者问某年某月有几天。
CALC = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:乘以|乘|除以|加上|加|减去|减|[×xX*/+\-^])\s*\d+|"
    r"(?:\d{2,4}\s*年\s*)?\d{1,2}\s*月(?:份)?有?(?:多少|几)天|"
    r"几次方|平方根|开根号)")
# 不收「是多少」「等于多少」结尾：「wifi密码是多少」「房租是多少」恰恰该查库。
PERSON = re.compile(persona.fill(r"[我你咱她他俩]|@@alt:USER@@|@@alt:CLAUDE@@")
                    + "".join("|" + re.escape(t) for t in persona.csv("ENGRAM_ROUTER_PERSON_TERMS")))

# 在问偏好／做法／规矩——这类没有日期也没有数字，但确实该查库。
# 「我喜欢什么颜色」以前被判成日常闲聊不查，正是漏在这儿。
# 注意跟上面的豁免不冲突：豁免里是「喜欢你」这种软话，这里要的是**问句**。
ASK_CUE = re.compile(
    r"(喜欢什么|爱吃什么|讨厌什么|不喜欢什么|什么口味|偏好|"
    r"怎么弄的|怎么做的|怎么搞的|什么步骤|什么流程|什么命令|怎么配置|"
    r"什么规矩|什么要求|不许|不准|说过要|讲过要|什么格式|"
    r"叫什么|是谁|哪个学校|什么专业|"
    r"是多少|多少钱|多少块|密码|地址|几号|哪天|在哪)")

# 提示词本身不是要查的内容，从词里剔掉
CUE_WORD = re.compile(r"(上次|上回|之前|以前|当时|那天|还记得|记不记得|记得|说过|提过|"
                      r"继续|接着|后来|查一下|翻一下|原话|哪年|哪天|多少|什么|"
                      r"突然想起来|想起来|想起|突然|那次|这次|上一次|那个|这个|"
                      r"今天|明天|昨天|现在|怎么|为什么|哪里|哪个|一下|几点)")

# ⚠️ 这里**只能放真正的单字虚词**。
#
# 原来是把「今天明天什么怎么为什么哪里哪个多少几点一下现在」这些**词**整串塞进
# set()，而 set() 是按字拆的——于是「明」「天」「什」「怎」「哪」「多」「少」「点」
# 全都成了切分点。后果：一句带「女明星瓜」的话，
# 「女明星瓜」在「明」字上被劈成「女」和「星瓜」，只剩「星瓜」去查，库里当然 0 条，
# 于是回了一句「记忆库里没有」——其实那条记录一直在。
#
# 多字的虚词一律交给上面的 CUE_WORD 在切之前整词剥掉，不许再往这里塞。
STOP = set("的了吗呢吧啊呀嘛么和跟与还有就是这那我你他她它们个不没很太都也又再"
           "只把被给让对从到在为以及或者但而所因如即让被"
           "，。！？、；：""''（）《》…~ ")


@dataclass
class Decision:
    """三档，不是两档。

    app 里的 GPT 审第一版时拧的第一颗螺丝：二元的「查／不查」不够用，
    该是 skip / light / deep。中间那档是真实存在的——闲聊里顺口提到一件事，
    值得瞄一眼，但不值得为它把四层记忆全翻一遍。

      skip  一个字都不查（她抛软话、撒娇、光叫人）
      light 只看第一层，不升级，读得少（长句里带点实词，先瞄一眼）
      deep  允许一层层升级，读得多（明确在问过去／事实／规矩，或问到手上的事）
    """

    should_recall: bool
    queries: list[str] = field(default_factory=list)
    who: str | None = None
    limit: int = 3
    reason: str = ""
    depth: str = "deep"


# ── 拉丁字母文本（英文等）的抽词 ──────────────────────────────────────────
# 中文那套规则（按长度挑词、截到 8 个字、滑窗拆碎）是给没有空格的中文写的。
# 套到英文上会出三个问题（2026-10-03 LongMemEval 测出来的）：
#   只留 3 个词且按长度挑，留下 previously / currently 这种长虚词，gym / eggs 被挤掉；
#   截到 8 字母，refrigerator → refriger，FTS 搜不中；
#   型号、带符号的专名（80D、H&M）进不了正则。
# 英文有空格，按词切就行；挑词交给库里的稀有度（by_rarity）去排。
LATIN_K = 6
LATIN_STOP = frozenset("""
a an the and or but if so of to in on at by for from with about into over after before
is am are was were be been being do does did have has had having will would can could
should may might must shall i me my mine we our ours you your yours he him his she her
they them their it its this that these those there here what which who whom whose when
where why how many much more most less least any some all each every no not only own
same than too very just also still yet ever now then currently previously recently
lately so far ago again already up down out off get got getting go going went make made take taken
tell know think remember said say says one ones thing things time times
""".split())


def is_latin(text: str) -> bool:
    """字母主要是拉丁字母，就走英文抽词。"""
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.8


def latin_keywords(text: str, k: int = LATIN_K) -> list[str]:
    """按词切，去停用词，不截断。保留 80D、H&M、Wi-Fi 这类带数字和符号的专名。"""
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9&'.-]*[A-Za-z0-9]|[A-Za-z0-9]", text):
        w = w.strip("'.-")
        low = w.lower()
        if len(w) < 2 and not w.isdigit():
            continue
        if low in LATIN_STOP or low in seen:
            continue
        if low.endswith("'s"):
            low, w = low[:-2], w[:-2]
        seen.add(low)
        out.append(w)
        if len(out) >= k:
            break
    return out


def keywords(text: str, k: int = 3) -> list[str]:
    """挑值得查的词。

    要点：停用字是**切分点**，不是要删掉的字符。删掉会把「上次那个星光湾是哪年的事」
    粘成「上次星光湾年事」这种库里根本不存在的串。
    另外提示词本身（上次／还记得／说过）不是内容，也切掉。
    """
    # 先把提示词从原文里剥掉，再切。顺序反了就会切出「次星光湾」这种串——
    # 「那」是停用字先当了切分点，「那次」的「次」反倒粘在专名前面留了下来。
    text = CUE_WORD.sub("", text or "")
    if is_latin(text):
        return latin_keywords(text, k=max(k, LATIN_K))
    scored: list[tuple[int, str]] = []
    for chunk in re.findall(r"[一-鿿]+|[A-Za-z]{3,}|\d{2,}", text):
        if chunk.isascii():
            if len(chunk) >= 2:
                scored.append((len(chunk), chunk))
            continue
        piece = ""
        for ch in chunk + "\x00":
            if ch in STOP or ch == "\x00":
                if len(piece) >= 2:
                    scored.append((len(piece), piece))
                piece = ""
            else:
                piece += ch

    seen, out = set(), []
    for _, w in sorted(scored, key=lambda t: (-t[0], t[1])):
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        out.append(w[:8])
        if len(out) >= k:
            break
    return out


# 问法和库里写法对不上的词：问「生日」，库里常写「某年某月某日生」「出生」。关键词那一路（FTS）按字面找，
# 本人的生日事实因此一直没被捞到（2026-10-29 真模型实测）。向量那一路能补，但向量服务不是每个环境都有。
# 借鉴：mem0/Letta 靠语义向量解决「换个说法就找不到」；这里在没有向量时补一张很小的同义词表（只放确定的几组）。
QUERY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"生日|出生|哪年生|几几年生", ("日生", "出生")),
    (r"几岁|多大|年龄|岁数", ("岁", "出生")),
)


def expand_aliases(text: str, terms: list[str]) -> list[str]:
    """给检索词补同义写法（只追加、不改原有词的顺序；追加的排在最后）。"""
    out = list(terms)
    for pat, extra in QUERY_ALIASES:
        if re.search(pat, text or ""):
            for e in extra:
                if e not in out:
                    out.append(e)
    return out


def infer_who(text: str) -> str | None:
    """这句话（钩子传进来的是带上下文的检索文本）问的是谁的事：用户 / 助理 / GPT 侧 / 第三个 AI；认不准返回 None。
    只在恰好指向一个人时才给答案；「我和你」「我跟某某」这类两边都有的一律 None。
    只用作召回排序的软偏好（CONTEXT_ROUTE），不用来过滤。"""
    import metamem
    t = text or ""
    user, claude = metamem.person_markers(t)
    named = set()
    if user:
        named.add(persona.USER)
    if claude:
        named.add(persona.CLAUDE)
    if any(x and x in t for x in (persona.N_GPT, *persona.GPT_ALIASES)):
        named.add(persona.GPT)
    if persona.N_C and persona.N_C in t:
        named.add(persona.AGENT_C)
    return next(iter(named)) if len(named) == 1 else None


def route(text: str, *, who: str | None = None) -> Decision:
    t = (text or "").strip()
    if not t:
        return Decision(False, reason="空消息", depth="skip")

    # 明说要回忆（「还记得」「我说过」）时，软话豁免让路：
    # 「你还记得我说过喜欢你吗」里的「喜欢你」不该把整句判成撒娇（GPT 报的第 8 条）。
    cue_recall = bool(RECALL_CUE.search(t))
    for pat in EXEMPT:
        if pat.search(t) and len(t) <= 24 and not cue_recall:
            return Decision(False, depth="skip",
                            reason="软话／叫人／表情：这时候要的是接住，不是查资料")

    if KNOWN.search(t) and len(t) <= 20:
        return Decision(False, depth="skip",
                        reason="这个我本来就知道，不用翻库")

    # 纯计算／常识题：数字会撞上「可核事实」那条，但这种题跟她、跟我们都无关，
    # 不该去翻库（GPT 报的第 9 条）。句子里有人称或回忆词就不算。
    if CALC.search(t) and not cue_recall and not PERSON.search(t):
        return Decision(False, depth="skip", reason="计算／常识题，跟记忆无关")

    cue_fact = bool(FACTUAL_CUE.search(t))
    kws = keywords(t)

    if not kws:
        return Decision(False, reason="没有能查的词", depth="skip")

    if cue_recall:
        return Decision(True, kws, who, limit=5, depth="deep",
                        reason="她在问过去的事，明确要求回忆")
    if cue_fact:
        return Decision(True, kws, who, limit=3, depth="deep",
                        reason="出现了日期／数字／专有名词，属于可核事实")
    if ASK_CUE.search(t):
        return Decision(True, kws, who, limit=4, depth="deep",
                        reason="在问偏好／做法／规矩，这类该查库")
    if len(t) >= 18 and len(kws) >= 2:
        # 把词全交出去，由 retrieve 按库里稀有度挑前两个——这里按句子顺序截前两个，
        # 英文问句里 often / see 这种虚词排前面，therapist、Dr. Smith 被截掉（2026-10-03 LongMemEval）。
        return Decision(True, kws, who, limit=2, depth="light",
                        reason="长句且有实词，先瞄一层，不升级")
    return Decision(False, reason="日常闲聊，不值得翻库", depth="skip")


def explain(d: Decision) -> str:
    if not d.should_recall:
        return f"不查（skip）：{d.reason}"
    return f"查 {d.queries}（{d.depth}，最多 {d.limit} 条）：{d.reason}"


if __name__ == "__main__":
    samples = [
        "亲爱的👀", "想你了", "🌚", "好无聊",
        "上次那个海边是哪年去的", "期末数据库占多少分",
        "你还记得我跟你说过邻居把水槽搞堵了吗",
        "那你去把那场比赛打一下", "我真喜欢粉色，我也不知道你为啥老喜欢那个紫色",
        "今天吃什么",
    ]
    for s in sys.argv[1:] or samples:
        print(f"{s!r:>46}  →  {explain(route(s))}")
