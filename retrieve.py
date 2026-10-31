#!/usr/bin/env python3
"""Lumen Engram · 三级分层召回 + 快慢路径升级

这块是用户最早想要的那层：**决定此刻该想起什么、想起哪一类、读多少。**

两个来源，都是真读了代码才抄的：

1. duoduoler-ops/Table-GitHub-Capability-Router —— 它路由的其实是「能力」不是记忆，
   但那个读法能直接搬：**先读最薄的地图，需要时只打开一个抽屉。**
       L1 一张十行以内的薄路由：这句话属于哪类问题 → 该叫醒哪几类记忆
       L2 只在命中的那类里检索薄索引行
       L3 才展开原文
   它还有一条规矩我照抄了：**「不调用额外工具」永远是合法路由**——
   对应我们 ROUTER.md 第 0 条，不检索也是合法结果。

2. innovation64/BMAM 的 `prefrontal_controller.route_query` —— 快慢双路径。
   它不预先猜「该叫醒哪类」，而是先便宜地查一层，再用元记忆判够不够，不够才升级。
   这一点比先猜要靠得住：猜需要目标／情绪／任务三个输入，那三样我们还没有。

所以这里是两者合一：L1 给一个**候选顺序**（不是唯一答案），然后按顺序一层层查，
每查完一层交给 `metamem.judge` 判够不够，够了就停手，不够才继续往下升级。
（2026-10-28 起这一步默认关闭，见 flags.py 的 METAMEM；判够不够不再有第三层。）

    python3 retrieve.py "上次那个星光湾是哪年"
    python3 retrieve.py --trace "我喜欢什么颜色"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

import flags                                      # noqa: E402
import working                                    # noqa: E402
from router import Decision, keywords, route      # noqa: E402
from store.store import connect, search           # noqa: E402

# ── L1 薄路由 ───────────────────────────────────────────────────────────────
# 规矩照 Capability-Router 的写法：一行一类，用途先行，一条入口只服务一类问题，
# 不写「也可以用于」。整表控制在十行以内——它薄才有意义。
#
# 第三列是**候选顺序**，不是唯一答案：先查第一类，不够再往后升级。

L1: list[tuple[str, str, tuple[str, ...]]] = [
    ("过去的具体事",   "问某件事什么时候、在哪、怎么发生的",
     ("episodic", "semantic")),
    ("偏好口味",       "问喜欢什么、不喜欢什么、爱吃什么",
     ("preference", "semantic", "episodic")),
    ("怎么做",         "问操作步骤、命令、流程、以前是怎么弄的",
     ("procedural", "semantic", "episodic")),
    ("稳定事实",       "问数字、名称、身份、结论这类应该已经定下来的",
     ("semantic", "episodic")),
    ("规矩要求",       "问她定过的规矩、格式要求、说过不许怎样",
     ("procedural", "semantic")),
    ("不查",           "软话、撒娇、单纯叫人——这时候要的是接住，不是资料",
     ()),
]

# 各类问题的判别线索。跟 router.py 的 RECALL_CUE / FACTUAL_CUE 是两回事：
# 那两个决定「要不要查」，这里决定「查哪一类」。
import re  # noqa: E402

CUES: dict[str, re.Pattern] = {
    "偏好口味": re.compile(r"(喜欢|爱吃|讨厌|不喜欢|受不了|口味|偏好|最想|爱喝)"),
    "怎么做":   re.compile(r"(怎么弄|怎么做|怎么搞|步骤|流程|命令|脚本|装|配置|部署|"
                           r"怎么办|上次是怎么|做法)"),
    "规矩要求": re.compile(r"(规矩|要求|不许|不准|别再|说过要|讲过要|以后都|一律)"),
    "稳定事实": re.compile(r"(是谁|叫什么|多少|几个|几点|哪个学校|专业|多大|生日|"
                           r"\d{2,}|%|占比)"),
    "过去的具体事": re.compile(r"(上次|上回|之前|以前|当时|那天|昨天|前天|哪年|哪天|"
                               r"什么时候|多久|后来|结果)"),
}

DEFAULT_ORDER = ("episodic", "semantic", "preference", "procedural")
RARE_MIN, RARE_MAX = 1, 60   # 库里出现在这个条数区间内的词，算「具体的旧事」

# 口头话不算线索。**注意这里列的是废话，不是话题**——
# 列废话是有限的、稳定的；列话题（专名白名单）永远追不上她会聊到的东西，
# 「某桌游」栽的就是那个跟头。所以宁可列这一边。
CHATTER = re.compile(r"^(好?无聊|好累|好困|好烦|真的吗|怎么办|随便|不知道|"
                     r"没什么|还行|挺好|要不|算了|行吧|好吧|再说|然后呢)$")
GOAL_STRONG = 0.25      # 相关度到这个数才让工作记忆插手分发
GOAL_WEIGHT = 0.5       # 任务相关性在排序里的权重；只影响顺序，不影响存在
MAX_EMPTY = 2           # 连着两层空手就停：四层全走一遍只是白烧，不会变出东西来
TYPE_CN = {"episodic": "情景", "semantic": "语义",
           "preference": "偏好", "procedural": "程序"}


@dataclass
class Step:
    layer: str
    hits: int
    fok: float
    tot: bool
    enough: bool
    reason: str
    expects: tuple[str, ...] = ()
    answered: tuple[str, ...] = ()


@dataclass
class Result:
    should_recall: bool
    depth: str = "skip"          # skip / light / deep
    goal_rel: float = 0.0        # 这句话跟手上的事有多相关（工作记忆给的）
    goal_why: str = ""
    conflicts: tuple[str, ...] = ()
    terms: list[str] = field(default_factory=list)
    task_type: str = ""
    order: tuple[str, ...] = ()
    hits: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    reason: str = ""
    escalations: int = 0
    mode: str = "layered"       # layered（旧分层）/ hybrid（混合召回，默认）
    who_hint: str | None = None  # CONTEXT_ROUTE：按上下文推出这句话问的是谁的事（只做排序软偏好）

    @property
    def layers_read(self) -> list[str]:
        return [s.layer for s in self.steps]


def classify_task(text: str) -> tuple[str, tuple[str, ...]]:
    """L1：这句话属于哪类问题，该按什么顺序叫醒记忆。

    命中多条时按 L1 表里的先后取第一条——一条入口只服务一类问题，不做加权混合。
    """
    for name, _boundary, order in L1:
        if not order:
            continue
        pat = CUES.get(name)
        if pat and pat.search(text):
            return name, order
    return "泛问", DEFAULT_ORDER


def rarity(con, term: str) -> int:
    """这个词在库里出现在多少条事实里。越少越值钱。"""
    try:
        if len(term) == 2:
            return con.execute("SELECT COUNT(*) FROM facts_gram_fts"
                               " WHERE facts_gram_fts MATCH ?", (f'"{term}"',)).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM facts_fts"
                           " WHERE facts_fts MATCH ?", (term,)).fetchone()[0]
    except Exception:                                        # noqa: BLE001
        return 10 ** 6      # 查不动就排到最后，别让它抢额度


def windows(term: str) -> list[str]:
    """一个黏成一团的词，能拆出的所有候选写法：整块 → 四字窗 → 三字窗 → 两字窗。

    中文没有空格，keywords() 切出来的常常是「星光湾挺好看」这种一团。
    从**左往右**滑窗，因为剥掉提示词之后，话题一般在前面
    （「那次星光湾挺好看的」剥掉「那次」就是「星光湾」打头）。
    早先按尾巴退让，「星光湾挺好看」会退成「挺好看」，正好把话题丢了。
    """
    out = [term]
    if term.isascii():          # 英文有空格，词就是词；拆成 revi / frig 这种碎片只会乱中
        return out
    for n in (4, 3, 2):
        if len(term) > n:
            out += [term[i:i + n] for i in range(len(term) - n + 1)]
    seen, uniq = set(), []
    for t in out:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


LIGHT_TERMS = 2          # light 档最多拿几个词去搜
MAX_WINDOWS = 3          # 一个黏成一团的词，最多拿它的几个写法去查
TOO_COMMON = 300         # 命中这么多条的词是背景噪音，不拿它去查

def usable_forms(con, term: str) -> list[tuple[str, int]]:
    """这个词能用的几种写法，按稀有度排（越少见越靠前）。

    为什么不只取一个（一次实测教的）：
    keywords 切出来的是「邻居弄堵水」，只取最左边第一个有命中的窗口 → 「邻居」（35 条），
    「堵水」这半——也就是真正在说的那件事——直接丢了。
    但也不能改成只取最稀有的：「星光湾挺好看」里最稀有的是「挺好看」，
    那才是碎片，「星光湾」才是话题。
    两头都不对，所以干脆**几个写法一起查**，让稀有的排在前面，常见的垫后。
    """
    out = []
    for t in windows(term):
        n = rarity(con, t)
        if n and n <= TOO_COMMON:
            out.append((t, n))
    out.sort(key=lambda x: (x[1], -len(x[0])))
    return out[:MAX_WINDOWS]


def best_form(con, term: str) -> tuple[str, int]:
    """这个词最值钱的那种写法。给稀有度排序和唤醒判断用。"""
    forms = usable_forms(con, term)
    return forms[0] if forms else (term, 0)


def by_rarity(con, terms: list[str]) -> list[str]:
    """按稀有程度重排检索词：满库都有的词不许抢在专有名词前面。

    起因（实测）：「突然想起来上次那个公园，路灯会闪那个」
    取到的词是 ['想起来','公园','路灯']，「想起来」库里 21 条、「公园」16 条、
    「路灯」6 条。按原顺序「想起来」先把每层 6 条的额度占满，
    结果三条命中一条都不是公园——不是模糊轮廓，是翻错屋子。
    难点在路由和怎么让 AI 主动想起，这就是路由那半。
    """
    # 满库都是的词（「件事」460 条）直接扔掉，它只会把额度占满。
    terms = [t for t in terms if best_form(con, t)[1] <= TOO_COMMON] or terms
    forms = [best_form(con, t) for t in terms]
    # 库里根本没有的（0 条）排最后：查了也白查，但也不能直接扔——
    # 万一是新东西，让它排在后面自然被预算挤掉就行。
    return [t for t, n in sorted(forms, key=lambda x: (x[1] == 0, x[1], -len(x[0])))]


# 抽取器整理出来的「事」单独算一层，排在最前面。
# 2026-09-23：原话片段都归在 episodic，light 档只看第一层就停手，
# 抽出来的「用户是短发」（semantic）问「用户是短发吗」根本够不着。
DISTILLED = "distilled"


def probe(con, term: str, *, layer: str, limit: int, subject: str | None = None) -> list[dict]:
    """查一个词；长词查不到就缩短再试一次。

    keywords() 最长给到 8 个字，像「工作记忆层做完」这种整块下去 FTS 一条都不中，
    而「工作记忆」是中的。所以长词失手后按 4 字、3 字各退一步——
    退到 3 字为止，再短就是噪音了。
    """
    if layer == DISTILLED:
        return search(con, term, limit=limit, subject=subject, extracted_only=True)
    hits = search(con, term, limit=limit, subject=subject, fact_types=[layer])
    if hits or len(term) <= 3:
        return hits
    out, seen = [], set()
    for t, _n in usable_forms(con, term):     # 几个写法一起查，稀有的在前
        for h in search(con, t, limit=limit, subject=subject, fact_types=[layer]):
            if h["ext_id"] not in seen:
                seen.add(h["ext_id"])
                out.append(h)
    return out[: limit * 2]


def retrieve(con, text: str, *, who: str | None = None,
             per_layer: int = 6, hard_cap: int = 12) -> Result:
    """完整一次召回：该不该查 → 查哪一类 → 够不够 → 不够再升级。"""
    d: Decision = route(text, who=who)
    subject = who if who in persona.ACTORS else None
    goal_rel, goal_why = working.relevance(con, text)

    if not d.should_recall:
        # 纯规则说不查，但有两票可以把它翻过来。**豁免和「本来就知道」永远压过这两票。**
        vetoed = d.reason.startswith("软话") or d.reason.startswith("这个我本来")
        vote = None

        # ── 一票：稀有词唤醒（实测一句提到旧事的话没触发检索）
        #
        # 那句话被判成「日常闲聊，不值得翻库」，因为 FACTUAL_CUE 是一张**手写的专名
        # 白名单**（科目甲／某大学／小镇／某游戏……），「某桌游」不在上面。
        # 白名单永远追不上她会聊到的东西，这条路本身就是错的。
        #
        # 改成问库：这个词在库里出现在几条记录里？
        #   0 条      → 库里没有，查了也没有
        #   1~60 条   → 具体的旧事，**该自己想起来**
        #   60 条以上 → 满库都是的词，不算线索
        # 她要的「聊到就自己去找，不用等我问」就是这一条。
        if not vetoed:
            rare = [t for t in keywords(text)
                    if not CHATTER.search(t)
                    and RARE_MIN <= best_form(con, t)[1] <= RARE_MAX]
            if rare:
                vote = Decision(True, keywords(text), who, limit=3, depth="light",
                                reason=f"聊到了具体的旧事（{'、'.join(rare)}），自己翻一眼")

        # ── 另一票：工作记忆。「工作记忆层做完了吗」按关键词是闲聊，按在办任务最该查。
        if vote is None and not vetoed and goal_rel >= GOAL_STRONG:
            vote = Decision(True, keywords(text), who, limit=4, depth="deep",
                            reason=f"跟手上的事有关（{goal_why}），工作记忆要求查")

        if vote is None:
            return Result(False, depth="skip", goal_rel=goal_rel, goal_why=goal_why,
                          reason=d.reason, task_type="不查")
        d = vote

    base_terms = d.queries or keywords(text)
    if flags.on("METAMEM"):                # 同义写法补词（生日↔某年某月某日生）；跟元记忆的「问谁的事」一起开
        import router as _r
        # 必须在 by_rarity 之前补：light 档只取前两个词，补在最后会被截掉；补进去后按稀有度排，
        # 「日生」这种库里很少的词自然排前面（2026-10-29 大样本实测发现：补在最后时 light 档根本用不上）。
        base_terms = _r.expand_aliases(text, list(base_terms))
    terms = by_rarity(con, base_terms)
    if d.depth == "light":
        terms = terms[:LIGHT_TERMS]      # light 档仍只拿两个词，但拿的是最稀有的两个
    task, order = classify_task(text)

    # ── 工作记忆参与分发（GLM 助手那份第三节的「按目标、任务」那两条）
    # 抄 CraniMEM 门控第一关的思路：先看这句话跟手上的事搭不搭。
    if goal_rel >= GOAL_STRONG and goal_why:
        # 正在办的事被问到 → 先看「怎么做」那一层；问的是目标 → 先看已经定下来的结论。
        lead = "procedural" if goal_why.startswith("在办") else "semantic"
        order = (lead,) + tuple(x for x in order if x != lead)

    # 先看整理过的事，再按原来的顺序看原话
    order = (DISTILLED,) + tuple(order)
    # light 档只看前两层（整理过的事 + 原来的第一层），不升级——闲聊里顺口提到的事，瞄一眼就够
    if d.depth == "light":
        order = order[:2]
        per_layer = min(per_layer, 3)

    res = Result(True, depth=d.depth, goal_rel=goal_rel, goal_why=goal_why,
                 terms=terms, task_type=task, order=order, reason=d.reason)
    # 2026-10-28（任务二·步骤3）：默认走混合召回（FTS + 向量 → RRF 合并 → hook 里精排）。
    # 上面的 L1 分类 / order 只剩「header 里显示哪类问题」的用途；
    # 旧的六类分层 + 逐层升级循环原样保留在下面，ENGRAM_LAYERED_RETRIEVE=1 回滚。
    if not flags.on("LAYERED_RETRIEVE"):
        import hybrid
        hint = None
        if who is None and flags.on("CONTEXT_ROUTE"):      # 按上下文分发（按人物）：没人显式指定 who 时才推
            try:
                import router as _router
                hint = _router.infer_who(text)
            except Exception:                              # noqa: BLE001
                hint = None
        res = hybrid.run(con, text, res, terms=terms, light=(d.depth == "light"),
                         goal_rel=goal_rel, goal_why=goal_why, who_hint=hint)
        res.who_hint = hint
        if flags.on("METAMEM"):
            # 混合召回不走旧分层循环，metamem 的证据体检（打架/作废）要在这里补上；
            # 「够不够」那半（FOK/TOT）已知循环论证，不在这里用。
            import metamem
            # 问的是谁的事，就把那个人的事排前面（问「我生日」不拿助理的生日顶上）；只换顺序，不丢条目
            metamem.annotate_actors(con, res.hits) if metamem.asked_subject(text) else None
            res.hits = metamem.prefer_subject(text, res.hits)
            # lexical=False：字面「还没做/做了」那条在真问句上 4 次报警 3 次是误报
            # （「Lumen」「打架」这种满库都是的词），混合召回只用 fact_key 那条。
            ev = metamem.check_evidence(con, res.hits, terms=terms, lexical=False)
            if ev.conflicts:
                res.conflicts = tuple(dict.fromkeys(res.conflicts + ev.conflicts))
        if subject:                    # who＝只看谁的事实（MCP/CLI 文档一直这么写；以前混合召回里丢了）
            res.hits = [h for h in res.hits if h.get("subject") == subject]
        return res

    seen: dict[str, dict] = {}

    empty_streak = 0
    for layer in order:
        got: list[dict] = []
        for rank, t in enumerate(terms):
            for h in probe(con, t, layer=layer, limit=per_layer, subject=subject):
                if h["ext_id"] not in seen:
                    h["term_rank"] = rank      # 0 = 最稀有的那个词找回来的
                    seen[h["ext_id"]] = h
                    got.append(h)
        # 传原始问句而不是检索词：期待类型是从疑问词里读出来的，
        # 而检索词命中是必然的，拿它判够不够就是循环论证（见 metamem.py 文件头）。
        # 带证据体检的判据（GPT 拧的第三颗螺丝）：字段在不等于可信，
        # 二手来源要打折、已作废的不算、自己打架的一律判不够。
        #
        # 2026-10-28（任务二·步骤2）：「要不要翻」已经有两层（router 规则 + rerank 模型精排）在管，
        # metamem 的 FOK/TOT 是第三层，而且判的是「检索词命中」够不够——循环论证，
        # 还出过「问颜色判够了却一条颜色都没答上」的缺陷。
        # 默认关：不再在每层之后判够不够，交给精排挑。开 ENGRAM_METAMEM=1 可恢复旧行为。
        # 软话不翻旧账的豁免在 router.route（上面 d.reason 里的「软话」），与 metamem 无关，不受影响。
        if flags.on("METAMEM"):
            import metamem
            j, ev = metamem.judge_with_evidence(con, text, got, terms=terms)
            if ev.conflicts:
                res.conflicts = tuple(dict.fromkeys(res.conflicts + ev.conflicts))
            res.steps.append(Step(layer, len(got), j.fok, j.tot, j.enough, j.reason,
                                  j.expects, j.answered))
            enough = j.enough
        else:
            res.steps.append(Step(layer, len(got), 0.0, False, False,
                                  "不判够不够（metamem 已摘，交给精排）"))
            enough = False
        res.hits.extend(got)
        if enough:
            break
        if flags.on("METAMEM"):
            res.escalations += 1
        empty_streak = empty_streak + 1 if not got else 0
        if empty_streak >= MAX_EMPTY:
            res.steps.append(Step("停手", 0, 0.0, False, False,
                                  f"连着 {MAX_EMPTY} 层空手，不再往下翻"))
            break
        if len(res.hits) >= hard_cap:
            break

    if d.depth == "light":
        res.escalations = 0      # light 本来就不升级，别记成升级了一次
    # ── 六维加权里的「当前任务相关性」：跟手上这件事对得上的往前排。
    # 只动排序，不动存在——这是 store.py 第一条铁律。
    if goal_rel >= GOAL_STRONG and goal_why:
        anchor = goal_why.split("「", 1)[-1].rstrip("」")
        for h in res.hits:
            h["goal_boost"] = round(
                GOAL_WEIGHT * working._overlap(h.get("content") or "", anchor), 4)
    # 排序先看「是哪个词找回来的」，再看激活度。
    #
    # 起因：问「那次公园挺好看的」，排第一的是一份跟公园无关的
    # 完工报告——因为最后只按激活度排，一条高激活度但离题的记录能盖过正相关的。
    # 检索词已经按稀有度排过了（越少见越值钱），那么**最稀有那个词找回来的东西
    # 就该排在前面**，激活度只在同一档里比。
    res.hits = sorted(
        res.hits,
        key=lambda h: (h.get("term_rank", 9),
                       -(float(h.get("score") or 0) + float(h.get("goal_boost") or 0)))
    )[:hard_cap]
    return res


def log_recall(con, text: str, r: Result, *, decision: str | None = None,
               reason: str | None = None,
               selected: "set[str] | tuple[str, ...] | None" = None) -> int | None:
    """把一次召回记进遥测表。返回 run_id。

    调用方自己说最终决定是什么——因为「库里查到了」和「真的注入了上下文」是两回事。
    钩子那条路上还有几道闸（眼前就有、命中为空），查到了也可能一个字都不注入；
    那种情况必须记成 skip，不能混进成功召回里，否则统计出来的召回率是假的。

    selected 给的是真正注入的 ext_id 集合。不给就退回「前三条」这个旧近似。
    判不出来的不填（比如「这条模型到底用没用」），别假装知道——这条规矩是 GPT 定的。
    """
    import json

    from telemetry import Run

    if decision is None:
        decision = "recall" if r.should_recall else "skip"
    if reason is None:
        reason = r.reason
    with Run(con, query=text, decision=decision, reason=reason,
             terms=r.terms, mode=r.mode) as run:
        for i, h in enumerate(r.hits, 1):
            fid = con.execute("SELECT id FROM facts WHERE ext_id=?",
                              (h["ext_id"],)).fetchone()
            if not fid:
                continue
            run.add_hit(fact_id=fid[0], rank=i, final_score=float(h.get("score") or 0),
                        salience_score=float(h.get("score") or 0),
                        layer=h.get("layer"), memory_type=h.get("fact_type"),
                        snippet_chars=len(h.get("content") or ""),
                        selected=(h["ext_id"] in selected) if selected is not None
                        else i <= 3)
    if run.run_id:
        for i, st in enumerate(r.steps, 1):
            con.execute(
                """INSERT OR IGNORE INTO recall_paths
                     (run_id, step_no, task_type, layer, hits, fok, tot, enough,
                      expects, answered, reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (run.run_id, i, r.task_type, st.layer, st.hits, st.fok,
                 int(st.tot), int(st.enough),
                 json.dumps(list(st.expects), ensure_ascii=False),
                 json.dumps(list(st.answered), ensure_ascii=False), st.reason))
        con.commit()
    return run.run_id


def retrieve_logged(con, text: str, *, who: str | None = None, **kw) -> Result:
    """查一次并记一次账。给 CLI 和 MCP 用——它们查完就直接把结果交出去，
    不像钩子那样后面还有闸，所以「查到了」就等于「用上了」。
    """
    r = retrieve(con, text, who=who, **kw)
    log_recall(con, text, r)
    return r


def explain(r: Result) -> str:
    if not r.should_recall:
        return f"不查：{r.reason}"
    head = (f"[{r.task_type}·{r.depth}] 词{r.terms} 顺序"
            f"{'→'.join(TYPE_CN.get(x, x) for x in r.order)}")
    if r.goal_rel:
        head += f"\n  手上的事：{r.goal_why}（相关度 {r.goal_rel}）"
    def line(i: int, s: Step) -> str:
        if s.layer == "停手":
            return f"  {i + 1}. 停手：{s.reason}"
        if s.layer == "hybrid":
            return f"  {i + 1}. 混合召回：{s.reason}"
        return (f"  {i + 1}. {TYPE_CN.get(s.layer, s.layer)}层：命中 {s.hits} 条，"
                f"信心 {s.fok:.2f}{'，话到嘴边' if s.tot else ''} → "
                f"{'停手' if s.enough else '升级'}（{s.reason}）")

    body = "\n".join(line(i, st) for i, st in enumerate(r.steps))
    tail = (f"  共读 {len(r.steps)} 层，升级 {r.escalations} 次，"
            f"返回 {len(r.hits)} 条")
    if r.conflicts:
        tail += "\n  ⚠️ 打架：" + "；".join(r.conflicts) + "\n     别挑一条顺嘴的说，" \
                "要么按时间取新的那条并说清，要么直接问她。"
    return f"{head}\n{body}\n{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--trace", action="store_true", help="连命中的内容一起打出来")
    ap.add_argument("--log", action="store_true", help="把这次召回记进遥测")
    args = ap.parse_args()

    samples = [
        "亲爱的👀", "想你了",
        "上次那个海边是哪年去的",
        "我喜欢什么颜色",
        "期末数据库占多少分",
        "上次那个通道掉线是怎么弄好的",
        "你还记得我跟你说过邻居把水槽搞堵了吗",
        "我说过发消息不许用什么格式",
    ]
    con = connect()
    runner = retrieve_logged if args.log else retrieve
    for text in args.text or samples:
        r = runner(con, text)
        print(f"\n{text!r}\n{explain(r)}")
        if args.trace:
            for h in r.hits[:6]:
                print(f"     · [{h['subject']}] {h['content'][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
