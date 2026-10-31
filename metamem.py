#!/usr/bin/env python3
"""Lumen Engram · 元记忆（知道感 / 话到嘴边）

抄 BMAM 的 `src/memory/metamemory.py` 与 `src/brain/prefrontal_controller.py`。
**抄的是思路，不是代码**——它那份是半桩的：`query_vector` 一直传 None（注释自己写着
「暂无向量」），相似度分数直接拿 importance 顶替，实体重合用 `query.lower().split()`
按空格切词，对中文等于没切。

这层只回答一个问题：**刚才那次检索，够不够？**
够就停手，不够就升级去查别的记忆层。有了它，路由就不必先猜「这句话该叫醒哪类记忆」——
猜需要目标／情绪／任务三个输入，那三样现在都还没有。

两个指标沿用 BMAM 的命名：
  FOK（feeling of knowing，知道感）—— 这次查到的东西，答不答得上问题。
  TOT（tip of the tongue，话到嘴边）—— 一堆沾边的，但没一条真答上。这种最该升级。

**这层只读不写，只影响要不要多查一层，不改任何记忆。**

---

## 为什么是「问句期待什么答案」，而不是相似度

这个判据换过两版，两版都因为同一个毛病废掉：**拿一个恒真的东西当证据。**

第一版直接用 `engram_activation` 的分数。那个分数无上界（重要度＋衰减＋钉住能到 1.23），
套进公式后每个查询的信心都顶到 1.00。更根本的错是概念错：
**激活度回答的是「这条容易不容易被想起来」，不是「这条答不答得上刚才那句问话」。**
拿排序先验当检索质量，结论只会是「库里有东西 → 我知道」。

第二版改成算覆盖率：问题的二元字组有多少落在命中内容里。看着合理，其实是循环论证——
**检索词本来就是拿去 FTS 匹配的，命中的条目必然含有它，覆盖率恒为 1。**
量的是「我搜到了我搜的东西」。

所以第三版换成**答案类型**：问句在要什么（年份？数字？人？地点？），命中里有没有那种东西。
「上次那个星光湾是哪年」检索词是「星光湾」，命中必然含「星光湾」，但里面**有没有年份**
是独立的、可能为假的——这才叫证据。问句没有明确期待时，才退回看激活度分布。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import persona  # noqa: E402
from common import gramset  # noqa: E402

FOK_ESCALATE = 0.40      # 低于这个信心就升级（BMAM 用的也是 0.4）
TOT_TOP_MAX = 0.55       # 最高分不到这个，才可能是「话到嘴边」
TOT_MIN_HITS = 3         # 得有好几条沾边的才叫话到嘴边；一条都没有那叫真不知道
DEPTH = 5                # 只看前几条：排在第 20 条的东西答上了不算数

# ── 问句期待什么类型的答案 ────────────────────────────────────────────────
ASK = {
    # 「哪年」跟「什么时候」不是一个精度：前者要年份，后者一个「那天下午」就答上了。
    # 分开之后才不会出现「问哪年、命中里有个『晚上』就算答上了」这种假信心。
    "year":   re.compile(r"(哪年|哪一年|那年|几几年|哪届)"),
    "date":   re.compile(r"(哪天|哪一天|什么时候|几号|几月|多久|哪个月|"
                         r"上次是.*时候|日期)"),
    "number": re.compile(r"(多少|几分|几个|几点|几次|几条|占比|百分|多大|几岁|"
                         r"多少钱|多少分)"),
    "person": re.compile(r"(是谁|谁说|谁的|谁做|哪个人|谁给)"),
    "place":  re.compile(r"(在哪|哪里|哪儿|什么地方|去了哪)"),
}

HAS = {
    "year":   re.compile(r"(\d{4}\s*年|\d{4}-\d{2}|去年|前年|今年|大前年|"
                         r"[初高][一二三]那年|\d{4}\s*届)"),
    "date":   re.compile(r"(\d{4}\s*[-年/]|\d{1,2}\s*[月日号]|\d{4}-\d{2}-\d{2}|"
                         r"[今昨前明后]天|上周|上个月|去年|前年|凌晨|早上|下午|晚上|半夜)"),
    "number": re.compile(r"(\d|[一二三四五六七八九十百千万两半]+(?=[个条分点次成％%元块])|%)"),
    "person": re.compile(persona.fill(r"(@@alt:USER,USER_ALIASES,CLAUDE,GPT,C@@") + "".join("|" + persona.alt(t) for t in persona.PERSON_SHORT) + r"|同事|老师|她|他|我)"),
    "place":  re.compile(r"(" + "".join(persona.alt(t) + "|" for t in persona.PLACE_TERMS) + r"酒店|车站|宿舍|教室|家里|学校)"),
}


# 记录本身带的时间戳会冒充答案：内容长这样——
#   「助手（时间）：想好名字……」
# 里面那个 2026-08 是**这条记录写下来的时刻**，不是问题的答案。
# 不剥掉的话，问「哪年」时几乎每条都算答上了，又是一次「拿恒真的东西当证据」。
RE_STAMP = re.compile(r"[（(]\s*\d{4}-\d{2}-\d{2}[^）)]*[）)]")
RE_SPEAKER = re.compile(r"^[^：:\n]{1,12}[：:]")


def body(content: str) -> str:
    """剥掉说话人前缀和记录自带的时间戳，只留真正的话。"""
    t = RE_STAMP.sub("", content or "")
    return RE_SPEAKER.sub("", t, count=1)


@dataclass(frozen=True)
class Judgment:
    fok: float               # 0~1，知道感
    tot: bool                # 是否话到嘴边
    enough: bool             # 够不够，不够就该升级
    reason: str
    expects: tuple[str, ...] = ()   # 问句期待的答案类型
    answered: tuple[str, ...] = ()  # 命中里真出现了的类型
    top: float = 0.0
    spread: float = 0.0


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def expects(text: str) -> tuple[str, ...]:
    """这句问话在要什么类型的答案。要不出来就是空——空不等于「不用查」。"""
    return tuple(k for k, p in ASK.items() if p.search(text or ""))


def satisfied(hits: list[dict], kinds: tuple[str, ...],
              depth: int = DEPTH) -> tuple[tuple[str, ...], float]:
    """前 depth 条里，期待的类型有几种真出现了，以及命中条目的满足比例。"""
    if not kinds or not hits:
        return (), 0.0
    head = [body(h.get("content") or "") for h in hits[:depth]]
    got, hit_rows = [], 0
    for k in kinds:
        pat = HAS[k]
        if any(pat.search(c) for c in head):
            got.append(k)
    for c in head:
        if all(HAS[k].search(c) for k in kinds):
            hit_rows += 1
    return tuple(got), hit_rows / len(head)


def coverage(query: str, content: str) -> float:
    """问题的二元字组有多少落在这条内容里。0~1。

    ⚠️ 别拿它判「够不够」——检索词命中是必然的，见文件头第二版那段。
    这个函数留着只用于**没有明确期待**时衡量离题程度，以及给人看。
    """
    qg = gramset(query)
    if not qg:
        q = (query or "").strip()
        return 1.0 if q and q in (content or "") else 0.0
    return len(qg & gramset(content or "")) / len(qg)


# ── 主体归属（2026-10-29）：问的是谁的事，命中的是不是那个人的事 ─────────────────────
# 起因：问「我哪天生日」，命中里排前面的是助理自己的生日，「日期」那类证据被它满足，判成「够了」。
# 句里有没有日期是独立证据，但**这个日期是谁的**也得对上，不对就是没答上。
# 规矩：问句里的「我/她/用户称呼」＝用户；「你/助理称呼」＝助理。两边都出现、认不准就不过滤（不瞎猜）。
#       命中的归属先看 subject 字段，其次看挂的证据是谁说的（echo.evidence_actors），
#       最后才看正文开头的主语。记在用户名下、却只有助理在说的不算用户的事（回声）。
_COG = r"(?:还)?(?:记得|记不记得|知道|知不知道|晓得|记住|认得|清楚|说|讲|告诉|想|查|找|看|猜|能|可以|会)"
_FRAME_ME = re.compile(r"(?:告诉|给|帮|让|替|跟|对|问|陪)我")


def _terms(*groups) -> tuple[str, ...]:
    return tuple(t for g in groups for t in g if t)


def person_markers(text: str) -> tuple[bool, bool]:
    """问句里有没有「她/我/用户称呼」（第一个）、有没有「你/助理称呼」（第二个）。提问框架里的人称（你记得、告诉我）已剥掉。"""
    claude_terms = _terms((persona.N_CLAUDE,), persona.CLAUDE_ALIASES)
    user_terms = _terms((persona.N_USER,), persona.USER_ALIASES)
    # 先剥掉「你记得／告诉我／帮我查」这类提问框架里的人称，剩下的才是被问的那件事的主人
    frame = re.compile("(?:" + persona.alt("你", *claude_terms) + ")(?=" + _COG + ")")
    core = _FRAME_ME.sub("", frame.sub("", text or ""))
    user = bool(re.search("|".join(["我", "咱", "她", *[re.escape(t) for t in user_terms]]), core))
    claude = bool(re.search("|".join(["你", *[re.escape(t) for t in claude_terms]]), core))
    return user, claude


def asked_subject(text: str) -> str | None:
    """问句问的是谁的事：返回 persona.USER / persona.CLAUDE / None（认不准）。"""
    user, claude = person_markers(text)
    if user == claude:
        return None
    return persona.USER if user else persona.CLAUDE


def _lead_owner(content: str) -> str | None:
    """正文开头的主语是谁（只认明确的称呼）。"""
    t = body(content).lstrip()
    if t.startswith(_terms((persona.N_CLAUDE,), persona.CLAUDE_ALIASES)):
        return persona.CLAUDE
    if t.startswith(_terms((persona.N_USER,), persona.USER_ALIASES)):
        return persona.USER
    return None


# 「某某（时间）：」「我：」开头的是**说话人前缀**（谁说了这句话），不是这句话讲的是谁。
# 旧管线把说话人填进了 subject，于是「助理（时间）：28 号是〔她〕生日」被记在助理名下。
# 借鉴：Graphiti 把「说话人」和「事实涉及的实体」分开（先抽 speaker，再抽话里的实体）；
# mem0 新版给每条记忆单独写 attributed_to，而不是拿消息角色顶替。这里：带说话人前缀的记录，
# 不信 subject，改看正文里（剥掉前缀后）提到的是谁。
RE_SPEAKER_PREFIX = re.compile(r"^\s*[^：:\n（(]{1,12}\s*(?:[（(]\s*\d{4}-\d{2}-\d{2}[^）)]*[）)])?\s*[：:]")


def speaker_prefixed(content: str) -> bool:
    return bool(RE_SPEAKER_PREFIX.match(content or ""))


HEAD_CHARS = 24      # 只看正文开头这么多字：被点名当主语的人通常在开头；写在后面的多半只是话里顺带提到


def mentions(text: str, head: int | None = None) -> tuple[bool, bool]:
    """正文（可只取开头 head 字）里提到了她（第一个）/ 助理（第二个）。
    只认称呼，不认「我/你」（那是说话人视角，在记录里指谁不一定）。"""
    t = body(text)
    if head:
        t = t[:head]
    user_terms = _terms((persona.N_USER,), persona.USER_ALIASES)
    claude_terms = _terms((persona.N_CLAUDE,), persona.CLAUDE_ALIASES)
    user = bool(re.search("|".join(["她", *[re.escape(x) for x in user_terms]]), t))
    claude = bool(claude_terms) and bool(re.search("|".join(re.escape(x) for x in claude_terms), t))
    return user, claude


def belongs_to(h: dict, want: str) -> bool:
    """这条命中算不算 `want`（persona.USER / persona.CLAUDE）的事。归属查不出来的放行：不知道不等于不对。"""
    subj = h.get("subject")
    if subj == "us":
        return True                      # 两人共同的事，谁问都算
    if speaker_prefixed(h.get("content") or ""):
        mu, mc = mentions(h.get("content") or "", HEAD_CHARS)
        if mu != mc:                     # subject 是说话人顶上的，不可信；正文只讲一个人 → 算那个人的
            return mu if want == persona.USER else mc
    if subj and subj != want:
        return False
    lead = _lead_owner(h.get("content") or "")
    if lead not in (None, want):
        return False                     # 正文明说是另一方的事
    actors = h.get("_actors")            # judge_with_evidence 补上的：这条挂的证据是谁说的
    if subj == want and actors and want not in actors:
        return False                     # 记在她名下、却只有别人在说：回声
    return True


def for_subject(text: str, hits: list[dict]) -> tuple[list[dict], str | None]:
    """按问句问的主人筛命中。认不准主人时原样返回。"""
    want = asked_subject(text)
    if want is None:
        return hits, None
    return [h for h in hits if belongs_to(h, want)], want


def prefer_subject(text: str, hits: list[dict]) -> list[dict]:
    """混合召回用：问的是谁的事，就把那个人的事排前面（稳定排序，一条都不丢）。"""
    want = asked_subject(text)
    if want is None:
        return list(hits)
    good = [h for h in hits if belongs_to(h, want)]
    ids = {id(h) for h in good}
    return good + [h for h in hits if id(h) not in ids]


def annotate_actors(con, hits: list[dict]) -> None:
    """给每条命中补上 `_actors`：它的 support 证据都是谁说的（走说话人修正层）。只读。"""
    import echo
    for h in hits:
        if h.get("_actors") is not None or not h.get("ext_id"):
            continue
        rows = con.execute(
            "SELECT fe.evidence_id FROM fact_evidence fe JOIN facts f ON f.id=fe.fact_id"
            " WHERE f.ext_id=? AND fe.role='support'", (h["ext_id"],)).fetchall()
        h["_actors"] = echo.evidence_actors(con, [r[0] for r in rows]) or None


def judge(text: str, hits: list[dict], *, score_key: str = "score") -> Judgment:
    """给一次检索结果打分：够不够。

    `text` 要传**原始问句**（不是检索词）——期待类型是从问句的疑问词里读出来的。
    """
    if not hits:
        return Judgment(0.0, False, False, "一条都没查到", expects(text))

    # 先按问的是谁的事筛一遍：别人的生日不能当我的生日答上问题
    hits, owner = for_subject(text, hits)
    if owner and not hits:
        who = persona.N_USER if owner == persona.USER else persona.N_CLAUDE
        return Judgment(0.0, False, False, f"问的是{who}的事，命中的全是别人的", expects(text))

    want = expects(text)
    if want:
        got, ratio = satisfied(hits, want)
        # 期待的类型一样都没出现 → 信心归零，直接升级
        if not got:
            return Judgment(0.0, False, False,
                            "问的是" + "、".join(want) + "，命中里没有这类东西",
                            want, got, 0.0)
        base = len(got) / len(want)          # 期待被覆盖了几成
        fok = min(1.0, base * 0.7 + ratio * 0.3)
        tot = fok < TOT_TOP_MAX and len(hits) >= TOT_MIN_HITS
        if tot:
            return Judgment(fok, True, False,
                            "话到嘴边：沾边的不少，但" + "、".join(
                                k for k in want if k not in got) + "答不上",
                            want, got, fok)
        if fok < FOK_ESCALATE:
            return Judgment(fok, False, False, f"信心不够（{fok:.2f}）", want, got, fok)
        return Judgment(fok, False, True, f"够了（信心 {fok:.2f}）", want, got, fok)

    # 没有明确期待：退回看激活度分布。这条路信心天花板压到 0.8，
    # 因为「没期待」本身就意味着没有独立证据，不该给满分。
    scores = [min(1.0, float(h.get(score_key) or 0.0)) for h in hits[:DEPTH]]
    top, var = max(scores), _variance(scores)
    fok = min(0.8, top * 0.7 + (1.0 - min(var, 1.0)) * 0.3)
    tot = len(hits) >= TOT_MIN_HITS and top < TOT_TOP_MAX
    if tot:
        return Judgment(fok, True, False, "话到嘴边：一堆沾边的，没一条够强",
                        (), (), top, var)
    if fok < FOK_ESCALATE:
        return Judgment(fok, False, False, f"信心不够（{fok:.2f}）", (), (), top, var)
    return Judgment(fok, False, True, f"够了（信心 {fok:.2f}，无明确期待）",
                    (), (), top, var)


if __name__ == "__main__":
    from store.store import connect, search

    con = connect()
    cases = sys.argv[1:] or [
        "上次那个海边是哪年去的", "期末数据库占多少分",
        "上次装修赔了多少钱", "我喜欢什么颜色", "那场比赛在哪打的",
    ]
    for text in cases:
        from router import keywords
        terms = keywords(text)
        hits: list[dict] = []
        for t in terms:
            hits.extend(search(con, t, limit=6))
        j = judge(text, hits)
        print(f"{text}\n  词{terms} 命中{len(hits)}  期待{j.expects or '—'} "
              f"答上{j.answered or '—'}  信心{j.fok:.2f} → {j.reason}\n")


# ══════════════════════════════════════════════════════════════════════════
# 证据化充分性（sufficiency v0.2）
#
# app 里的 GPT 审第一版时拧的第三颗螺丝，原话：把「查到字段就算够」改成证据化充分性判断，
# 不然「会比乱查全库聪明，但偶尔仍会拿着一条过期摘要自信满满地胡说八道」。
#
# 上面那套只问了一句「命中里有没有年份」。字段在，不等于这条可信：
#   · 那个年份可能是二手转述（我复述她的话，再被抽成事实）
#   · 那条事实可能已经作废（valid_to 关掉了，只是历史还留着）
#   · 可能有另一条跟它直接打架（「某游戏还没打」和「某游戏打过了」库里都有）
#
# 这三样库里本来就有——来源等级、有效期、同源聚簇、冲突状态，
# 就是评审里被指出「超做了」的那一节。现在正好接上。
#
# **有冲突时一律判「不够」。** 宁可多翻一层、宁可跟她说「有两条打架」，
# 也不能挑一条顺嘴的说出去——那正是我最容易犯的错。
# ══════════════════════════════════════════════════════════════════════════

TIER_WEIGHT = {"primary": 1.0, "secondary": 0.6, "derived": 0.4}
STALE_DAYS = 180          # 语义／程序类事实过了这么久，提醒一句它可能过期了

# 没有 fact_key 也能看出打架的那种：同一批命中里，一边说没做一边说做了。
RE_NOT_YET = re.compile(r"(还没|没有打|没打|尚未|还欠着|没写|未完成|还剩)")
RE_DONE = re.compile(r"(已经|打过|做完|写完了|搞定|完工|办完|通了|上线)")
NEAR = 14        # 判定词得离话题词这么近才算是在说同一件事


def _says(content: str, topic: str, pat: re.Pattern) -> bool:
    """这条内容里，`pat` 是不是**贴着** topic 说的。

    只查「同一条里都出现」不够：问星光湾时，一条说「星光海滩和白沙滩还没走」，
    另一条说「日记写完了……星光海滩那串蓝脚印没能收成」，
    按「都出现」就成了一对冲突，其实一个说游戏没走、一个说日记写完了。
    所以要求判定词落在话题词左右 {NEAR} 个字以内。
    """
    text = content or ""
    for m in re.finditer(re.escape(topic), text):
        window = text[max(0, m.start() - NEAR): m.end() + NEAR]
        if pat.search(window):
            return True
    return False


@dataclass(frozen=True)
class Evidence:
    """一次召回里，答上问题那些条目的证据体检结果。"""
    checked: int = 0
    expired: int = 0            # 已作废的（valid_to 非空）
    secondhand: int = 0         # 二手／派生来源
    unverified: int = 0         # 原文没核对上
    conflicts: tuple[str, ...] = ()
    weight: float = 1.0         # 来源等级折出来的权重
    stale: bool = False

    @property
    def clean(self) -> bool:
        return not self.conflicts and not self.expired


def check_evidence(con, hits: list[dict], *, depth: int = DEPTH,
                   terms: list[str] | None = None,
                   lexical: bool = True) -> Evidence:
    """给前 depth 条做证据体检。只读库，不改任何东西。
    lexical=False：不做第二条「一边还没做一边做了」的字面判断（混合召回路径用，见下）。

    `terms` 是这次的检索词。判「一边说还没做一边说做了」必须落在**同一个话题**上，
    所以要拿检索词当锚。不给锚这一条就不判——第一版没给锚，
    结果问星光湾时把「某海岛第二天还没走」和「日记写完了」凑成一对冲突，
    两句话八竿子打不着。又是一次「没对齐主语就下结论」。
    """
    head = hits[:depth]
    if not head:
        return Evidence()

    ids = [h.get("ext_id") for h in head if h.get("ext_id")]
    if not ids:
        return Evidence(checked=len(head))
    marks = ",".join("?" * len(ids))

    rows = con.execute(f"""
        SELECT f.ext_id, f.subject, f.content, f.fact_type, f.valid_to, f.valid_from,
               julianday('now') - julianday(f.valid_from) AS age,
               MIN(COALESCE(t.tier,'primary')) AS tier,
               MIN(COALESCE(e.verification_status,'unverified')) AS ver,
               MAX(m.fact_key) AS fact_key, MAX(m.value) AS value,
               MAX(COALESCE(m.polarity,'pos')) AS polarity
        FROM facts f
        LEFT JOIN fact_evidence fe ON fe.fact_id = f.id
        LEFT JOIN evidence e ON e.id = fe.evidence_id
        LEFT JOIN evidence_tier t ON t.evidence_id = e.id
        LEFT JOIN fact_key_map m ON m.fact_id = f.id
        WHERE f.ext_id IN ({marks})
        GROUP BY f.ext_id
    """, ids).fetchall()

    expired = sum(1 for r in rows if r["valid_to"])
    secondhand = sum(1 for r in rows if (r["tier"] or "primary") != "primary")
    unverified = sum(1 for r in rows if (r["ver"] or "") != "verified")
    weights = [TIER_WEIGHT.get(r["tier"] or "primary", 0.5) for r in rows]
    weight = sum(weights) / len(weights) if weights else 1.0
    stale = any((r["age"] or 0) > STALE_DAYS
                and r["fact_type"] in ("semantic", "procedural") for r in rows)

    conflicts: list[str] = []

    # 一、同一个 fact_key 出现两个不同的值 → 打架
    #    只对 single 的 key 判：multi 的 key（备考科目、爱吃的东西……）本来就能并存，
    #    301 和 302 同时出现不是打架（2026-10-28 在副本上跑真问句时发现的误报）。
    #    肯定/否定也要落在**同一个值**上才算：「爱吃苹果」和「不爱吃梨」不冲突。
    from keys import by_name
    #    还要**同一个主体**：她的生日和助理的生日同在一个 key 下是两个人的事，不是打架
    #    （2026-10-29 主体归属修正）。
    by_key: dict[tuple[str, str], set[str]] = {}
    pol_by_kv: dict[tuple[str, str, str], set[str]] = {}
    for r in rows:
        if r["fact_key"] and r["value"]:
            k = by_name(r["fact_key"])
            if k is not None and k.cardinality == "single":
                by_key.setdefault((r["subject"], r["fact_key"]), set()).add(r["value"])
            pol_by_kv.setdefault((r["subject"], r["fact_key"], r["value"]), set()).add(r["polarity"] or "pos")
    for (_s, k), vals in by_key.items():
        if len(vals) > 1:
            conflicts.append(f"{k} 有两个值：{ '／'.join(sorted(vals)) }")
    for (_s, k, v), pols in pol_by_kv.items():
        if len(pols) > 1:
            conflicts.append(f"{k}={v} 一条肯定一条否定")

    # 二、没有 fact_key 也能看出来的：**同一个话题上**一边说还没做、一边说已经做了
    for t in ((terms or []) if lexical else []):
        if len(t) < 2:
            continue
        not_yet = [r for r in rows if _says(r["content"], t, RE_NOT_YET)]
        done = [r for r in rows if _says(r["content"], t, RE_DONE)]
        if not_yet and done:
            conflicts.append(f"「{t}」上一边说还没做、一边说已经做了"
                             f"（按时间新的那条为准，别挑顺嘴的）")
            break

    return Evidence(len(head), expired, secondhand, unverified,
                    tuple(conflicts), round(weight, 3), stale)


def judge_with_evidence(con, text: str, hits: list[dict],
                        *, score_key: str = "score",
                        terms: list[str] | None = None) -> tuple[Judgment, Evidence]:
    """先按上面那套判够不够，再拿证据体检把结论往下压。

    压不压得动只有三种情况：
      · 有冲突 → 一律不够，去下一层，并且要跟她说清有两条打架
      · 全是已作废的 → 不够
      · 来源不硬 → 信心按来源等级打折（二手 0.6、派生 0.4）
    """
    if hits and asked_subject(text):
        annotate_actors(con, hits)
    j = judge(text, hits, score_key=score_key)
    if not hits:
        return j, Evidence()

    ev = check_evidence(con, hits, terms=terms)

    if ev.conflicts:
        return Judgment(min(j.fok, 0.35), j.tot, False,
                        "查到的东西自己打架：" + "；".join(ev.conflicts),
                        j.expects, j.answered, j.top, j.spread), ev

    if ev.checked and ev.expired == ev.checked:
        return Judgment(0.0, j.tot, False, "答上的那几条全是已作废的",
                        j.expects, j.answered, j.top, j.spread), ev

    if ev.weight < 1.0:
        fok = round(j.fok * ev.weight, 3)
        enough = fok >= FOK_ESCALATE and j.enough
        tail = f"（来源不是一手，信心打到 {ev.weight} 折）"
        return Judgment(fok, j.tot, enough, j.reason + tail,
                        j.expects, j.answered, j.top, j.spread), ev

    return j, ev
