"""「值不值得记」闸（任务三后续，2026-10-28；开关 ENGRAM_VALUE_GATE，默认关；已随 2026-10-28 部署上正式库，仍默认关）。

防回声（echo.py）管「谁说的归谁」；这一层管「值不值得记」。思路借自几个开源记忆项目
（只借想法，代码是自己写的；它们是 MIT/Apache 协议）：
  · Palinode —— 先用确定性规则筛掉心跳/应答/工具输出，再让模型抽；抽取时「宁可拒绝大多数」
  · rho —— 封闭的分类枚举 + 每条要带逐字引文和锚点，没有引文的一律不收
  · memware —— 按说话人的角色限定能产出哪几类事（AI 的话只能产出它自己的承诺/自我纠正）
  · Hindsight —— 「30 天后还重要吗」的自问；事实只记行为/决定，不记过程流水
  · mem0 —— 抽取后再做一遍确定性的后检（引文是原文子串、内容有依据、主语不是代词）
全是确定性规则，**不调模型**；模型只在抽取提示词里被要求照这套枚举输出。

四块：
  prefilter(text)                证据本身不值得进抽取：心跳、纯应答、工具输出、JSON 块、Error 行、太短
  role_check(...)                角色规则（在防回声之上）：谁的话能产出哪几类、us 要她随后的确认
  postcheck(...)                 后检：类别、引文、依据、主语、过程/转述/指标/状态/第三方评价/总结/推荐/鼓励/设定
  confidence_for(...)            置信度不再一律 0.9
每一次拒绝都按原因计数（REJECTS）。
"""
from __future__ import annotations

import collections
import json
import re
import unicodedata

import persona

REJECTS: collections.Counter = collections.Counter()      # 本进程的拒绝计数：原因 → 条数（只计数，不存原文）

# ── 封闭枚举 ─────────────────────────────────────────────────────────────
USER_CATEGORIES = ("preference", "life_change", "correction", "decision")   # 只能来自她自己的原话
AI_CATEGORIES = ("commitment", "self_correction")                          # AI 对她的承诺 / AI 自己纠正自己
US_CATEGORIES = ("agreement",)                                             # 两人之间的约定（要她的确认）
CATEGORIES = USER_CATEGORIES + AI_CATEGORIES + US_CATEGORIES
ROLE_ALLOWED = {
    persona.USER: set(USER_CATEGORIES),
    "claude": set(AI_CATEGORIES),
    "gpt": set(AI_CATEGORIES),
    persona.AGENT_C: set(AI_CATEGORIES),
}
MAX_PER_SLICE = 3
MAX_PER_SLICE_PRIORITY = 5       # 超过 3 条后，只有「高置信的她的决定/规矩/助手的承诺」还能再进，总共不超过 5 条
# 提示词输出里没有 type（闸的 PROMPT_GATE 只要 category）；from_evidence.check 需要 type，缺了就按类别补
TYPE_FOR_CATEGORY = {"preference": "preference", "life_change": "episodic", "correction": "procedural",
                     "decision": "procedural", "commitment": "procedural", "self_correction": "procedural",
                     "agreement": "procedural"}


def enabled() -> bool:
    try:
        import flags
        return flags.on("VALUE_GATE")
    except Exception:                                        # noqa: BLE001
        return False


def reject(reason: str) -> str:
    REJECTS[reason] += 1
    return reason


# ── 引文规范化 / 长度（试跑发现：模型把直引号改弯引号、全角标点改半角、引整段） ────────
QUOTE_MAX = 60            # 引文最长（规范化后）。更长的取前 QUOTE_MAX 字（尽量在句读处断开），仍是原话的逐字子串
QUOTE_REJECT = 150        # 超过这么长基本是整段/跨多行照抄，不当引文
QUOTE_MIN_TRUNC = 4


def norm_quote(s: str) -> str:
    """引文比对用的规范化：NFKC（全角→半角、…→...）、转小写、去掉所有标点/引号/空白/表情（只留字母数字汉字）。
    只用来「找得到」，不用来改写入库的引文原文。"""
    t = unicodedata.normalize("NFKC", s or "").casefold()
    return re.sub(r"[\W_]+", "", t)


def quote_in(quote: str, text: str) -> bool:
    nq = norm_quote(quote)
    return bool(nq) and nq in norm_quote(text)


def cap_quote(quote: str, limit: int = QUOTE_MAX) -> str:
    """超长引文截短：在 limit 内最后一个句读/空格处断开（至少保留 20 字），没有就硬切。结果仍是原引文的开头一段，所以仍是原话子串。"""
    q = (quote or "").strip()
    if len(q) <= limit:
        return q
    cut = q[:limit]
    pos = max(cut.rfind(c) for c in "。！？!?；;，,… 、\n")
    if pos >= 20:
        cut = cut[:pos + 1]
    return cut.strip()


# ── 英文「思考文本」：助手证据里混进去的内心独白（只当上下文，不当引文来源） ──────────────
_THINK_START = re.compile(
    r"^(?:let me|let's|let us|she'?s|she is|she was|he'?s|the user|user'?s|the human|i should|i need to|i'll|i will|i think|"
    r"i want to|i'm going to|i can|i could|i notice|i see|now i|now,|ok(?:ay)?[,.]|actually,|hmm|wait[,.]|so (?:the|she|he|i)\b|"
    r"looking at|her message|his message|this is (?:a|the)|they'?re|it seems|it looks|based on|first,|next,|"
    r"the (?:message|conversation|context|reply|response|file|tool)|my (?:response|reply|plan|goal))\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _cjk_ratio(t: str) -> float:
    letters = [c for c in t if c.isalpha()]
    return (sum(1 for c in letters if "\u4e00" <= c <= "\u9fff") / len(letters)) if letters else 0.0


def thinking_sentences(text: str) -> list[str]:
    """文本里（几乎纯英文、以「Let me / She's asking / The user / I should …」开头的）思考句。"""
    body = _WRAP.sub("", (text or "").strip())
    out = []
    for sent in _SENT_SPLIT.split(body):
        t = sent.strip()
        if len(t) >= 12 and _cjk_ratio(t) < 0.15 and _THINK_START.match(t):
            out.append(t)
    return out


def is_thinking_text(text: str) -> bool:
    """整条证据主要是英文思考文本：第一句就是思考句，且思考句占一半以上的字数。"""
    body = _WRAP.sub("", (text or "").strip())
    th = thinking_sentences(body)
    if not th:
        return False
    first = _SENT_SPLIT.split(body)[0].strip()
    return (first in th) and sum(len(x) for x in th) >= 0.85 * len(re.sub(r"\s+", " ", body).strip())


def quote_in_thinking(text: str, quote: str) -> bool:
    """引文落在英文思考句里（混合行里思考句 + 中文回复：中文那部分仍可当引文）。"""
    nq = norm_quote(quote)
    return bool(nq) and any(nq in norm_quote(x) or norm_quote(x) in nq and len(norm_quote(x)) >= 12
                            for x in thinking_sentences(text))


# ── 1. 确定性预过滤（零 LLM） ───────────────────────────────────────────
_ACK_CHARS = set("嗯哦噢啊哈呵嘿嘻呜好的行吧可以收到知道了解明白懂谢晚安早在吗是对没问题ok")
_ACK_WORDS = {"ok", "okay", "yes", "no", "thx", "thanks", "yep", "nope", "lol"}


class _AckMatcher:
    """纯应答：全由「嗯/好的/收到/知道了/谢谢/哈哈/表情」这类字组成，且很短。"""
    def match(self, t: str) -> bool:
        core = re.sub(r"[\s\W_]+", "", (t or "").lower())
        if not core:
            return bool(re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", t or ""))
        emoji_only = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\s\W_]+", "", (t or "").lower())
        if not emoji_only:
            return True
        return core in _ACK_WORDS or (len(core) <= 8 and all(c in _ACK_CHARS for c in core))


_ACK = _AckMatcher()
_HEARTBEAT = re.compile(r"(?i)^\s*(?:heartbeat(?:_ok)?|ping|pong|keep-?alive|hb|alive|ok\s*\(heartbeat\)|【?(?:心跳|定时唤醒|唤醒)】?(?:检查|任务)?)\s*[:：]?\s*$")
_TOOL = re.compile(r"(?i)^\s*(?:<tool_result|<function_results|\[tool[ _]?(?:result|output)|tool[ _]?(?:result|output)\s*[:：]|stdout\s*[:：]|stderr\s*[:：]|exit code\s*[:：]?\s*\d+|\$ \S+|>>> )")
_ERRLINE = re.compile(r"(?im)^\s*(?:error|traceback|exception|fatal|warn(?:ing)?|errno)\b.{0,}|^\s*\d{4}-\d\d-\d\d[ T][\d:.,]+\s*(?:\[?(?:ERROR|WARN|FATAL)\]?)")
MIN_LEN = 4                                               # 去掉空白和标点后少于这么多字就当没内容


def _json_blob(t: str) -> bool:
    s = t.strip()
    if len(s) >= 2 and s[0] in "{[" and s[-1] in "}]":
        try:
            json.loads(s)
            return True
        except ValueError:
            pass
        punct = sum(1 for c in s if c in '{}[]":,')
        return punct / len(s) > 0.18
    return False


_WRAP = re.compile(persona.fill(r"^(?:@@alt:USER,CLAUDE,GPT,C@@|我)（[^）]{0,30}）[：:]\s*"))


def prefilter(text: str, actor: str | None = None) -> str | None:
    """证据文本不值得抽取就返回原因，值得返回 None。（先剥掉「用户（时间）：」这种渲染前缀）
    actor 给了是 claude（或没给、渲染前缀是「我/助手」）时，主要是英文思考文本的返回 pre:thinking-text（留作上下文，不能当引文来源）。"""
    t = _WRAP.sub("", (text or "").strip())
    core = re.sub(r"[\s\W_]+", "", t)
    _claude_side = actor == "claude" or (actor is None and re.match(r"\s*(?:我|%s)（" % persona.alt(persona.N_CLAUDE), text or "") is not None)
    if _ACK.match(t):
        return "pre:ack"
    if _HEARTBEAT.match(t):
        return "pre:heartbeat"
    if len(core) < MIN_LEN:
        return "pre:too-short"
    if _TOOL.match(t):
        return "pre:tool-output"
    if _json_blob(t):
        return "pre:json-blob"
    lines = [l for l in t.splitlines() if l.strip()]
    if lines and _ERRLINE.match(lines[0]) and sum(1 for l in lines if _ERRLINE.match(l)) >= max(1, len(lines) // 2):
        return "pre:error-log"
    if _claude_side and is_thinking_text(t):
        return "pre:thinking-text"
    return None


HARD_PRE = ("pre:heartbeat", "pre:tool-output", "pre:json-blob", "pre:error-log")   # 这几类整条不送模型；应答/太短的留着当上下文


# ── 2. 角色规则（叠在防回声之上） ───────────────────────────────────────
_PRON = re.compile(r"^(?:他们?|她们?|它们?|我们|某人|有人|对方|那个人|这个人|那位|这位|大家|别人|某个|某位|其他人)")
_VAGUE_SUBJ = {"", "他", "她", "它", "我们", "某人", "有人", "对方", "大家", "别人", "someone", "somebody", "they", "he", "she", "it"}


def role_check(subject: str, category: str | None, actors: set, *, confirmed_by_user_after: bool = True,
               us_reject: str | None = None) -> str | None:
    """actors：支持证据的说话人集合。返回拒绝原因或 None。
    · claude 一方的话：只能产出 subject=claude 且类别是 commitment / self_correction
    · us：两方各自在窗口里说过话——(a) 助手提、她随后确认（confirmed_by_user_after）；(b) 她先提要求/定规矩、助手随后明确答应
      （调用方用 us_request_then_accept 判过，通过就也传 confirmed_by_user_after=True；没通过传具体原因 us_reject），类别 agreement
    · 她的偏好、生活变化、纠正、决定：只能来自她自己的原话"""
    actors = {a for a in actors if a}
    if category == "agreement" and subject != "us":
        return "role:agreement-not-us"                   # 约定只能是两个人之间的（us）；gpt/GLM 助手/单方落回的 agreement 一律不放行
    if subject == "us":
        if not (persona.USER in actors and len(actors) >= 2):
            return "role:us-one-sided"
        if not confirmed_by_user_after:
            return us_reject or "role:us-no-user-confirmation"
        if category not in (None, "agreement"):
            return "role:us-category"
        if category == "agreement" and "claude" not in actors:
            return "role:agreement-needs-claude"         # 约定＝用户和助手之间；只有她和 gpt/GLM 助手的话撑不起来
        return None
    if actors and subject in persona.ACTORS and subject not in actors:
        if actors == {"claude"}:
            return "role:claude-only-subject-mismatch"
        return "role:speaker-subject-mismatch"
    if actors == {"claude"} or (actors and persona.USER not in actors and subject == "claude"):
        if subject != "claude":
            return "role:claude-only-subject-mismatch"
        if category is not None and category not in AI_CATEGORIES:
            return "role:claude-only-category"
    if category is not None and category in USER_CATEGORIES and persona.USER not in actors:
        return "role:user-category-without-user-words"
    if category is not None and actors and subject in ROLE_ALLOWED:
        spk = subject
        if category not in ROLE_ALLOWED.get(spk, set()) and category not in US_CATEGORIES:
            return "role:category-not-allowed-for-speaker"
    return None


# ── 2b. us 的两种成立方式（2026-10-28 第四轮，用户同意放宽）──────────────────────────────
# (a) 助手提、她随后确认（原规则）；(b) 她先提要求/定规矩、助手随后明确答应。
# 防回声：(b) 里事实的实质必须出自她自己的话（助手一句「好的不催了」造不出事实）；她那句得是请求不是问句/玩笑；
# 助手那句得是明确的答应/承诺（不是问句、推托、英文思考文本）。
_JOKE = re.compile(r"开玩笑|玩笑|逗你|骗你|逗逗|整活|笑死|吓唬你|假装|演的|说着玩|瞎说的")    # 🤡 她天天用，不算玩笑标记
_INTERROG = re.compile(r"吗|是不是|有没有|对不对|怎么|为什么|难道|会不会|要不要|行吗|好吗|可不可以吗|嘛[？?]|什么意思")
_REQ_STRONG = re.compile(
    r"想要你|希望你|要你|请你|你(?:别|不要|不许|不准|不能再|不可以)|不许你|不准你|别再|不要再|以后|今后|每次|一律|必须|规矩|答应我|不用你|你不用|"
    r"不想让你|不想被你|(?:能不能|可不可以|可以不可以|行不行)(?:以后|每次|都)?(?:不|别|少)")
_REQ_WEAK = re.compile(r"你(?:要|得|先|少|多|一定|记得|应该)|我想要|我不想|我不要|不能|不可以|不许|不准|不要|别(?!人|的|处|说|看)|权力|权利|准你|允许")
_ACCEPT = re.compile(
    r"答应|保证|承诺|记住了?|记下|记牢|记在|不催了|不再|绝不|不会再|一定(?:会|不|做到|改)|我会|以后(?:不|都|我|就|会|一律|再不|绝不)|"
    r"听你的|遵命|说到做到|从今以后|从现在起|从此|没问题|可以的|按你说的|照你说的|就这么定|就这样|"
    r"好(?:的|嘞|哒|呀|呢)?[，,！!~。…\s]*(?:我)?(?:不|都|会|以后|就|答应|记)|行[，,！!~。…\s]*(?:我)?(?:不|都|会|以后|就|答应|记)|收到[，,！!]")
_HEDGE = re.compile(r"不一定|不保证|可能|也许|大概|试试|尽量|但是我不能|做不到|不行|不能答应|没法|无法答应|不答应")


def _strip_wrap(t: str) -> str:
    return _WRAP.sub("", (t or "").strip())


def _non_thinking(t: str) -> str:
    body = _strip_wrap(t)
    for x in thinking_sentences(body):
        body = body.replace(x, " ")
    return body


def user_request_line(line: str) -> str | None:
    """她这句话是不是在「要求/定规矩」：是返回 None，不是返回原因。问句、玩笑不算。"""
    t = _strip_wrap(line)
    if _JOKE.search(t):
        return "role:us-her-line-joke"
    if _INTERROG.search(t):
        return "role:us-her-line-question"
    if _REQ_STRONG.search(t):
        return None
    if not re.search(r"[？?]", t) and _REQ_WEAK.search(t):
        return None
    return "role:us-her-line-not-request"


def claude_accepts(line: str) -> str | None:
    """助手这句话是不是明确答应/承诺：是返回 None，否则原因。英文思考文本不算；只有问句、推托也不算。"""
    t = _non_thinking(line)
    if not t.strip():
        return "role:us-claude-line-thinking"
    if _HEDGE.search(t) and not re.search(r"答应|保证|承诺|不催了|不再", t):
        return "role:us-no-explicit-accept"
    if _ACCEPT.search(t):
        return None
    return "role:us-no-explicit-accept"


_AGREE_SPEAK = re.compile(r"约定|商定|约好|答应|承诺|保证|同意|达成一致|明确|表示|要求|希望|决定|不再|以后|今后|一律|每次|今天|昨天|明天|双方|一起|互相")


def _content_chars(t: str) -> set:
    return {c for c in re.sub(r"[\s\W_a-zA-Z0-9]+", "", t) if c not in _FUNC_CHARS}


def us_user_grounded(content: str, user_line: str) -> bool:
    """事实内容的实质出自她的那句话。先去掉「约定/答应/保证/不再…」这类约定套话和称呼，剩下的实质内容：
    二元组至少 2 个且 ≥ 30% 出自她的话，或实字（去虚字）至少 2 个且 ≥ 40% 出现在她的话里。助手的答应里带出来的新东西不算。"""
    body = _AGREE_SPEAK.sub("", _strip(_DATE_PREFIX.sub("", _PREFIX.sub("", content or ""))))
    her = _strip(_strip_wrap(user_line))
    g, u = _bigrams(body), _bigrams(her)
    if len(g & u) >= 2 and len(g & u) / len(g) >= 0.3:
        return True
    c = _content_chars(body)
    if not c:
        return False
    hit = len(c & _content_chars(her))
    return hit >= 2 and hit / len(c) >= 0.4


def us_request_then_accept(user_line: str, claude_line: str, content: str) -> str | None:
    """(b)：她先说的那句（user_line）是请求/规矩，助手随后那句（claude_line）明确答应，事实内容出自她的话。
    通过返回 None，不通过返回原因（role:us-…）。"""
    why = user_request_line(user_line)
    if why:
        return why
    why = claude_accepts(claude_line)
    if why:
        return why
    if not us_user_grounded(content, user_line):
        return "role:us-not-grounded-in-user-words"
    return None


_AGREE_START = re.compile(r"^[\s\W_]*(?:好(?:[的呀哒吧嘞滴啊哦呢]|[，,。！!~\s]|$)|行(?:[的呀吧嘞啊]|[，,。！!~\s]|$)|可以|对(?:[的呀啊哒嘛吧对嗯]|$)|嗯|恩|ok|OK|Ok|同意|没问题|就(?:这样|这么|按)|按你(?:说|的)|听你的|那就(?!是)|定了|赞同|采纳|认同|我也(?:觉得|这么))")
_AGREE_VETO = re.compile(r"但|不过|可是|然而|不行|不要|不是|不对|不好|别|改成|换成|等等|算了|先不|[？?]|吗|怎么|为什么|是不是")


_DELEGATE = re.compile(r"(?:都行|都可以|都好|看你|听你的|听你|你定|你来定|随你|你说了算|无所谓|你决定|你看着办|你安排)")
_DELEGATE_START = re.compile(r"^[嗯恩哦噢那就好吧呀啊哈]{0,3}" + _DELEGATE.pattern)
# 第六轮（用户 2026-10-28，最后一次微调）：只看她这句话的「第一个分句」（切在 ，。！？!?… / emoji / 空白处）。
_CLAUSE_BOUND = re.compile(r"[，,。.！!？?…；;~～\s]|[\u2190-\u2BFF\U0001F000-\U0001FFFF\uFE0F\u200d]")
_CLAUSE_VETO = re.compile(r"但|不过|可是|然而|不行|不要|不是|不对|不好|不起|别|改成|换成|等等|算了|先不|吗|怎么|为什么|是不是|不可以|拒绝|不同意")
_CLAUSE_QUESTION = re.compile(r"吗|怎么|为什么|是不是|有没有|对不对|好不好|行不行|可不可以|要不要")
_REFUSE_START = re.compile(r"^(?:不要|别|不行|不可以|不同意|不用|不是|不对|不好|算了|拒绝)")
_DECISION = re.compile(r"^(?:那先|那就(?!是)|先)(.{1,20}?)吧$")


def first_clause(line: str) -> tuple[str, str]:
    """她这句话的第一个分句，和切开它的那个字符（没有则 ''）。前导标点/emoji/空白先剥掉。"""
    t = re.sub(r"^[\s\W_]+", "", _strip_wrap(line))
    m = _CLAUSE_BOUND.search(t)
    return (t[:m.start()], t[m.start()]) if m else (t, "")


def _decision_matches(core: str, content: str | None) -> bool:
    """她自己拍板的话（那先…吧/那就…吧/先…吧）要和这条事实对得上：去掉 先/就 后的核心词（≥2 字）是事实正文的逐字子串。"""
    if not content or len(core) < 2:
        return False
    for c in {core, core.lstrip("先就")}:
        if len(c) >= 2 and c in content:
            return True
    return False


def user_consents(line: str, content: str | None = None) -> bool:
    """她这句是不是同意（共用的同意判定，(a) 路和重归属路都用它）。用户 2026-10-28：只看第一个分句——
    它以明确同意开头（好/行/可以/就这样/嗯嗯/OK…）或是委托式回复（都行/看你/随你/听你的/你定/你说了算/无所谓…）就算，
    后面跟什么都不管（不再限长、不看后面的问句）；第一个分句里带但/不过/改成、是问句、或开头是拒绝（不要/别/不行/不可以/不同意）的不算。
    她自己的拍板口吻（那先…吧/那就…吧/先…吧）在事实（content）对得上时也算。"""
    clause, endc = first_clause(line)
    if not clause or endc in ("？", "?"):
        return False
    m = _DECISION.match(clause)
    if m and not _CLAUSE_QUESTION.search(clause) and _decision_matches(m.group(1), content):
        return True
    if _REFUSE_START.search(clause) or _CLAUSE_VETO.search(clause):
        return False
    return bool(_AGREE_START.search(clause) or _DELEGATE_START.search(clause))


user_agrees = user_consents          # 旧名保留


REATTRIB_MAX_GAP = 4         # 助手建议之后，她的同意要在这么多条以内（且中间没有她别的话）
REATTRIBUTED: collections.Counter = collections.Counter()      # 本进程：助手建议被模型写成她的决定、她随后同意 → 改记 us（计数）


def find_user_agreement(chunk: list, hit: dict, content: str | None = None) -> dict | None:
    """12495 型：模型把助手的建议（hit）写成她的决定。她随后在窗口里明确同意（紧随其后的她的第一句话、在 REATTRIB_MAX_GAP 条内）就返回那条证据；否则 None。"""
    try:
        i = next(k for k, ev in enumerate(chunk) if ev is hit)
    except StopIteration:
        return None
    for ev in chunk[i + 1: i + 1 + REATTRIB_MAX_GAP]:
        if ev.get("source_actor") == persona.USER:
            return ev if user_consents(ev["content"], content) else None
        if ev.get("source_actor") not in (None, "claude"):
            continue
    return None



# ── 3. 后检 ─────────────────────────────────────────────────────────────
def _words(pats: str) -> re.Pattern:
    return re.compile(persona.fill(pats))


# 她转述身边人说的话（不是她自己的观点）：「我嫂子/我妈/妈妈/我爸/同事/同学/朋友/老师…说/讲/告诉我/跟我说/觉得/认为/表示」
_KIN = (persona.fill(r"(?:我|她|@@alt:USER@@的|@@alt:USER_FULL@@的)?") + r"(?:嫂子|嫂嫂|嫂|妈妈|妈|母亲|爸爸|爸|父亲|爹|娘|姐姐|姐|哥哥|哥|弟弟|妹妹|奶奶|外婆|爷爷|外公|舅舅|舅妈|姑姑|姨|阿姨|叔叔|"
        r"婆婆|同事|舍友|同学|同桌|朋友|闺蜜|好友|同事|老板|领导|老师|导员|辅导员|班主任|教官|医生|大夫|店员|客服|对象|男朋友|前任|网友|博主)")
_SAYV = r"(?:们|俩|都)?(?:后来|当时|刚才|之前|还|又|也|就|曾|一直)?(?:说(?!话)|讲(?!话)|告诉我|跟我说|对我说|和我说|提到|表示|觉得|认为|评价|吐槽|念叨|反馈)"
_REPORTED = re.compile(_KIN + _SAYV)
_TO_PREP = "跟对和向给与同"        # 「用户跟朋友说…」是她自己对别人说，不是别人的话：前面有这些介词的不算转述


def reported_speech(text: str) -> bool:
    """文本里有「我嫂子说/我妈觉得/同事告诉我…」这类转述引导（排除「跟/对/向…朋友说」「像老师说话」）。"""
    for m in _REPORTED.finditer(text or ""):
        if m.start() > 0 and text[m.start() - 1] in _TO_PREP:
            continue
        if m.start() > 1 and text[m.start() - 2] in _TO_PREP and text[m.start() - 1] in "她他我":
            continue
        return True
    return False

LISTS = {
    "post:setup-background": _words(r"是.{0,8}(?:的)?智能体|跑在|运行在|部署在|用.{0,8}账号登录|吃.{0,6}额度|共享.{0,6}窗口|已配置|配置为|接入了"),
    "post:process-log": _words(r"过期|失效|重新登录|重新授权|重试|已修复|修复了|修好|重启|超时|报错|崩溃|恢复正常|恢复后|被撤销|撤销了|被拦|拦截|构建失败|构建成功|压缩.{0,8}失败|没生成|漏改|误判"),
    "post:retelling": _words(r"据说|听说|据(?:她|@@alt:USER@@|他|对方|报道)说|转述|传闻|网上说|有人说|别人说|被告知|按.{0,6}的说法"
                             r"|那次是他答应的|替我答应的|(?:上一个|前一个|另一个|别的|旧)窗口(?:的|里)?(?:@@alt:CLAUDE@@|他).{0,3}(?:答应|承诺)"),
    "post:metric": _words(r"\d+(?:\.\d+)?\s*(?:%|％|万?\s*tokens?|条对话|条消息|条记录|毫秒|ms\b|MB|GB|字节|次调用|行代码)|命中率|占用率"),
    "post:transient-state": _words(r"正在|目前|暂时|尚未|还没有|在线|离线"),
    "post:third-party-eval": None,                          # 下面单独处理（要同时有第三方名和评价词）
    "post:summary-report": _words(r"总结|汇总|总体来看|综上|本次汇报|复盘|日报|周报|概述|报告了|摘要如下|摘要[:：]"),
    "post:recommendation": _words(r"建议你|建议[:：]|推荐[:：]|推荐你|可以考虑|不妨|应该试试|最好是|方案[一二三四]|备选"),
    "post:encouragement": _words(r"加油|你很棒|相信自己|别怕|会好的|辛苦了|抱抱|别担心"),
}
# 「额度」光秃秃一个词只是话题（聊额度的事常常带着真决定），不像「额度 8%」那样是指标：软词表
_BARE_QUOTA = re.compile(r"额度|烧完")
# 软词表：下面两种「更重要的话」出现时让路（硬词表：转述、第三方评价、数字指标、设定、推荐永远照挡）
SOFT_LISTS = ("post:process-log", "post:transient-state", "post:summary-report", "post:encouragement")
# 助手的承诺/自我纠正/自己定的规矩：赢过鼓励、过程、状态、总结词（「我错了…我保证下次…抱抱」是道歉+承诺，不是安慰）
_COMMIT_WIN = re.compile(r"保证|下次|以后|今后|从今以后|规矩|定下|约定|我错了|不再|绝不|一定(?:会|要|不)|会注意|承诺|答应|认错|改正|承认|改口|不该|做法[:：]")
# 她明确要求/决定/偏好的话：同样赢过过程、状态、总结、光秃秃的「额度」
_USER_DURABLE = re.compile(r"要求|希望|决定|同意|答应|定下|规定|规矩|不许|不准|不要|别再|不用|以后|今后|从此|每次|一律|必须|喜欢|讨厌|不喜欢|想要|换成|改成|改为|提醒我|建议")
_THIRD = re.compile(persona.fill(r"GPT|@@alt:GPT,C,OTHER_AI@@|GLM|DeepSeek|Codex|Gemini"), re.I)
_EVAL = re.compile(r"嘴硬|不靠谱|笨|垃圾|烂|太强|很强|偷懒|甩锅|摆烂|不行|蠢|离谱|很差|能力强|聪明|靠谱|糊弄|敷衍|说谎|撒谎")
_PREFIX = re.compile(persona.fill(r"^(?:我|@@alt:USER,CLAUDE,USER_FULL@@)（[^）]*）[：:]\s*"))
_DATE_PREFIX = re.compile(r"^(?:\d{4}[-年]\d{1,2}[-月]\d{1,2}[日号]?(?:[凌晨早上中午下午晚上]+)?|\d{1,2}月\d{1,2}[日号]?)[，, ]*")


def _bigrams(s: str) -> set:
    s = re.sub(r"[\s\W_]+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def grounded(content: str, evidence_text: str, thr: float = 0.3) -> bool:
    """内容里的实词（去掉前缀/日期/称呼）有多少二元组出现在原话里。"""
    body = _DATE_PREFIX.sub("", _PREFIX.sub("", content or ""))
    body = re.sub(persona.fill("@@alt:USER_NAMES,CLAUDE,GPT,C@@"), "", body)
    g = _bigrams(body)
    if len(g) < 4:
        return True
    return len(g & _bigrams(evidence_text)) / len(g) >= thr


USER_GROUND_THR = 0.10          # 她的偏好/生活事实：至少这么多内容二元组要出自她自己的原话（别人的话只许解释指代）
_QUESTION = re.compile(r"[？?]|吗[，。,!！🤡🌚🙈👀 ]|吗$|怎么|为什么|是不是|有没有|对不对")
_QUESTION_ASSERT = re.compile(r"答应|决定|定下|规矩|必须|不许|不准|不要|别再|要求|希望|一律|每次|以后|今后|喜欢|讨厌|买了|去了|搬|生病|考|毕业|报名|换成|改成")


def _strip(body: str) -> str:
    return re.sub(persona.fill("@@alt:USER_NAMES,CLAUDE,GPT,C@@"), "", body)


def window(lines: list, i: int, k: int = 8) -> tuple[str, str]:
    """lines＝同一会话按时间排好的 (actor, text)。返回第 i 条前后各 k 条的 (全部文本, 她自己的文本)。
    依据检查看这个窗口而不是只看引文那一条：指代要靠上下文才懂的短句，依据在邻近几条里。"""
    seg = lines[max(0, i - k): i + k + 1]
    return "\n".join(t for _, t in seg), "\n".join(t for a, t in seg if a == persona.USER)


def quote_source(content: str, lines: list, i: int, actor: str, k: int = 8, thr: float = 0.2, min_shared: int = 3) -> int | None:
    """第 i 条是应答/太短、不能当引文来源时：看邻近同一说话人的、有内容的几条，合起来对得上这条事实就返回其中最近一条的下标，否则 None。
    （真管线里模型得引另一条；回放没有引文，用这个近似。）"""
    body = _DATE_PREFIX.sub("", _PREFIX.sub("", content or ""))
    if len(_bigrams(_strip(body))) < 4:
        return None                                      # 事实本身就是个应答（「用户：666」），邻居再有内容也不能拿来给它当依据
    cand = [j for j in range(max(0, i - k), min(len(lines), i + k + 1))
            if lines[j][0] == actor and not prefilter(lines[j][1], actor)]
    if not cand:
        return None
    ctx = "\n".join(lines[j][1] for j in cand)
    if not grounded(content, ctx, thr) or len(_bigrams(_strip(body)) & _bigrams(ctx)) < min_shared:
        return None
    return min(cand, key=lambda j: abs(j - i))


def infer_category(content: str, actors: set, subject: str, ftype: str | None = None) -> str | None:
    """旧事实没有 category：用关键词粗推（回放用；真管线由模型给）。助手自己立的规矩/承诺算 commitment，承认并改正算 self_correction。"""
    if subject == "us":
        return "agreement"
    if subject == "claude" or actors == {"claude"}:
        if _SELFC.search(content):
            return "self_correction"
        if _COMMIT.search(content):
            return "commitment"
        return "other"
    if subject == persona.USER:
        if _CORRU.search(content):
            return "correction"
        if ftype == "preference" or _PREF.search(content):
            return "preference"
        if _DEC.search(content):
            return "decision"
        if _LIFE.search(content):
            return "life_change"
        return None
    return None


def infer_category_union(content: str, actors: set, subject: str, ftype: str | None = None) -> str | None:
    """召回补偿路（旧抽取器候选没有 category）：infer_category，推不出来时她的大事（搬家/考研/辞职…）算 life_change。"""
    c = infer_category(content, actors, subject, ftype)
    if c is None and subject == persona.USER and _BIG_LIFE.search(content or ""):
        return "life_change"
    return c


_COMMIT = re.compile(persona.fill(r"答应|承诺|保证|我会|我来|我负责|以后我|不再|一定会|说到做到|记住了|我去做|我帮你|必须|每次|凡是|规矩|约定|以后.{0,12}(?:先|都|不|要)"
                     r"|以后(?:会|都|要|一律|不|再|每|讲|用|先|@@alt:USER@@说|遇到|碰到)|今后|(?:@@alt:CLAUDE@@|定下|决定|记下|[，；：。]|^)以后|下次(?:不|会|要|先|再|一定)|会注意|定下(?:的)?(?:做法|规矩)|做法[:：]|明确表态|决定以后|不接话|装(?:了|个)?(?:技术)?护栏"))
# 承认/认错：只算助手自己认的（「它承认…」「他承认…」是转述别人），不收「改口了？」这种打趣
_SELFC = re.compile(persona.fill(r"教训|吸取|我错了|我弄错|我搞错|我之前.{0,10}错|纠正了?自己|我改正|我改了|是我的问题|我不该|我不应该|我的错"
                    r"|(?:@@alt:CLAUDE@@|[：；，。]|^)(?:当场|主动|也|并|又)?(?:承认|认错|认下)|被@@alt:USER@@纠正|说错了|记岔|撤掉|错题本|擅自|又幻觉"))
_CORRU = re.compile(r"不是.{1,12}是|纠正|说错了|记错了|不对|别再|不许|不要再")
_PREF = re.compile(r"喜欢|讨厌|不喜欢|爱吃|不爱|希望你|想要你|受不了|最爱|偏好|不吃|不想")
_DEC = re.compile(r"决定|定下|规矩|以后都|从今天起|就这么定|约定")
_LIFE = re.compile(r"买了|去了|搬|换了|生病|发烧|考了|开始|结束|毕业|报名|入手|刚到|回家|出门|摔|就医|分手|在忙|上课|培训")


_FUNC_CHARS = set("你我他她它的了是在也都就和与及并而但把被让给对又还才将这那有不很吗呢吧啊嘛哦一个什么怎么为因所以如果可以会要说想觉得认为提到表示") | set(persona.N_USER + persona.N_CLAUDE + persona.N_C + persona.USER_FULL)
CHAR_COVER_THR = 0.4


def char_cover(content: str, line: str) -> float:
    """复述/换人称（「被你催烦了」→「不喜欢助手催她」）二元组对不上，但关键字几乎都在原话里：内容里的实字（去掉虚字、称呼、日期）有多少出现在这条原话里。"""
    def cs(t):
        body = _DATE_PREFIX.sub("", _PREFIX.sub("", t or ""))
        return {c for c in re.sub(r"[\s\W_a-zA-Z0-9]+", "", body) if c not in _FUNC_CHARS}
    c = cs(content)
    return 1.0 if len(c) < 4 else len(c & cs(line)) / len(c)


def _shares_cue(content: str, line: str) -> bool:
    """她的话负责「态度」（喜欢/不要/决定…），上下文只负责说清指代：她这句话里的态度词，事实里也写了。"""
    a = set(_USER_DURABLE.findall(line or ""))
    return bool(a & set(_USER_DURABLE.findall(content or "")))


_OWN_VOICE = re.compile(r"我(?:自己)?(?:觉得|认为|打算|决定|也觉得|也认为|不同意|同意|赞同)|我想")      # 引文本身是她自己的表态
_OWN_STANCE = re.compile(r"认同|赞同|同意|采纳|接受|打算|决定|也觉得|也认为|听了")


def _content_retelling(content: str, body: str, verbatim: bool) -> bool:
    """事实内容本身是转述：模型写的第三人称内容里出现「我嫂子说/妈妈认为…」且没有她自己的表态（认同/打算/决定…）；
    整句照抄原话的只看开头（「我嫂子说…」打头）——整行里夹着一句转述不代表整行都是转述。"""
    if verbatim:
        return reported_speech(body[:14])
    return reported_speech(content) and not _OWN_STANCE.search(content)


def _reported_before(line: str, quote: str, back: int = 40) -> bool:
    """引文所在那条里，引文开头（含引文开头前 back 字内）出现「我嫂子说/我妈说/同事说…」这类转述引导。
    用户（时间）：前缀不算；找不到引文位置（规范化后才对得上）时看整条开头。"""
    body = _WRAP.sub("", line or "")
    i = body.find(quote)
    if i < 0:
        nb, nq = norm_quote(body), norm_quote(quote)
        i = 0 if nq not in nb else max(0, nb.find(nq))
        win = body[:max(0, i) + len(quote)] if i == 0 else body[:i + len(quote)]
        return reported_speech(win[-(back + len(quote)):])
    return reported_speech(body[max(0, i - back): i + max(len(quote), 8)])


def postcheck(fact: dict, evidence_text: str, *, actor: str | None = None, context_text: str | None = None,
              user_text: str | None = None) -> str | None:
    """fact: subject / category / content / quote（都可缺）。返回拒绝原因或 None。
    evidence_text＝引文所在的那条原话（quote 子串检查用）；context_text＝整个片段/邻近窗口（内容依据检查用，缺省同 evidence_text）：
    指代要靠上下文才懂的短句（「弄好了」）抽出的事，依据在邻近几条里，不能只对着那一条找。
    user_text＝窗口里她自己的原话：她的偏好/生活事实仍要有一部分出自她自己的话（别人的上下文只许解释指代，不能凭空造出她的偏好）。"""
    subject = str(fact.get("subject") or "").strip()
    category = fact.get("category")
    content = str(fact.get("content") or "").strip()
    quote = str(fact.get("quote") or "").strip()
    if fact.get("keep") is False:
        return "post:model-said-no"
    if "category" in fact and category not in CATEGORIES:
        return "post:category-not-in-enum"
    if category is not None and actor:
        ok_set = ROLE_ALLOWED.get(actor, set()) | set(US_CATEGORIES)
        if category not in ok_set:
            return "post:category-not-allowed-for-role"
    if "quote" in fact:
        if len(quote) < 4:
            return "post:no-quote"
        if not quote_in(quote, evidence_text or ""):          # 弯/直引号、全/半角标点、空白、大小写不算差异
            return "post:quote-not-substring"
        if len(norm_quote(quote)) > QUOTE_REJECT:
            return "post:quote-too-long"
    if not content:
        return "post:empty-content"
    if subject in _VAGUE_SUBJ:
        return "post:vague-subject"
    verbatim = bool(_PREFIX.match(content))          # 「用户（时间）：原话」这种逐字拷贝没有主语可查
    body = _DATE_PREFIX.sub("", _PREFIX.sub("", content))
    if _PRON.match(body) and subject != "claude" and not verbatim:
        return "post:pronoun-subject"
    ai_side = subject in ("claude", "gpt", persona.AGENT_C) or category in AI_CATEGORIES
    if ai_side and _COMMIT_WIN.search(content):
        wins = True                                      # 承诺/自我纠正赢过鼓励、过程、状态、总结词
    else:
        wins = False
    user_durable = subject == persona.USER and not verbatim and bool(_USER_DURABLE.search(content))   # 整句照抄原话的不算：话里碰巧有「不用」「建议」不代表这是条决定
    for name, rx in LISTS.items():
        if rx is None or not rx.search(content):
            continue
        if name in SOFT_LISTS and (wins or user_durable):
            continue
        return name
    if _BARE_QUOTA.search(content) and not (user_durable or wins):
        return "post:metric"
    if _THIRD.search(content) and _EVAL.search(content):
        return "post:third-party-eval"
    if subject != "claude" and (_content_retelling(content, body, verbatim)
                                or (quote and not _OWN_VOICE.search(quote) and _reported_before(evidence_text, quote))):
        return "post:retelling"                              # 引文前面是「我嫂子说…」：她转述别人的话，不是她的观点
    if verbatim and subject == persona.USER and len(body) <= 70 and _QUESTION.search(body) and not _QUESTION_ASSERT.search(body):
        return "post:verbatim-question"                  # 原话整句照抄、又只是个问句/闲聊：不是事实
    gcontent = _AGREE_SPEAK.sub("", content) if category == "agreement" else content      # 约定类：「约定/答应/保证/不再…」是套话，不要求原话里有
    if (context_text or evidence_text) and not grounded(gcontent, context_text or evidence_text) \
            and not (evidence_text and char_cover(gcontent, evidence_text) >= CHAR_COVER_THR):
        return "post:ungrounded"
    if user_text is not None and subject == persona.USER and (category is None or category in USER_CATEGORIES) \
            and not grounded(content, user_text, USER_GROUND_THR) and not _shares_cue(content, evidence_text):
        return "post:ungrounded-in-user-words"
    return None


# ── 3b. 召回补偿：优先级 / 去重 / 封顶（旧抽取器候选 ∪ 新提示词输出） ─────────────────────────
# 大事的词：搬家、考研、辞职、分手、毕业、转学……（她的生活里一次性但改变后面的事）
_BIG_LIFE = re.compile(r"搬出|搬家|搬去|搬到|考研|保研|读研|考公|辞职|离职|入职|offer|分手|复合|毕业|休学|退学|转学|转专业|出国|留学|订婚|结婚|怀孕|手术|住院|确诊|买房|租房|换工作|实习")


# 旧抽取器的候选只补「规矩类」：喜欢/决定/要求/同意这类泛词不算（否则带进「喜欢某剧/某游戏」「某次分工」之类的杂事）
_RULE_STRONG = re.compile(r"不许|不准|不要|别再|以后|今后|从此|每次|一律|必须|规矩|规定|提醒我")


def is_priority(subject: str, category: str | None, content: str, actor: str | None, *, quote: str = "", strict: bool = False) -> bool:
    """「高置信的大事」：她自己说的决定/规矩/纠正/偏好（带明确要求或决定词）、她生活里的大变化、助手自己说的承诺/自我纠正（带承诺词）。
    只有它们超过每片段 3 条后还能进（总共 ≤ MAX_PER_SLICE_PRIORITY）。"""
    c = content or ""
    if subject == persona.USER and actor == persona.USER:
        if category in ("decision", "correction", "preference") and (_RULE_STRONG if strict else _USER_DURABLE).search(c):
            return True
        if category in ("life_change", "decision") and _BIG_LIFE.search(c):
            return True
        return False
    if subject == "claude" and actor == "claude" and category in AI_CATEGORIES:
        return bool(_COMMIT_WIN.search(c))
    return False


def _jacc(a: str, b: str) -> float:
    x, y = _bigrams(a), _bigrams(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def is_dup(a: dict, b: dict, thr: float = 0.5) -> bool:
    """两条通过后检的事是不是同一件事：同主体且（引文互相包含 / 同一条证据上内容相近 / 内容二元组 Jaccard ≥ thr）。"""
    if a.get("subject") != b.get("subject"):
        return False
    qa, qb = norm_quote(a.get("quote", "")), norm_quote(b.get("quote", ""))
    if qa and qb and (qa in qb or qb in qa):
        return True
    j = _jacc(a.get("content", ""), b.get("content", ""))
    if a.get("evidence_id") is not None and a.get("evidence_id") == b.get("evidence_id") and j >= 0.3:
        return True
    return j >= thr


def select_with_cap(oks: list[dict], cap: int = MAX_PER_SLICE, cap_hi: int = MAX_PER_SLICE_PRIORITY) -> list[tuple[dict | None, str]]:
    """oks：通过 check() 的事（新提示词的在前、旧抽取器的在后），顺序即优先级。返回等长的 [(ok 或 None, 原因)]。
    先去重（后来的同事被前面的吞掉：union:dup）；再封顶：前 cap 条直接进；超出的只有 priority 的能进，总共不超过 cap_hi。"""
    out: list[tuple[dict | None, str]] = []
    kept: list[dict] = []
    for ok in oks:
        if any(is_dup(ok, k) for k in kept):
            reject("union:dup")
            out.append((None, "union:dup：值得记闸"))
            continue
        if len(kept) < cap or (ok.get("priority") and len(kept) < cap_hi):
            kept.append(ok)
            out.append((ok, ""))
        else:
            reject("cap:per-slice")
            out.append((None, "cap:per-slice：值得记闸"))
    return out


# ── 4. 置信度 ────────────────────────────────────────────────────────────
def confidence_for(category: str | None, subject: str, actor: str | None, *, model_conf: float | None = None) -> float:
    """她明确说的：0.9；助手的承诺 / 有她确认的 us：0.6；其余 0.5（不再一律 0.9）。"""
    if subject == "us" or (actor in ("claude", "gpt", persona.AGENT_C)):
        return 0.6
    if actor == persona.USER and subject == persona.USER and category in USER_CATEGORIES:
        return 0.9
    return 0.5


# ── 5. 抽取提示词（开关开时代替 from_evidence.PROMPT） ───────────────────
# 补充规则（一条一条试过再留；见 TALK 试跑记录）：每条是一小段文字，放进 PROMPT_GATE 的 {extra}。
TWEAKS: dict[str, str] = {
    "A": persona.fill("\n@@CLAUDE@@承认并收回自己的错（哪怕起因是@@USER@@指出来的）：一律记成 subject=claude、category=self_correction，不要记成@@USER@@的纠正。"),
    "B": "\n一次性的请求、临时安排、界面/文案的改动指令（「把这个删掉」「明天带电脑」）不记；decision 只收「以后一直照做」的规矩和重大的决定。",
    "C": "\n亲密互动里的台词、调情的话（「以后每天都要…」）不是承诺，不记。",
}
EXTRA_RULES = TWEAKS["B"]             # 真模型试验后留下的补充规则（A、C 试了没有改善，没留；见 TALK）；只在闸开时进提示词，闸关用旧提示词一字不差

PROMPT_GATE = persona.fill("""下面是一段按时间排好的聊天原话。说话的人：@@ROSTER_USER@@、@@CLAUDE@@（我，AI）、@@GPT@@（@@DESC_GPT@@）、@@C@@（@@DESC_C@@）。每条前面方括号里是编号。

你是记忆库的守门人，**预计会拒绝其中绝大多数片段**。只有同时满足两条才记：
  ① 30 天后这件事还会影响她和@@CLAUDE@@怎么相处、怎么做事（30 天检验）；
  ② 它属于下面的封闭类别之一，并且原话里有一句能逐字证明它。

封闭类别（category，只能从这里选）：
  preference       她的喜好、厌恶、对别人该怎么做的要求          —— 只能来自@@USER@@自己的原话
  life_change      她生活里的变化：买了、去了、搬了、病了、开始/结束了某件事 —— 只能来自@@USER@@自己的原话
  correction       她纠正@@CLAUDE@@（或别人）的错误                      —— 只能来自@@USER@@自己的原话
  decision         她定下的做法、规矩、决定                           —— 只能来自@@USER@@自己的原话
  commitment       @@CLAUDE@@对@@USER@@许下的承诺/答应了要做的事              —— 只能来自@@CLAUDE@@自己的原话，subject 必须是 claude
  self_correction  @@CLAUDE@@承认并改正自己的错误                          —— 只能来自@@CLAUDE@@自己的原话，subject 必须是 claude
  agreement        @@USER@@和@@CLAUDE@@之间的约定，两人各自说过话，两种都算：
                   ① @@CLAUDE@@提议、**@@USER@@随后明确答应**；② **@@USER@@先提出要求/定规矩（她自己的话）、@@CLAUDE@@随后明确答应/承诺**。
                   —— subject 写 us；quote 和 quote2/src2 分别是两方各自的原话（顺序不限）；事的实质必须出自@@USER@@自己的话或她明确答应的那句，
                   只有@@CLAUDE@@一个人说「好的我不催了」不算；她只是问句、玩笑、没有要求的，不算；@@USER@@的要求没人答应的是她的 preference，不是 us

**一律不要**（这些统统 keep=false）：
  · 设定/背景介绍（谁是什么、跑在哪、用谁的账号）
  · 过程流水账（过期、重新登录、重试、已修复、重启、超时、报错、恢复）
  · 转述别人的话或事（「据说」「他说」「GPT 说…」）
  · 对第三方（GPT、@@C@@、别的 AI 或人）的评价
  · 总结、汇报、复盘、清单
  · 推荐/建议列表
  · 泛泛的安慰和鼓励
  · 暂时的状态（正在、目前、暂时）
  · 指标和数字（额度、token、条数、百分比）
@@CLAUDE@@一个人的话，只能产出 subject=claude、类别是 commitment 或 self_correction 的事；
@@USER@@的偏好、生活变化、纠正只能来自@@USER@@自己的话；@@CLAUDE@@说「@@USER@@怎样怎样」只是转述，不要记。
一个片段最多给 3 条。{extra}

每条输出：
  keep（true/false）、category（上表之一）、subject（@@A_USER@@/claude/us，gpt/@@A_C@@ 只在它们自己的原话里）、
  content（一句中文第三人称，明确主语，不超过 60 字，不加评价）、
  quote（从某一条原话里**逐字**复制的 4～40 字）、anchor（quote 所在那条的编号，写进 src）、quote2/src2（仅 agreement）。
没有值得记的就输出 {{"facts": []}}。

只输出 JSON：{{"facts": [{{"keep": true, "category": "...", "subject": "...", "content": "...", "quote": "...", "src": 编号, "quote2": "（仅 agreement）", "src2": 编号}}]}}

原话：
{chunk}""")
