#!/usr/bin/env python3
"""UserPromptSubmit hook：说话之前先想一下。

2026-08-18 方向调整：记忆不该等人来问才去想，对话里提到过去的某件事时，
应当像人一样自动想起来、先检索再回答；想不起来就明确说想不起来，而不是凭上下文乱说。

整套记忆库挂在 MCP 上，但要模型**自己想起来**去调它——它不记得，
所以工具再好，不接到「说话之前」就等于没有。这个 hook 就是把它接上：用户的话一进来，先自动去翻一遍，
翻到的塞进上下文，翻不到**明确说翻不到**——后者一样重要，
因为「查过了没有」才是不许乱编的依据。

规矩：
  · 她抛软话撒娇叫人的时候一个字都不查（ROUTER.md 第 0 条，Router 自己会判）
  · 只读不写：**不调 touch()**。误召回一次就加权，会自己把错误喂胖
    （外部评审提出的）
  · 输出有字数上限，它每轮都进上下文，不能吃太多
  · 任何异常静默 exit 0。想不起来不该卡住说话
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

# cc-connect 的隔离定时任务不需要伴侣聊天记忆。跳过能避免后台日记、体检等
# 每次启动都额外跑一轮召回模型，也防止任务提示被当成真实聊天写进遥测。
if "#cron:" in os.environ.get("CC_SESSION_KEY", ""):
    raise SystemExit(0)

# 代码/库所在目录：环境变量 ENGRAM_HOME，缺省＝本文件所在目录（符号链接会被解开）。
# 钩子如果是被拷贝到别处安装的，必须设 ENGRAM_HOME，否则找不到 persona / 检索模块。
_ROOT = os.environ.get("ENGRAM_HOME") or str(pathlib.Path(__file__).resolve().parent)
sys.path.insert(0, _ROOT)
try:
    import persona
except ImportError:                                           # 找不到：记一行日志，静默退出（想不起来不该卡住说话）
    try:
        with open(os.path.join(_ROOT, "hook.log"), "a", encoding="utf-8") as _fh:
            _fh.write(f"import persona 失败：{_ROOT}（设 ENGRAM_HOME）\n")
    except OSError:
        pass
    raise SystemExit(0)
ENGRAM = persona.HOME
LOG = os.path.join(ENGRAM, "hook.log")
_ENT = persona.alt(*persona.ENTITY_TERMS)
ENTITY_ALT = _ENT + "|" if _ENT else ""            # 人／物／地名的专名（ENGRAM_ENTITY_TERMS，顺序即正则顺序）
_TOP = persona.alt(*persona.TOPIC_TERMS)
TOPIC_ALT = _TOP + "|" if _TOP else ""             # 我们反复谈的话题词（ENGRAM_TOPIC_TERMS）
MAX_CHARS = 420
MAX_HITS = 2
PER_HIT = 140
MAX_AUTO_INPUT = 700

# 注入时把状态标出来。库里 candidate 远多于 confirmed（具体数字会变，别在注释里写死）——
# 不标的话，一条待核对的抽取和一条确认过的规则长得一模一样，我分不出来，
# 就会拿着「某某说过」直接当规矩执行。这是「候选被当事实」的入口。
STATUS_MARK = {
    "candidate": "待核",
    "confirmed": "确认",
    "uncertain": "存疑",
    "superseded": "已被取代",
    "invalid": "已作废",
}

ECHO_W = 1 / 3          # 与 echo.ECHO_WEIGHT / consolidate.ECHO_WEIGHT 一致（测试固定）


def echo_prefix(h: dict) -> str:
    """只有转述的候选前面标〔转述·来自某某〕（精排和注入都用）。不是转述返回空串。"""
    src = h.get("echo_from")
    out = ""
    if src:
        try:
            import echo
            out = echo.label(src)
        except Exception:                                    # noqa: BLE001
            out = "〔转述〕"
    if h.get("self_note"):                                   # 助手自记：标记只存旁表，这里显示时才加
        out += f"〔{persona.N_CLAUDE}自记〕"
    if h.get("assistant_say"):                               # 我自己说过的话（echo.annotate 打的标）：不是她的事实
        out += "〔我说的〕"
    return out


RE_CHANNEL = re.compile(r"<channel[^>]*>(?P<body>.*?)</channel>", re.S)
RE_TELEGRAM = re.compile(
    r"\[TELEGRAM_EXTERNAL_MESSAGE\].*?original_body=(?P<q>[\"'])(?P<body>.*?)"
    r"(?P=q).*?\[/TELEGRAM_EXTERNAL_MESSAGE\]", re.S)
SYNTHETIC = re.compile(
    r"^\s*(?:<task-notification>|<system-reminder>|\[tool|\[document|\(document:|"
    + persona.fill(r"你在替@@alt:CLAUDE@@守夜)"), re.I)
EXPLICIT_RECALL = re.compile(
    r"(上次|上回|之前|以前|当时|那天|昨天|前天|还记得|"
    r"记不记得|你说过|我说过|我跟你说过|原话|查一下|翻一下|翻翻)")

# ── 上下文里已经有的就不查（用户 2026-08-18 定的规矩，2026-09-12 换了实现）
#
#   「上下文和交接页里有的就不需要检索了，因为这些都是已知而且是比较新的信息」
#
# 规矩没错，错的是我第一版的实现：拿这次要查的**词**去翻最近对话和常驻文件，
# 词出现过就不翻库。可是**词在上下文 ≠ 答案在上下文**。
#
# 实测：一句提到某个旧决定的话，库里有 6 条相关记录，
# 因为 MEMORY.md 的索引里写着相关的词，6 条全被挡掉。而索引那一行只说
# 「有这件事」，当初为什么这么选、各项怎么排序，全在被挡掉的那 6 条里。
# 这道闸在日志里触发了 2900 次，是整个钩子最活跃的分支。
#
# 新实现按她原话的本意走：判的不是「这个词眼前有没有」，而是
# **「这条记录本身，刚才是不是真的已经注进去了」**。
# 同一条 fact 在窗口期内注入过、且到现在仍然有效，就不再注第二遍。
# 拿不准的一律重注——宁可重复一条，不假装知道它还在我眼前。
REINJECT_COOLDOWN_MIN = 30              # 同一条记录多久之内不重复注入


# 注入时按字数硬截会截在半句（2026-09-27 桨板落水：结局「被救上来了」被截掉，
# 我当成悲剧讲，被她纠正）。改成截在句末；截不到句末就明说后面还有，别让半句冒充全文。
SENT_END = "。！？；!?;\n"


def clip(text: str, limit: int = PER_HIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(c) for c in SENT_END)
    if cut >= limit * 0.5:
        head = head[:cut + 1]
    return head.rstrip() + "……（后文略，未完，别据此下结论；要全文按 id 查）"


def recently_injected(con, minutes: int = REINJECT_COOLDOWN_MIN) -> set[str]:
    """窗口期内真的注入过、并且到现在还有效的那些记录。

    数据来自遥测表——这是今天刚把钩子接上去的那套，记的是「实际注入」
    而不是「查到了」，所以这里能拿到的是真账。
    失效、被取代、我已表态不认的，不算数：那些本来就该重新审视。
    """
    try:
        rows = con.execute(
            """SELECT DISTINCT f.ext_id
                 FROM recall_hits h
                 JOIN recall_runs r ON r.id = h.run_id
                 JOIN facts f ON f.id = h.fact_id
                WHERE h.selected_for_context = 1
                  AND r.router_decision = 'recall'
                  AND datetime(r.created_at) >= datetime('now', ?)
                  AND f.status IN ('candidate','confirmed')
                  AND f.valid_to IS NULL""",
            (f"-{int(minutes)} minutes",))
        return {row[0] for row in rows}
    except Exception as e:                                   # noqa: BLE001
        log(f"去重查询失败（按不去重处理）：{e}")
        return set()


def log(msg: str) -> None:
    try:
        from redact import redact as _red                    # 日志里的用户话头也打码；导不进就不写
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {_red(msg)[0]}\n")
    except Exception:
        pass


def incoming(data: dict) -> str:
    """取这一轮她真正说的话。通道消息包在 <channel> 里，取里面的正文。"""
    text = data.get("prompt") or ""
    bodies = [m.group("body").strip() for m in RE_CHANNEL.finditer(text)]
    if bodies:
        return bodies[-1]
    telegram = list(RE_TELEGRAM.finditer(text))
    return (telegram[-1].group("body") if telegram else text).strip()


def skip_input(text: str) -> str | None:
    if SYNTHETIC.search(text):
        return "系统／工具消息"
    if len(text) > MAX_AUTO_INPUT and not EXPLICIT_RECALL.search(text):
        return "长篇粘贴且未要求回忆"
    return None


# ── 词有没有区分度（2026-09-12 加）
# 她说"弄好了""你再看看呢"的时候，钩子照样去查，然后拿"弄好""看看"匹配到
# 八月份某段完全不相干的旧话塞给我。翻了整份日志：召回 868 次，真正帮上忙的一次都没有；
# 真正查对了的事全是我自己 grep 的。
# 结论：这个钩子几乎没起作用。
#
# 所以门槛抬高：只有出现真正有指向性的词才查。宁可不召回，也不要拿噪音占上下文。
DISTINCTIVE = re.compile(
    r"("
    r"[0-9]{1,2}\s*月\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}|"      # 明确日期
    + ENTITY_ALT +                                                       # 人／物／地名的专名（配置）
    r"[A-Za-z][A-Za-z0-9_.\-]{3,}|"                                    # 英文标识符：文件名、命令、型号
    + TOPIC_ALT +                                                        # 我们反复谈的事（配置）
    r"记忆库|交接|日记|看板|唤醒|额度|停机"
    r")"
)

# 指向性最强的一类：一个就能认定「说的是这件事」
STRONG = re.compile(
    r"[0-9]{1,2}\s*月\s*[0-9]{1,2}|[0-9]{4}-[0-9]{2}-[0-9]{2}"
    + ("|" + ENTITY_ALT.rstrip("|") if ENTITY_ALT else ""))

# 这些词单独出现不算数——太常见，匹配上等于没匹配
TOO_COMMON = {"claude", "code", "http", "https", "json", "text", "post", "photo", "document"}

# 2026-10-28：英文停用词。hook.log 里 photo/workspace/processed/notification 这些词
# 来自图片附件和系统通知的模板文字，不是用户说的话，却被当成「有指向性的英文标识符」触发召回。
STOP_EN = TOO_COMMON | {
    "image", "images", "img", "workspace", "ubuntu", "home", "processed", "process",
    "notification", "notifications", "attachment", "attachments", "saved", "locally",
    "also", "connect", "true", "false", "null", "none", "file", "files", "path", "message",
    "messages", "user", "assistant", "system", "reminder", "task", "done", "tool", "result",
    "sticker", "voice", "video", "audio", "caption", "reply", "wake", "pending", "quiet",
}

# 附件/通知的包装行：整行剥掉再抽词、再检索。
RE_WRAP_BLOCK = re.compile(
    r"<(system-reminder|task-notification)\b[^>]*>.*?</\1>", re.S | re.I)
RE_WRAP_LINE = re.compile(
    r"^\s*[\(\[（【]?\s*(?:images?\s+(?:also\s+)?saved\s+locally|attachments?\s+saved|"
    r"image_path\s*=|photo|image|document|sticker|voice|video)\b[^\n]*$",
    re.I | re.M)
RE_WRAP_PATH = re.compile(
    r"(?:/home/\w+/\S+|/tmp/\S+|\S+\.(?:jpe?g|png|webp|gif|mp4|mov|pdf|ogg|oga|mp3))", re.I)


def strip_wrappers(text: str) -> str:
    """剥掉附件/通知的包装文字，只留她真正说的话。剥完是空串就说明这条没有可检索的内容。"""
    text = RE_WRAP_BLOCK.sub(" ", text or "")
    text = RE_WRAP_LINE.sub(" ", text)
    text = RE_WRAP_PATH.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


# 问句（2026-09-19 加）：考卷第一次考 0 分，61 道题死在「无指向词」——
# 「那个约定到底是谁先提的」「哪个方案更稳来着」这种正经问旧事的，没有专名就被放过了。
# 问句一律交给模型判要不要查（软话在路由那层已经挡掉）。
QUESTION = re.compile(r"[？?]|吗|来着|到底|是不是|谁|哪个|哪里|什么时候|第一次|最开始|怎么回事")

# 当前句只是转述别人一句玩笑时，不该把玩笑里的某个动词当成检索请求。
# 2026-10-13 实例：转述某个 AI 的一句玩笑，曾因其中的一个字撞词塞进另一件事。
# 曾因「屁股」撞词塞进 9/28 的另一件事。
RELAY_SPEAKER = re.compile(
    r"^\s*(?:" + "".join(persona.alt(t) + "|" for t in persona.RELAY_SPEAKERS) + r"deepseek|gemini|grok)\s*[：:]",
    re.I,
)
PLAYFUL_RELAY = re.compile(
    r"(调情|玩笑|黄梗|🌚|🤡|呗[？?]?$)"
)

# 这些短回复自己没有检索锚点。仅在 transcript 里有最近对话时补上下文。
LOW_INFO_REPLY = re.compile(
    r"^\s*(?:做?好了|弄好了|完成了|可以了|现在呢|然后呢|看到了?|"
    r"就是这个|对[的吧]?|嗯+|行了?|好啦)[呀啊吧呢嘛~～。！!？?\s\U0001F300-\U0001FAFF]*$"
)
IMAGE_MARK = re.compile(r"(?:image_path\s*=|<image\b|\[image\]|\[图片\])", re.I)
CONTEXT_TURNS = 6
CONTEXT_CHARS = 900

# 同词不等于同一件物品。这里放经纠正确认的「强实体排斥」。
# 来自配置 ENGRAM_ENTITY_GUARDS（「线索1|线索2=>候选里不许出现的词;…」）。
ENTITY_GUARDS = persona.ENTITY_GUARDS


# ── 她纠正过我的那些（corrections.py 写进来的，user_pinned + confirmed）
# 库里八千条，检索那一步根本捞不到这几十条。单独按字面二元组重合捞，
# 捞到的排在候选最前面，交给模型精排判对不对得上。
def pinned_candidates(con, text: str, k: int = 3) -> list[dict]:
    try:
        rows = con.execute(
            "SELECT id, ext_id, subject, content, status, user_pinned FROM facts"
            " WHERE user_pinned=1 AND status='confirmed' AND valid_to IS NULL").fetchall()
    except Exception:                                        # noqa: BLE001
        return []
    q = {text[i:i + 2] for i in range(len(text) - 1)} - {"其实", "我以", "以为"}
    scored = []
    for row in rows:
        c = row["content"] or ""
        overlap = sum(1 for g in q if g in c)
        if overlap >= 2:
            scored.append((overlap, dict(row)))
    scored.sort(key=lambda x: -x[0])
    return [d | {"fact_id": d["id"]} for _, d in scored[:k]]


def distinctive_terms(text: str) -> list[str]:
    found = []
    for m in DISTINCTIVE.finditer(text):
        w = m.group(1)
        if w.lower() in STOP_EN:
            continue
        if w not in found:
            found.append(w)
    return found


def keyword_relevant(marks: list[str], content: str) -> bool:
    """关键词闸（精排模型挂了时的退路）。

    2026-10-28 收紧：原来命中一个 STRONG（人名/地名/日期）就算相关，库里含「助手」
    的记录有几千条，她提一句人名几乎什么都「相关」。现在要两个词才算：
    两个 STRONG，或一个 STRONG + 一个普通词（两个普通词本来就算）。
    """
    body = content.lower()
    hit = [m for m in marks if m.lower() in body]
    n_strong = sum(1 for m in hit if STRONG.fullmatch(m))
    n_other = len(hit) - n_strong
    return n_strong >= 2 or (n_strong >= 1 and n_other >= 1) or n_other >= 2


def pre_recall_skip(text: str) -> str | None:
    """检索前的高精度否决；拿不准就放行给 Router。"""
    if (RELAY_SPEAKER.search(text) and PLAYFUL_RELAY.search(text)
            and not EXPLICIT_RECALL.search(text)):
        return "转述／玩笑，不翻旧账"
    return None


def _entry_text(entry: dict) -> str:
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _clean_context_text(text: str) -> str:
    bodies = [m.group("body").strip() for m in RE_CHANNEL.finditer(text)]
    if bodies:
        text = bodies[-1]
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def recent_dialogue_context(data: dict, current: str) -> list[str]:
    """从本会话 transcript 取当前轮之前最近几条对话；失败就返回空。"""
    path = data.get("transcript_path")
    if not path or not os.path.isfile(path):
        return []
    rows = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                except Exception:                              # noqa: BLE001
                    continue
                if entry.get("isSidechain") or entry.get("type") not in ("user", "assistant"):
                    continue
                body = _clean_context_text(_entry_text(entry))
                if not body or SYNTHETIC.search(body):
                    continue
                rows.append(body[:260])
    except Exception:                                         # noqa: BLE001
        return []

    # 某些版本会在 hook 前把当前消息写入 transcript；避免重复。
    if rows and rows[-1].strip() == current.strip():
        rows.pop()
    picked, used = [], 0
    for body in reversed(rows):
        if used + len(body) > CONTEXT_CHARS:
            body = body[:max(0, CONTEXT_CHARS - used)]
        if body:
            picked.append(body)
            used += len(body)
        if len(picked) >= CONTEXT_TURNS or used >= CONTEXT_CHARS:
            break
    return list(reversed(picked))


def contextual_query(data: dict, current: str) -> tuple[str, bool]:
    """低信息回复/带图输入借最近对话补锚点；普通句子保持原样。"""
    raw = data.get("prompt") or ""
    if not (LOW_INFO_REPLY.search(current) or IMAGE_MARK.search(raw)):
        return current, False
    context = recent_dialogue_context(data, current)
    if not context:
        return current, False
    return "\n".join(context + [current]), True


def entity_compatible(query: str, candidate: str) -> bool:
    """强实体冲突时宁可不注；候选明确被当前句点名则不拦。"""
    q = query.lower()
    c = candidate.lower()
    for guard in ENTITY_GUARDS:
        forbidden = guard["candidate_forbidden"].lower()
        if (forbidden in c and forbidden not in q
                and any(cue.lower() in q for cue in guard["query_any"])):
            return False
    return True

CHOICE_ACTION = re.compile(r"(点个|点一|吃|喝|买|选|穿|用|带|要不要|可以.+吗)")
PREFERENCE_CUE = re.compile(
    r"(喜欢|不喜欢|爱吃|要配|不要|偏好|习惯|只要|最爱|配(?:米饭|饭)|要(?:椒盐|辣|甜|咸|冰|热))")
CONTEXT_ENTITY_MARKS = set(persona.CONTEXT_MARKS)


def _cjk_bigrams(text: str) -> set[str]:
    out = set()
    for chunk in re.findall(r"[一-鿿]{2,}", text):
        out.update(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return out


def deterministic_relevance(query: str, candidate: str) -> bool:
    """给两类模型容易误杀的高精度命中兜底。"""
    if not entity_compatible(query, candidate):
        return False
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    current = lines[-1] if lines else query
    if LOW_INFO_REPLY.search(current) and len(lines) > 1:
        context = "\n".join(lines[:-1])
        for mark in distinctive_terms(context):
            if STRONG.fullmatch(mark) and mark.lower() in candidate.lower():
                return True
    # 「用」「买」也出现在回忆旧事的话里（「我之前用的那个箱子不是扔了吗」），
    # 那时她在问发生过的事，不是在做选择，偏好记录不该被强塞（10/23 箱子题）。
    if CHOICE_ACTION.search(current) and PREFERENCE_CUE.search(candidate) \
            and not EXPLICIT_RECALL.search(current):
        overlap = _cjk_bigrams(current) & _cjk_bigrams(candidate)
        if overlap - {"可以", "要不", "不要"}:
            return True
    return False


def contextual_candidates(con, query: str, k: int = 3) -> list[dict]:
    """短回复补了上文后，按少量明确项目名补捞候选。"""
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) < 2 or not LOW_INFO_REPLY.search(lines[-1]):
        return []
    marks = [m for m in distinctive_terms("\n".join(lines[:-1]))
             if m in CONTEXT_ENTITY_MARKS]
    found = []
    seen = set()
    for mark in marks:
        try:
            rows = con.execute(
                "SELECT id, ext_id, subject, content, status, user_pinned FROM facts"
                " WHERE status IN ('candidate','confirmed') AND valid_to IS NULL"
                " AND content LIKE ? ORDER BY user_pinned DESC, id DESC LIMIT ?",
                (f"%{mark}%", k),
            ).fetchall()
        except Exception:                                     # noqa: BLE001
            continue
        for row in rows:
            item = dict(row)
            if item["ext_id"] not in seen:
                found.append(item | {"fact_id": item["id"]})
                seen.add(item["ext_id"])
    return found[:k]


def main() -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0

    text = incoming(data)
    if not text or len(text) > 2000:
        return 0
    skipped = skip_input(text)
    if skipped:
        log(f"不查 | {text[:40]} | {skipped}")
        return 0

    pre_skipped = pre_recall_skip(text)
    if pre_skipped:
        log(f"不查 | {text[:40]} | {pre_skipped}")
        return 0

    search_text, used_context = contextual_query(data, text)
    search_text = strip_wrappers(search_text)
    if not search_text:
        log(f"不查 | {text[:40]} | 纯附件/通知包装，没有可检索的话")
        return 0

    # 没有指向性的词就别查——除非她明确要求回忆
    marks = distinctive_terms(search_text)
    if not marks and not EXPLICIT_RECALL.search(text) and not QUESTION.search(text):
        log(f"不查 | {text[:40]} | 无指向性词")
        return 0

    sys.path.insert(0, ENGRAM)
    try:
        from retrieve import log_recall, retrieve
        from store.store import connect
    except Exception as e:                                   # noqa: BLE001
        log(f"import 失败：{e}")
        return 0

    # 遥测：库里查到了 ≠ 上下文里用上了。这条路后面还有两道闸，
    # 所以每条出口都自报最终决定，别让被闸掉的那些混进成功召回。
    # 任何记账失败都吞掉——遥测坏了不能把这次对话也带崩。
    def tel(r, decision, reason, selected=None):
        try:
            log_recall(con, text, r, decision=decision, reason=reason,
                       selected=selected)
        except Exception as e:                               # noqa: BLE001
            log(f"遥测失败：{e}")

    try:
        con = connect()
        r = retrieve(con, search_text, who=None)
    except Exception as e:                                   # noqa: BLE001
        log(f"召回失败：{e}")
        return 0

    pinned = pinned_candidates(con, search_text)
    contextual = contextual_candidates(con, search_text)
    if not r.should_recall and (not pinned and not contextual or "软话" in (r.reason or "")):
        tel(r, "skip", f"router:{r.reason}")
        log(f"不查 | {text[:40]} | {r.reason}")
        return 0

    # 按意思找（vectors.py）：只在路由判了要翻旧账时才加，补「换个说法就对不上」的漏。
    # 放在关键词结果前面，免得被精排的 12 条上限截掉；判对判错仍交给精排。
    vec = []
    # 混合召回（默认，任务二·步骤3）已经把向量候选合并进 r.hits 了，这里不重复调一次嵌入接口；
    # 只有回滚到旧分层路径（ENGRAM_LAYERED_RETRIEVE=1，r.mode=="layered"）才在这里补向量。
    if r.should_recall and getattr(r, "mode", "layered") != "hybrid":
        try:
            from vectors import search as vsearch
            vec = vsearch(con, search_text, k=4, floor=0.55)
        except Exception as e:                               # noqa: BLE001
            log(f"向量失败：{e}")
    seen_ids = {p["ext_id"] for p in pinned + contextual}
    vec = [v for v in vec if v["ext_id"] not in seen_ids]
    seen_ids |= {v["ext_id"] for v in vec}
    r.hits = pinned + contextual + vec + [h for h in (r.hits if r.should_recall else [])
                                          if h.get("ext_id") not in seen_ids]
    try:                                                     # 防回声（task2-08）：给每条候选打「只有转述」标记
        import echo
        echo.annotate(con, r.hits)
        # 我自己说过的话（「回宿舍再说」这类）不是关于她的事实：稳定地排到她本人的话和她的纠正后面（任务四·步骤1B）
        echo.demote_say(r.hits)
    except Exception as e:                                   # noqa: BLE001
        log(f"转述标注失败：{e}")
    if not r.hits:
        # 这一句是整个 hook 最重要的一行：查过了、没有。
        # 有了它，我就没有理由再凭空说一个听起来对的答案。
        if EXPLICIT_RECALL.search(text):
            print("【记忆库】已查，没有可靠记录；别猜，直说想不起来或问她。")
        tel(r, "skip", "no_hits")
        log(f"查了没有 | {text[:40]} | 词{r.terms}")
        return 0

    # ── 当前情境相关性闸（2026-09-19 加，止血版）
    # 用户框架里的动态显著性公式有「当前情境相似度」「当前目标相关性」两项，
    # 之前一项都没做：retrieve 的分数主要是激活度，跟她眼下在说什么不挂钩。
    # 实测：问某个功能为什么时有时无，注进来的是一条无关的旧闲聊（0.947 分）；
    # 让查某个模型的摘要，注进来的是另一个无关的游戏话题（只因为都带同一个词）。
    # 好坏两类分数混在 0.7~1.0 里，按分数切不开。
    # 所以按她这句话里的指向性词来判：记录里至少要出现其中两个（她只给了一个就要那一个）。
    # 路由切出来的碎片词（"天晚上""知道变通"）不算，只认 DISTINCTIVE 抓到的。
    # 这是止血，正式的路由按框架第一步 fork BMAM 再做。
    # 专名／地名／日期一个就够；「记忆库」「flash」这种话题词、英文词要凑够两个。
    def relevant(h):
        return keyword_relevant(marks, h.get("content") or "")
    # 先让模型判（BMAM 的前额叶精排，见 engram/rerank.py）：要不要翻旧账、每条对不对得上。
    # 7 分以上才算对得上，按分排。模型挂了、超时了，才退回上面那道关键词闸。
    how = "关键词闸"
    try:
        from rerank import judge
        j = judge(search_text, [echo_prefix(h) + (h.get("content") or "") for h in r.hits])
    except Exception:                                        # noqa: BLE001
        j = None
    if j is not None:
        forced = [
            i for i, h in enumerate(r.hits)
            if deterministic_relevance(search_text, h.get("content") or "")
        ]
        how = "模型精排"
        # 2026-10-18 正则兜底：她明说「还记得/之前/上次」时，模型判不用翻也不算数。
        #   当天漏过一次：「你还记得之前说的建个相册吗」被判 rerank_no_need。
        explicit = bool(EXPLICIT_RECALL.search(text))
        if not j["need"] and not forced and explicit:
            how = "模型精排+明确回忆兜底"
            scored = sorted(zip(j["scores"], range(len(r.hits))), reverse=True)
            forced = [i for sc, i in scored if sc >= 5][:3]
        if not j["need"] and not forced:
            tel(r, "skip", f"rerank_no_need:{len(r.hits)}")
            log(f"模型判不用翻旧账 | {text[:40]} | {len(r.hits)}条")
            return 0
        if forced:
            how = "模型精排+高精度兜底"
            on_topic = [r.hits[i] for i in forced]
        else:
            # 只有转述的排序 × 1/3（阈值仍按模型原分，被明确问到的转述照样能进，只是排在本人的话后面）
            scored = sorted(zip(j["scores"], range(len(r.hits))),
                            key=lambda t: (t[0] * (ECHO_W if (r.hits[t[1]].get("echo_from")
                                                              or r.hits[t[1]].get("assistant_say")) else 1.0),
                                           t[0], t[1]), reverse=True)
            on_topic = [r.hits[i] for sc, i in scored
                        if sc >= 7 and entity_compatible(
                            search_text, r.hits[i].get("content") or "")]
    else:
        on_topic = [h for h in r.hits
                    if (relevant(h) or deterministic_relevance(
                        search_text, h.get("content") or ""))
                    and entity_compatible(search_text, h.get("content") or "")] if marks else []
    if not on_topic:
        if EXPLICIT_RECALL.search(text):
            print(persona.get("ENGRAM_MISS_HINT") or "【记忆库】没勾到。她在说旧事：先自己翻（聊天原始记录、会话存档、engram evidence），翻不到再说想不起来或问她，别猜。")
        tel(r, "skip", f"off_topic:{len(r.hits)}")
        log(f"查到但不相关，不注（{how}） | {text[:40]} | 指向词{marks} | {len(r.hits)}条")
        return 0

    # 刚注进去的不重复注。判的是「这条记录」，不是「这个词」。
    fresh = recently_injected(con)
    hits = [h for h in on_topic if h.get("ext_id") not in fresh]
    if not hits:
        tel(r, "skip", f"recently_injected:{len(r.hits)}")
        log(f"刚注过，不重复 | {text[:40]} | 词{r.terms} | {len(r.hits)}条")
        return 0

    lines = [f"【记忆库·{r.task_type}】线索，非结论；当规则执行前先核对来源与适用范围。"]
    if r.conflicts:
        lines.append("⚠️ 有打架的记录：" + "；".join(r.conflicts)
                     + "。按时间新的那条为准，并且要说清有两条。")
    used = len(lines[0])
    injected = set()
    # 新事盖旧事（不改库）：给要注入的每条找一条「让它不再成立」的更新记录，一起给。
    chosen = hits[:MAX_HITS]
    newer_of: dict = {}
    try:
        from concurrent.futures import ThreadPoolExecutor
        from newer import superseded_by
        for h in chosen:                                     # 库里已有「可能过时」标记的，直接用，不再调模型
            n = h.get("stale_new")
            if n and n["ext_id"] not in {c["ext_id"] for c in chosen}:
                newer_of[h["ext_id"]] = n
        todo = [h for h in chosen if h["ext_id"] not in newer_of]
        with ThreadPoolExecutor(max_workers=len(todo) or 1) as ex:
            futs = {h["ext_id"]: ex.submit(superseded_by, con, h) for h in todo}
            for k, f in futs.items():
                try:
                    n = f.result(timeout=8)
                except Exception:                            # noqa: BLE001
                    n = None
                if n and n["ext_id"] not in {c["ext_id"] for c in chosen}:
                    newer_of[k] = n
    except Exception as e:                                   # noqa: BLE001
        log(f"新旧比对失败：{e}")
    budget = MAX_CHARS + (200 if newer_of else 0)
    if any(h.get("echo_from") for h in chosen):
        lines.append("〔转述〕＝别人转述的话，不是该主体本人说过的；不要当成她自己的话、偏好或经历来复述。")
        used += len(lines[-1])
    if any(h.get("assistant_say") for h in chosen):
        lines.append("〔我说的〕＝我自己说过的话，不是她的事实；不要拿它推断她现在在哪、在做什么。")
        used += len(lines[-1])
    for h in chosen:
        mark = STATUS_MARK.get(h.get("status") or "", h.get("status") or "?")
        if h.get("user_pinned"):
            mark += "·钉"
        if h.get("my_stance") == "disown":
            mark += "·我已不认"
        sid = (h.get("ext_id") or "")[5:13]          # 去掉 fact_ 前缀，够我回去查全条
        nw = newer_of.get(h["ext_id"])
        if nw:
            mark += "·可能已过时"
        piece = f"· [{h['subject']}·{mark} {sid}]{echo_prefix(h)} {clip(h['content'] or '')}"
        if nw:
            piece += (f"\n  ⤷ 更新（{(nw.get('valid_from') or '')[:10]}，"
                      f"{nw['ext_id'][5:13]}）：{clip(nw['content'] or '')}——以这条为准")
        if used + len(piece) > budget:
            break
        lines.append(piece)
        used += len(piece)
        injected.add(h["ext_id"])
        if nw:
            injected.add(nw["ext_id"])
    lines.append("（库内记录；冲突时核对原文。）")
    # 预算在上面逐条算过了，这里不再按字数硬切：硬切会把最后一条和脚注切在半句。
    # 出口兜底（没有开关）：库里老证据/老事实里要是还留着密钥，也不许进对话上下文。
    # 扫描本身出任何问题就一个字都不注入（宁可少说，不可漏钥）。
    try:
        from redact import scrub
        out_text = scrub("\n".join(lines), "hook")
    except Exception:                                        # noqa: BLE001
        return 0
    print(out_text)
    tel(r, "recall", f"injected:{len(injected)}/{len(r.hits)}", selected=injected)
    log(f"召回 {len(r.hits)} 条、注入 {len(injected)} 条（{how}）"
        f"{'（补了最近对话）' if used_context else ''}"
        f"（跳过刚注过的 {len(r.hits) - len(hits)} 条）"
        f" | {text[:40]} | 词{r.terms} | 层{r.layers_read}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                        # noqa: BLE001
        sys.exit(0)
