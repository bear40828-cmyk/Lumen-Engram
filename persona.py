"""部署相关的「人」配置：说话人标签、称呼、TG 号、路径——代码里不写死任何真人信息，全部从这里读。

读取顺序（先到先得）：
  1. 环境变量 ENGRAM_*
  2. persona.env 文件（默认放在代码同目录，已 .gitignore；可用 ENGRAM_PERSONA_FILE 指到别处）
  3. ENGRAM_ENV_FILE 指向的 .env（和 llm.py / flags.py 用的是同一个文件，缺省 .env）
都没配时用下面的通用默认值（user / 用户 …），代码照常能跑，只是认不出真实的说话人标签。
读一次、缓存；测试里要换配置就改环境变量后调 persona.reload()。

库里 evidence.source_actor / facts.subject 存的是「标签」（USER / claude / gpt / AGENT_C）。
标签必须和已存数据一致，所以真实部署必须把 ENGRAM_ACTOR_USER / ENGRAM_ACTOR_AGENT_C 配成库里的值；
称呼（NAME_*）是正文里出现的叫法（「X：…」前缀、提示词里的名字），同样要和已存正文一致。
claude / gpt 是产品名，不算个人信息，标签固定不配置。

配置项（全部可选，括号里是默认值）：
  ENGRAM_ACTOR_USER        人类说话人的库内标签（user）
  ENGRAM_ACTOR_AGENT_C     第三个 AI 的库内标签（agent_c）
  ENGRAM_NAME_USER         人类的称呼（用户）
  ENGRAM_USER_FULLNAME     人类的全名，只在提示词里写一次（空）
  ENGRAM_USER_ALIASES      人类的其他叫法，逗号分隔，按顺序拼进正则（空）
  ENGRAM_NAME_CLAUDE       Claude 侧 AI 的称呼（助理）
  ENGRAM_NAME_GPT          GPT 侧的称呼（助手B）
  ENGRAM_NAME_AGENT_C      第三个 AI 的称呼（助手C）
  ENGRAM_DESC_GPT          提示词里对 GPT 侧的一句描述（AI）
  ENGRAM_DESC_AGENT_C      提示词里对第三个 AI 的一句描述（AI）
  ENGRAM_GPT_ALIASES       GPT 侧的其他叫法，逗号分隔（空）
  ENGRAM_CLAUDE_ALIASES    Claude 侧的其他叫法，逗号分隔（空）
  ENGRAM_OTHER_AI_TERMS    其他 AI/助手的外号或名字，逗号分隔，当「第三方」识别用（空）
  ENGRAM_ATTACH_TAG        聊天附件标记 [[<tag>-xxx|…]] 里的 <tag>（attach）
  ENGRAM_PERSON_REGEX      「人物」槽位里额外的原始正则选择支（如某个简称的带断言写法），空＝没有
  ENGRAM_PERSON_SHORT      元记忆「人物」判断里按字面匹配的简称/昵称，逗号分隔（空）
  ENGRAM_PLACE_TERMS       元记忆「地点」判断里额外的地名，逗号分隔（空）
  ENGRAM_CLASSIFY_TERMS    classify「稳定事实」里「考…（学校|X）」的 X，逗号分隔（空）
  ENGRAM_SLOT_TERMS        distill_self 事实槽位判断里额外的专名，逗号分隔（空）
  ENGRAM_ENTITY_GUARDS     钩子的「强实体排斥」：「线索1|线索2=>候选里不许出现的词;…」（空）
  ENGRAM_ROUTER_PERSON_TERMS 路由「提到人」判定里额外的称呼，逗号分隔（空）
  ENGRAM_PET_NAME          提示词示例里用的宠物名（小狗）
  ENGRAM_PROMPT_GLOSSARY_FILE  抽取提示词里「几个固定的人」那一段的正文文件（空＝不写这一段）
  ENGRAM_EXAMPLE_SHORT     统一判定提示词里「短不是理由」的示例（缺省「考某大学」「爱粉色」）
  ENGRAM_EXAMPLE_PLAYFUL   抽取提示词里的一句示例（缺省由称呼拼出）
  ENGRAM_CHANNEL_IDS       Telegram 数字 id → 标签，「id:标签,id:标签」（空）
  ENGRAM_CHANNEL_NAMES     Telegram 用户名 → 标签，「名字:标签,名字:标签」（空）
  ENGRAM_MAILBOX_ALIASES   信箱 from 字段的额外同义词，「值:标签,…」（空）
  ENGRAM_MAILBOX_LOG       信箱 messages.jsonl 的路径（空＝没有信箱，入库脚本直接退出）
  ENGRAM_SOFT_TERMS        路由「软话/叫人」豁免词，逗号分隔（如：嗯嗯,哈哈,好的）
  ENGRAM_DOMAIN_TERMS      路由/闸的「专名线索」词，逗号分隔（空）
  ENGRAM_ENTITY_TERMS      钩子里的人/物专名，逗号分隔，顺序即正则顺序（空）
  ENGRAM_TOPIC_TERMS       钩子里的话题词，逗号分隔（空）
  ENGRAM_RELAY_SPEAKERS    钩子里「某某：」开头的转述发言人名单，逗号分隔（空）
  ENGRAM_MISS_HINT         钩子「勾不到旧事」时打印的整句提示（缺省是通用提示）
  ENGRAM_GATE_TERMS        晋升门控「专名」槽位的词，逗号分隔（空）
  ENGRAM_CONTEXT_MARKS     钩子的上下文专名（会拉高记忆库命中的名词），逗号分隔（空）
  ENGRAM_KEYS_FILE         fact_key 注册表（缺省 keys.local.yaml，没有就用 keys.yaml）
  ENGRAM_ENV_FILE          LLM 密钥 .env 的路径（缺省 .env）
  ENGRAM_HOME              代码/库所在目录（缺省＝本文件所在目录）
  ENGRAM_TRANSCRIPT_DIR    会话 jsonl 目录（缺省 ~/.claude/projects）
  ENGRAM_HOOK_PATH         eval_recall 载入的钩子脚本（缺省＝同目录 hooks_engram_recall.py）
  ENGRAM_DIGEST_DIR        每日摘要目录（缺省 ~/.engram/digests）
"""
from __future__ import annotations

import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_cache: dict | None = None


def _parse(path) -> dict:
    out: dict = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k.startswith("ENGRAM_"):
            out[k] = v
    return out


def _persona_file() -> Path:
    return Path(os.environ.get("ENGRAM_PERSONA_FILE") or (HERE / "persona.env"))


def env_file() -> str:
    """LLM 密钥那份 .env 的路径：环境变量 ENGRAM_ENV_FILE > persona.env 里的 ENGRAM_ENV_FILE > 当前目录 .env。"""
    v = os.environ.get("ENGRAM_ENV_FILE")
    if v:
        return v
    return _parse(_persona_file()).get("ENGRAM_ENV_FILE") or ".env"


def _files() -> list[Path]:
    return [_persona_file(), Path(env_file())]


def reload() -> None:
    global _cache
    _cache = None
    _bind()


def get(key: str, default: str = "") -> str:
    global _cache
    v = os.environ.get(key)
    if v is not None and v != "":
        return v
    if _cache is None:
        _cache = {}
        for f in reversed(_files()):          # 先读低优先级的，高优先级的覆盖
            _cache.update(_parse(f))
    return _cache.get(key, default)


def csv(key: str, default: str = "") -> tuple[str, ...]:
    return tuple(t.strip() for t in get(key, default).split(",") if t.strip())


def pairs(key: str) -> dict[str, str]:
    return dict(x.strip().split(":", 1) for x in get(key).split(",") if ":" in x)


def alt(*terms: str) -> str:
    """把若干词拼成正则选择支（保持传入顺序，丢掉空串，转义）。"""
    return "|".join(re.escape(t) for t in terms if t)


def alt_raw(terms) -> str:
    return "|".join(re.escape(t) for t in terms if t)


def _bind() -> None:
    g = globals()
    g["USER"] = get("ENGRAM_ACTOR_USER", "user")
    g["CLAUDE"] = "claude"
    g["GPT"] = "gpt"
    g["AGENT_C"] = get("ENGRAM_ACTOR_AGENT_C", "agent_c")
    g["ACTORS"] = (g["USER"], g["CLAUDE"], g["GPT"], g["AGENT_C"])
    g["N_USER"] = get("ENGRAM_NAME_USER", "用户")
    g["N_CLAUDE"] = get("ENGRAM_NAME_CLAUDE", "助理")
    g["N_GPT"] = get("ENGRAM_NAME_GPT", "其他助手")
    g["N_C"] = get("ENGRAM_NAME_AGENT_C", "第三方助手")
    g["USER_FULL"] = get("ENGRAM_USER_FULLNAME")
    g["USER_ALIASES"] = csv("ENGRAM_USER_ALIASES")
    g["GPT_ALIASES"] = csv("ENGRAM_GPT_ALIASES")
    g["CLAUDE_ALIASES"] = csv("ENGRAM_CLAUDE_ALIASES")
    g["DESC_GPT"] = get("ENGRAM_DESC_GPT", "AI")
    g["DESC_C"] = get("ENGRAM_DESC_AGENT_C", "AI")
    g["NAMES"] = {g["USER"]: g["N_USER"], "claude": g["N_CLAUDE"], "gpt": g["N_GPT"], g["AGENT_C"]: g["N_C"]}
    g["CHANNEL_BY_ID"] = pairs("ENGRAM_CHANNEL_IDS")
    g["CHANNEL_BY_NAME"] = pairs("ENGRAM_CHANNEL_NAMES")
    g["SOFT_TERMS"] = csv("ENGRAM_SOFT_TERMS", "嗯嗯,哈哈,好的")
    g["DOMAIN_TERMS"] = csv("ENGRAM_DOMAIN_TERMS")
    g["ENTITY_TERMS"] = csv("ENGRAM_ENTITY_TERMS")
    g["TOPIC_TERMS"] = csv("ENGRAM_TOPIC_TERMS")
    g["RELAY_SPEAKERS"] = csv("ENGRAM_RELAY_SPEAKERS")
    g["GATE_TERMS"] = csv("ENGRAM_GATE_TERMS")
    g["CONTEXT_MARKS"] = csv("ENGRAM_CONTEXT_MARKS")
    g["OTHER_AI_TERMS"] = csv("ENGRAM_OTHER_AI_TERMS")
    g["PERSON_SHORT"] = csv("ENGRAM_PERSON_SHORT")
    g["PLACE_TERMS"] = csv("ENGRAM_PLACE_TERMS")
    g["CLASSIFY_TERMS"] = csv("ENGRAM_CLASSIFY_TERMS")
    g["SLOT_TERMS"] = csv("ENGRAM_SLOT_TERMS")
    g["ENTITY_GUARDS"] = tuple(
        {"query_any": tuple(c for c in cues.split("|") if c), "candidate_forbidden": forb.strip()}
        for cues, _, forb in (x.partition("=>") for x in get("ENGRAM_ENTITY_GUARDS").split(";")) if forb.strip())
    g["PERSON_RE"] = get("ENGRAM_PERSON_REGEX")
    g["ATTACH_TAG"] = get("ENGRAM_ATTACH_TAG", "attach")
    g["PET"] = get("ENGRAM_PET_NAME", "")
    g["MAILBOX_LOG"] = get("ENGRAM_MAILBOX_LOG")
    g["HOME"] = get("ENGRAM_HOME", str(HERE))


_TOKENS = (
    ("@@EX_SHORT@@", lambda: get("ENGRAM_EXAMPLE_SHORT") or "「考某大学」「爱粉色」"),
    ("@@GLOSSARY_BLOCK@@", lambda: glossary_block()), ("@@EX_PLAYFUL@@", lambda: example_playful()), ("@@PET@@", lambda: PET),
    ("@@ROSTER_USER@@", lambda: user_roster()), ("@@A_USER@@", lambda: USER), ("@@A_C@@", lambda: AGENT_C),
    ("@@USER_FULL@@", lambda: USER_FULL), ("@@USER@@", lambda: N_USER),
    ("@@CLAUDE@@", lambda: N_CLAUDE), ("@@GPT@@", lambda: N_GPT), ("@@C@@", lambda: N_C),
    ("@@DESC_GPT@@", lambda: DESC_GPT), ("@@DESC_C@@", lambda: DESC_C),
)


_ALT_RE = re.compile(r"@@alt:([A-Z_,]+)@@")


def _alt_lists() -> dict:
    return {
        "USER": [N_USER], "USER_FULL": [USER_FULL], "USER_ALIASES": list(USER_ALIASES),
        "USER_NAMES": [N_USER, *USER_ALIASES],
        "CLAUDE": [N_CLAUDE], "CLAUDE_ALIASES": list(CLAUDE_ALIASES),
        "GPT": [N_GPT], "GPT_ALIASES": list(GPT_ALIASES), "C": [N_C],
        "OTHER_AI": list(OTHER_AI_TERMS), "PET": [PET], "A_USER": [USER], "A_C": [AGENT_C],
    }


def _alt_sub(m) -> str:
    lists = _alt_lists()
    terms: list[str] = []
    for key in m.group(1).split(","):
        terms.extend(lists[key])
    return alt(*terms) or "(?!)"          # 空 → 永不匹配（不能留空选择支，否则匹配一切）


def fill(text: str) -> str:
    """把提示词/正则里的占位符换成配置里的称呼/标签。
    普通占位符 @@USER@@ 等原样替换（提示词用）；@@alt:USER_NAMES,CLAUDE@@ 这种展开成转义后的正则选择支
    （正则用；某一项没配就丢掉，全空则变成永不匹配的 (?!)）。"""
    text = _ALT_RE.sub(_alt_sub, text)
    for tok, fn in _TOKENS:
        text = text.replace(tok, fn())
    return text


def mask_re() -> str:
    """报告里要打码的名字（称呼、别名、其他 AI、宠物名）拼成的正则。"""
    return persona_alt()


def persona_alt() -> str:
    return alt(N_USER, *USER_ALIASES, N_CLAUDE, N_GPT, N_C, *OTHER_AI_TERMS, "GPT", "GLM", PET)


def person_re() -> str:
    """额外的人物正则选择支，前面自带一个 | ；没配返回空串。"""
    return ("|" + PERSON_RE) if PERSON_RE else ""


def glossary_block() -> str:
    """抽取提示词里「几个固定的人（原话里常这么叫）」一段。正文放在 ENGRAM_PROMPT_GLOSSARY_FILE 指向的文件里（不进仓库）；没配就整段不写。"""
    p = get("ENGRAM_PROMPT_GLOSSARY_FILE")
    if not p:
        return ""
    try:
        body = Path(p).read_text(encoding="utf-8").rstrip("\n")
    except OSError:
        return ""
    return ("几个固定的人（原话里常这么叫）：\n" + body + "\n") if body.strip() else ""


def example_playful() -> str:
    return get("ENGRAM_EXAMPLE_PLAYFUL") or f"{N_USER}逗{N_C}要亲亲被拒，{N_CLAUDE}看到后吃醋"


def user_roster() -> str:
    """提示词里写人类的那一小段：称呼，有全名就加「（全名）」之类由调用方决定；这里只给「称呼（全名，人类）」。"""
    return f"{N_USER}（{USER_FULL}，人类）" if USER_FULL else f"{N_USER}（人类）"


_bind()
