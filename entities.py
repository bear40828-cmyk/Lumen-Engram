#!/usr/bin/env python3
"""Lumen Engram · 专名表（自动维护）与专名直通（召回改造·第 1 层，2026-11-04）

痛点（2026-10-31 实测）：一句提到叠字小名的话
——库里这个名字有 11 条，FTS 一搜全中，却死在最前面的规则路由（判「无指向性词」）。
专名表是手写的 ENGRAM_ENTITY_TERMS 追不上用户聊到的新名字，所以改成**从库里自动长出来**。

借鉴（只借想法与打分公式，代码自己写；协议见各条）：
  · mem0（Apache-2.0）mem0/memory/main.py `_compute_entity_boosts` + utils/scoring.py：
    「实体 → 挂着它的记忆」的链接，命中实体的记忆在检索分上加分；实体挂的记忆越多越不稀奇，
    加分按 memory_count_weight = 1/(1+0.001·(n−1)²) 衰减。这里原样用这个公式（见 spread_weight）。
  · mem0 utils/entity_extraction.py：实体三类——专有名词 / 引号里的词 / 名词短语。这里对应 named / title+quoted / oov。
  · jieba（MIT，可选依赖）：词典外的高频词 + 词性 nr/ns/nt/nz 当候选；没装就只用下面的规则。

专名从哪来（kind）：
  config  人设配置里手写的 ENGRAM_ENTITY_TERMS / 名字 / 宠物名       →  直通
  title   《书名》「作品名」                                             →  直通
  quoted  「引号里」的 2~10 字词                                          →  软（引号也常是强调）
  ident   英文标识符（项目/型号/人名），≥3 位，非停用词                  →  直通
  named   「叫X」「名叫X」「取名X」「外号X」「昵称X」                    →  直通（一次就够：人起名字的句式本身就是证据）
  redup   叠字名（如某个叠字小名），在 ≥2 条事实里出现，不在叠字常用词表      →  直通
  tag     jieba 词性 nr/ns/nt/nz，≥2 条事实                                →  直通
  oov     词典外的词，≥3 条事实（噪音多）                                 →  软：只给检索加分，不直通
「直通」还要求出现在 ≤ MAX_DIRECT_DF 条事实里：满库都是的词（她的昵称、主要对话对象）命中了也没信息，
不触发直通，交给 FTS+向量+精排。

表 entity_index 是**派生表**：随时可以重建；重建只 upsert，不物理删除（失效的专名置 direct=0、df=0）。
钩子只读不写；建表/刷新由维护脚本做（拿 .ingest.lock）：
    python3 entities.py --db /path/engram.db --stats      # 看现状（只读）
    python3 entities.py --db /path/engram.db --refresh    # 重建/增量刷新
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MAX_DIRECT_DF = 80         # 出现在这么多条事实以上的词（她的昵称、常聊的另一个 AI），命中了也没信息量，不直通，走 FTS+向量+精排
# 自动发现的专名精度不如手写，直通的上限更低（再多就更像普通词）。
MAX_DIRECT_DF_BY_KIND = {"config": 80, "title": 30, "named": 60, "ident": 80, "redup": 60, "tag": 60}
MIN_DF = {"named": 1, "title": 1, "ident": 2, "redup": 2, "tag": 2, "oov": 3, "quoted": 2, "config": 0}
DIRECT_KINDS = {"config", "title", "ident", "named", "redup", "tag"}
# 泛称/碎片：自动发现会误当专名（回放 #27「小三」）。只取消直通，仍可当软词。config（人工配置）不受影响。
GENERIC_BLOCK = frozenset({
    "小三", "男主", "女主", "男配", "女配", "对象", "老公", "老婆", "男友", "女友", "男朋友", "女朋友",
    "前任", "宝妈", "养女儿", "饲养员", "两个人", 
    "自己选", "分阶段", "外包", "对人", "并留下", "今天下午", "今天上午", "张图", "张牌", "张卡",
    "塞进", "里塞", "顺嘴", "重连", "直连", "新加", "松口", "库中", "别光", "查岗", "满屏", "续发",
    "爱喝", "卡死", "加粗", "旧版本", "查时", "拉满", "真香", "成真", "里藏", "信自", "双休",
    "高潮时", "全留", "四楼", "错名字", "大点声", "多大声", "没赶上", "守规矩", "时推送", "对齐做",
    "结构分析", "一声试试", "频率异常", "一只桌子", "爱爱", "亲亲", "发牌", "讨封", "战术卖惨",
    "提前埋伏", "叫顺口", "逐字硬凑", "情绪向量", "吃醋巡检", "群主咬", "攻略小克",
})
# 前缀是量词/介词/虚词的 2~3 字自动发现词，多半是句子碎片
_FRAG_PREFIX = ("里", "从", "成", "别", "并", "对", "时", "叫", "谁", "两个", "一声", "一只", "多", "没")


def is_generic(term: str, kind: str = "") -> bool:
    if kind == "config":
        return False
    if term in GENERIC_BLOCK:
        return True
    return kind in ("named", "tag") and len(term) <= 3 and term.startswith(_FRAG_PREFIX)
# 专名里不该出现的字（代词/助词/连词/介词）：含这些字的「叫X」「叠字」多半是句子碎片，不是名字。
NOT_NAME_CHARS = set("我你她他它们的了是在就也都不有吗呢吧啊嘛么这那还把被让给和跟与及或但而之以为说想要会能可很太又再去来到过着得地")
# 叠字名前面常有的称呼线索（女儿XX、小狗XX、带着XX）；没有线索的叠字多半是量词/动词重叠（块块、画画、逛逛）。
REDUP_CUE = re.compile(r"(?:叫|名叫|女儿|儿子|孩子|姐姐|妹妹|弟弟|哥哥|小狗|小猫|狗狗|猫猫|宠物|闺蜜|室友|同学|朋友|老师|带着|陪着|找|抱|想|爱|养|叫做)$")

# 叠字里常见的非专名（副词/亲属称呼）。亲属称呼不直通：她每天都会说「妈妈」。
REDUP_STOP = set("""天天 慢慢 好好 常常 渐渐 刚刚 偷偷 默默 悄悄 轻轻 稍稍 狠狠 紧紧 深深 静静 微微 往往 统统 仅仅 纷纷
每每 individually 次次 句句 件件 条条 样样 个个 人人 家家 处处 时时 步步 一一 样样 渐渐 稳稳 满满 淡淡 白白 乖乖 暗暗
匆匆 悠悠 隐隐 频频 连连 屡屡 究竟 宝宝 妈妈 爸爸 哥哥 姐姐 弟弟 妹妹 爷爷 奶奶 姥姥 公公 婆婆 舅舅 姑姑 叔叔 阿姨
宝贝 娃娃 星星 谢谢 看看 想想 试试 聊聊 说说 等等 走走 玩玩 问问 查查 改改 吃吃 睡睡 算算 笑笑 哈哈 呵呵 嘻嘻 嘿嘿
呜呜 哇哇 啊啊 嗯嗯 对对 好像 仿佛 种种 丝丝 点点 层层 重重 年年 岁岁 夜夜 日日 月月 久久 远远 近近 早早 晚晚
大大 小小 多多 少少 快快 高高 长长 短短 红红 白白 瞬间""".split())

# 命名句式：叫X / 名叫X / 取名X / 起名X / 昵称X / 外号X / 绰号X。X 取 2~4 个连续汉字或字母数字。
RE_NAMED = re.compile(
    r"(?:名叫|叫做|叫作|取名叫|起名叫|叫(?![醒停住来起出人声法了过着去回我你她他它])|取名为?|起名为?|命名为|昵称是?|外号是?|绰号是?|名字是|取名叫)"
    r"[「『“\"'‘]?([一-鿿A-Za-z0-9]{2,4})")
RE_TITLE = re.compile(r"《([^《》\n]{2,20})》")
RE_QUOTED = re.compile(r"[「『“\"‘']([^「」『』“”\"‘’'\n]{2,10})[」』”\"’']")
RE_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{2,}")
RE_REDUP = re.compile(r"([一-鿿])\1")
NAME_TAIL_STOP = set("的了是在和就也都不有吗呢吧啊嘛么")     # 命名句式里被带进来的尾巴字，按位置从尾部剥掉
IDENT_STOP = {
    "claude", "code", "http", "https", "json", "text", "post", "photo", "document", "image", "images", "workspace",
    "ubuntu", "home", "processed", "process", "notification", "message", "messages", "user", "assistant", "system",
    "file", "files", "path", "true", "false", "null", "none", "the", "and", "for", "with", "this", "that", "from",
    "have", "not", "are", "was", "you", "but", "can", "web", "app", "api", "key", "bug", "fix", "tag", "log",
    "test", "tests", "data", "list", "type", "name", "item", "node", "port", "mode", "flag", "step", "task",
    "done", "tool", "result", "sticker", "voice", "video", "audio", "caption", "reply", "wake", "pending",
}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def ensure_table(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS entity_index (
        term TEXT PRIMARY KEY, kind TEXT NOT NULL, direct INTEGER NOT NULL DEFAULT 0,
        df INTEGER NOT NULL DEFAULT 0, weight REAL NOT NULL DEFAULT 1.0,
        updated_at TEXT NOT NULL)""")


def table_exists(con) -> bool:
    try:
        return bool(con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_index'").fetchone())
    except Exception:                                        # noqa: BLE001
        return False


def spread_weight(df: int) -> float:
    """借自 mem0 main.py `_compute_entity_boosts`：memory_count_weight = 1/(1+0.001·(n−1)²)。
    挂着的记忆越多，这个专名越不稀奇。"""
    n = max(int(df), 1)
    return 1.0 / (1.0 + 0.001 * ((n - 1) ** 2))


def _strip_name_tail(s: str) -> str:
    s = s.strip()
    while len(s) > 2 and s[-1] in NAME_TAIL_STOP:
        s = s[:-1]
    return s


def _live_facts(con) -> list[tuple[int, str]]:
    return [(r[0], r[1] or "") for r in con.execute(
        "SELECT id, content FROM facts WHERE valid_to IS NULL AND status IN ('candidate','confirmed')")]


def _config_terms() -> set[str]:
    out: set[str] = set()
    try:
        import persona
        for t in (*persona.ENTITY_TERMS, persona.N_USER, persona.N_CLAUDE, persona.N_GPT, persona.N_C,
                  persona.PET, *getattr(persona, "USER_ALIASES", ()), *getattr(persona, "OTHER_AI_TERMS", ())):
            if t and len(t) >= 2:
                out.add(t)
    except Exception:                                        # noqa: BLE001
        pass
    return out


def discover(facts: list[tuple[int, str]], *, use_jieba: bool = True, extra_terms=()) -> dict[str, dict]:
    """从事实正文里发现专名。返回 {term: {kind, fids:set}}。同一个词被多种方式发现，kind 取最强的（config>title>named>ident>tag>redup>quoted>oov）。"""
    order = ["config", "title", "named", "ident", "tag", "redup", "quoted", "oov"]
    found: dict[str, dict] = {}

    def add(term: str, kind: str, fid: int | None):
        if not term or len(term) < 2:
            return
        e = found.setdefault(term, {"kind": kind, "fids": set(), "hit_kinds": set()})
        e["hit_kinds"].add(kind)
        if order.index(kind) < order.index(e["kind"]):
            e["kind"] = kind
        if fid is not None:
            e["fids"].add(fid)

    for t in set(_config_terms()) | set(extra_terms):
        add(t, "config", None)
    for fid, txt in facts:
        for m in RE_TITLE.finditer(txt):
            add(m.group(1).strip(), "title", fid)
        for m in RE_NAMED.finditer(txt):
            raw = m.group(1)
            cut = next((i for i, ch in enumerate(raw) if ch in NOT_NAME_CHARS), len(raw))
            w = _strip_name_tail(raw[:cut])                  # 名字到第一个助词/代词为止（「叫小花的猫」→ 小花）
            if not w or len(w) < 2:
                continue
            if w.isascii() and w.lower() in IDENT_STOP:
                continue
            tail = raw[cut:cut + 1] or txt[m.end(1):m.end(1) + 1]
            delim = (cut < len(raw) or tail == "" or tail in "，。、；：！？!?,.;:）)」』”\"'’ \n")
            # 名字后面紧跟标点/句末才算「叫X」的 X 完整取到了（避免「叫剁椒鱼……」这种取半截）；否则要在别处也出现过（df≥2，后面 refresh 里数）
            add(w, "named" if delim else "oov", fid)
        for m in RE_QUOTED.finditer(txt):
            add(m.group(1).strip(), "quoted", fid)
        for m in RE_IDENT.finditer(txt):
            w = m.group(0).rstrip(".-_")
            if len(w) < 3 or w.lower() in IDENT_STOP:
                continue
            # 纯小写英文词多半是代码/日志里的普通单词（eat/build/color），只给软分；
            # 带大写/数字/连字符/下划线的（Python、ChatGPT、galatea-garden、user_id）才直通。
            strong = bool(re.search(r"[A-Z0-9_\-]", w)) and not w.isdigit()
            add(w, "ident" if strong else "oov", fid)
        for m in RE_REDUP.finditer(txt):
            w = m.group(0)
            # 叠字要独立成词：前后不是同一个字（排除「慢慢慢」「哈哈哈」这种拉长）
            i = m.start()
            if (i > 0 and txt[i - 1] == w[0]) or (i + 2 < len(txt) and txt[i + 2] == w[0]):
                continue
            if w in REDUP_STOP or w[0] in NOT_NAME_CHARS:
                continue
            add(w, "redup" if REDUP_CUE.search(txt[max(0, i - 4):i]) else "oov", fid)
    if use_jieba:
        try:
            import jieba
            import jieba.posseg as pseg
            jieba.setLogLevel(60)
            for fid, txt in facts:
                for w, flag in pseg.cut(txt):
                    if len(w) < 2 or not CJK_RE.search(w) or not all("\u4e00" <= ch <= "\u9fff" for ch in w):
                        continue
                    if any(ch in NOT_NAME_CHARS for ch in w):
                        continue
                    if flag in ("nr", "ns") and 2 <= len(w) <= 4:
                        add(w, "tag", fid)
                    elif jieba.dt.FREQ.get(w, 0) == 0:
                        add(w, "oov", fid)
        except Exception:                                    # noqa: BLE001
            pass
    return found


COMMON_FREQ = 30           # jieba 词典里词频 ≥ 这个数的是普通词（习惯/钩子/爸爸），不是专名：自动发现的直通要排除


def _common_checker(use_jieba: bool):
    if not use_jieba:
        return lambda term: False
    try:
        import jieba
        jieba.setLogLevel(60)
        jieba.initialize()
        return lambda term: jieba.dt.FREQ.get(term, 0) >= COMMON_FREQ
    except Exception:                                        # noqa: BLE001
        return lambda term: False


def refresh(con, *, use_jieba: bool = True, dry_run: bool = False) -> dict:
    """重建/增量刷新 entity_index。df 一律按「正文里含这个词的活事实条数」重数（不是发现时的条数）。"""
    t0 = time.time()
    facts = _live_facts(con)
    found = discover(facts, use_jieba=use_jieba)
    texts = [t for _, t in facts]
    is_common = _common_checker(use_jieba)
    rows = []
    for term, e in found.items():
        df = sum(1 for t in texts if term in t)
        kind = e["kind"]
        if kind != "config" and df < MIN_DF.get(kind, 2):
            continue
        direct = int(kind in DIRECT_KINDS and 0 < df <= MAX_DIRECT_DF_BY_KIND.get(kind, 0))
        if direct and is_generic(term, kind):
            direct = 0
        if direct and kind in ("named", "tag", "redup", "title") and len(term) <= 4 and is_common(term):
            direct = 0                                       # 词典里的普通词不直通，只当软专名
        rows.append((term, kind, direct, df, round(spread_weight(df), 4)))
    stat = {"facts": len(facts), "found": len(found), "kept": len(rows),
            "direct": sum(r[2] for r in rows), "seconds": round(time.time() - t0, 1)}
    if dry_run:
        stat["rows"] = rows
        return stat
    ensure_table(con)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    con.execute("UPDATE entity_index SET direct=0, df=0 WHERE 1")        # 先全部「失效」（不删），下面 upsert 把还成立的救回来
    con.executemany(
        """INSERT INTO entity_index(term, kind, direct, df, weight, updated_at) VALUES (?,?,?,?,?,?)
           ON CONFLICT(term) DO UPDATE SET kind=excluded.kind, direct=excluded.direct, df=excluded.df,
             weight=excluded.weight, updated_at=excluded.updated_at""",
        [(*r, stamp) for r in rows])
    con.commit()
    return stat


# ── 查询时 ────────────────────────────────────────────────────────────────────
_CACHE: dict = {}


def _load(con) -> dict[str, tuple]:
    """{term: (kind, direct, df, weight)}；按库连接缓存（钩子是一次性进程，缓存只给测试/回放用）。"""
    key = id(con)
    if not table_exists(con):
        return {}
    n = con.execute("SELECT COUNT(*), COALESCE(MAX(updated_at),''), COALESCE(SUM(direct),0), COALESCE(SUM(df),0)"
                    " FROM entity_index").fetchone()
    sig = tuple(n)
    c = _CACHE.get(key)
    if c and c[0] == sig:
        return c[1]
    tab = {r[0]: (r[1], r[2], r[3], r[4]) for r in con.execute(
        "SELECT term, kind, direct, df, weight FROM entity_index WHERE df > 0 OR kind='config'")}
    _CACHE[key] = (sig, tab)
    return tab


def match(con, text: str, *, max_len: int = 12) -> list[dict]:
    """当前消息里出现了哪些专名。最长匹配优先（「小花」命中后不再单独报「花花」）；
    直通的排前面，再按稀有度（df 小的在前）。表不存在 → 空列表（调用方照旧走 FTS+向量）。"""
    tab = _load(con)
    if not tab or not text:
        return []
    low = text.lower()
    spans = []
    n = len(low)
    for i in range(n):
        for L in range(2, min(max_len, n - i) + 1):
            w = low[i:i + L]
            # 关键：原文里的词按小写比；表里的 term 也小写
            hit = tab.get(w)
            if hit is not None:
                spans.append((i, i + L, w, hit))
    if not spans:
        return []
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    taken: list[tuple[int, int]] = []
    out = []
    for a, b, w, (kind, direct, df, weight) in spans:
        if any(a >= x and b <= y for x, y in taken):
            continue
        taken.append((a, b))
        out.append({"term": text[a:b], "kind": kind, "direct": bool(direct) and not is_generic(text[a:b], kind),
                    "df": df, "weight": weight})
    out.sort(key=lambda e: (not e["direct"], e["df"], -len(e["term"])))
    return out


def lower_table_terms(con) -> None:           # 供测试：确保表里 term 是小写可比的（英文标识符大小写）
    pass


def facts_for(con, term: str, limit: int = 30) -> list[dict]:
    """挂着这个专名的活事实（mem0 的「实体 → 记忆」链接，这里直接按正文子串，不另存链接表）。
    排序：钉住 > 已确认 > 新近。字段跟 vectors.search 返回的一样。"""
    rows = con.execute(
        """SELECT id, ext_id, subject, content, status, user_pinned, valid_from, importance, confidence,
                  layer, access_count, last_accessed, fact_type
           FROM facts WHERE valid_to IS NULL AND status IN ('candidate','confirmed')
             AND instr(lower(content), lower(?)) > 0
           ORDER BY user_pinned DESC, (status='confirmed') DESC,
                    COALESCE(valid_from, created_at) DESC, id DESC LIMIT ?""", (term, limit)).fetchall()
    return [dict(r) | {"score": 0.0, "my_stance": None, "via": "entity"} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--no-jieba", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--match", help="试匹配一句话（只读）")
    a = ap.parse_args()
    from store.store import connect
    con = connect(a.db) if a.db else connect()
    if a.refresh:
        s = refresh(con, use_jieba=not a.no_jieba, dry_run=a.dry_run)
        s.pop("rows", None)
        print(s)
    if a.stats or not (a.refresh or a.match):
        tab = _load(con)
        kinds: dict[str, list[int]] = {}
        for k, d, df, w in tab.values():
            kinds.setdefault(k, [0, 0])
            kinds[k][0] += 1
            kinds[k][1] += d
        print({"terms": len(tab), "by_kind(total,direct)": kinds})
    if a.match:
        for e in match(con, a.match):
            print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
