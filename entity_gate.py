"""实体归属闸（2026-11-03）。

防回声只核说话人，不核内容里的实体注解（例：原话只说了某个外号，事实却写成「外号（GPT）」）。
这里补一条和「数字必须出自原话」同级的硬规矩：事实里出现的 AI 名 / 外号 / 括号注解，
必须在原话（引文、那条证据、片段上下文）里出现，或能由词表从原话里的称呼直接映射到同一个对象。

词表就是抽取提示词用的那份（ENGRAM_PROMPT_GLOSSARY_FILE，不进仓库）。格式：
    别名1 / 别名2 / ... = 说明            ← 一行一组，左边是同一个对象的称呼
    #! 上位 子组首名 ⊂ 父组首名            ← 原话只说了父组的称呼时，子组算「可能是」
    #! 外部 词1 / 词2 / ...                ← 外部语境（别人家、QQ 群……）：此时父组称呼不能推成子组
「#!」开头的行只给闸用，不进提示词（persona.glossary_block 会跳过）。
单个汉字的别名（「鱼」）不参与匹配：太容易撞上普通词。

判定（借 graphiti dedupe 的原则：拿不准就不并，宁可送审也不硬归给某个人）：
  ok       全部对象都有依据
  strip    没依据的称呼只出现在括号注解里 → 去掉注解后入库
  review   只能经「父组 → 子组」推出来、而上下文是外部语境（或没有任何直接依据但父组在场）→ 入库但 status=uncertain 待审
  reject   内容/主语里的对象在原话里完全找不到依据，或原话指向的是另一个对象 → 不入库
没配词表 → 闸整个不起作用（返回 ok）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_CACHE: dict = {}


@dataclass
class Glossary:
    groups: dict[str, list[str]] = field(default_factory=dict)     # 首名 → 别名（含首名）
    parent: dict[str, str] = field(default_factory=dict)           # 子组 → 父组
    foreign: list[str] = field(default_factory=list)               # 外部语境词
    subj: dict[str, str] = field(default_factory=dict)             # 库里的 subject 代号 → 组首名（#! 主语 gpt = 某别名）


def _split(s: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s*/\s*", s) if a.strip()]


def parse(text: str) -> Glossary:
    g = Glossary()
    for raw in text.splitlines():
        line = raw.strip().lstrip("-").strip()
        if not line:
            continue
        if line.startswith("#!"):
            body = line[2:].strip()
            m = re.match(r"上位\s+(.+?)\s*⊂\s*(.+)$", body)
            if m:
                g.parent[m.group(1).strip()] = m.group(2).strip()
            m = re.match(r"主语\s+(\S+)\s*=\s*(.+)$", body)
            if m:
                g.subj[m.group(1).strip().lower()] = m.group(2).strip()
            m = re.match(r"外部\s+(.+)$", body)
            if m:
                g.foreign += _split(m.group(1))
            continue
        if line.startswith("#"):
            continue
        for seg in re.split(r"[；;]", line):               # 一行可写多组：「A / B = X；C = Y」
            if "=" not in seg:
                continue
            left = seg.split("=", 1)[0]
            left = re.sub(r"^[「“\"]|[」”\"]$", "", left.strip())
            names = [re.sub(r"[「」“”\"]", "", a) for a in _split(left)]
            names = [a for a in names if a and len(a) <= 12]
            if names:
                g.groups.setdefault(names[0], [])
                for a in names:
                    if a not in g.groups[names[0]]:
                        g.groups[names[0]].append(a)
    return g


def load(path: str | None = None) -> Glossary | None:
    p = path or os.environ.get("ENGRAM_PROMPT_GLOSSARY_FILE")
    if not p:
        try:
            import persona
            p = persona.get("ENGRAM_PROMPT_GLOSSARY_FILE")
        except Exception:
            p = None
    if not p:
        return None
    try:
        st = Path(p).stat()
        key = (p, st.st_mtime)
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = parse(Path(p).read_text(encoding="utf-8"))
        return _CACHE[key]
    except OSError:
        return None


def _alias_rx(a: str) -> re.Pattern | None:
    if re.fullmatch(r"[\u4e00-\u9fff]", a):          # 单字别名不匹配
        return None
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", a):          # 英文别名：整词、忽略大小写
        return re.compile(r"(?<![A-Za-z0-9])" + re.escape(a) + r"(?![A-Za-z0-9])", re.I)
    return re.compile(re.escape(a), re.I)


def _single_char_support(g: Glossary, text: str) -> set[str]:
    return {k for k, al in g.groups.items() for a in al
            if re.fullmatch(r"[\u4e00-\u9fff]", a) and a in (text or "")}


def mentions(g: Glossary, text: str) -> dict[str, list[tuple[int, int]]]:
    """文本里出现了哪些组 → 命中位置。长别名优先：「小X」里的「X」不重复算；被更长的别名盖住的短别名不算。"""
    hits: list[tuple[int, int, str]] = []
    pairs = sorted(((a, k) for k, al in g.groups.items() for a in al), key=lambda x: -len(x[0]))
    taken: list[tuple[int, int]] = []
    for a, k in pairs:
        rx = _alias_rx(a)
        if rx is None:
            continue
        for m in rx.finditer(text or ""):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            hits.append((s, e, k))
    out: dict[str, list[tuple[int, int]]] = {}
    for s, e, k in hits:
        out.setdefault(k, []).append((s, e))
    return out


_SPEAKER = re.compile(r"^[^：:（(\n]{1,10}[（(][^）)]{0,30}[）)][：:]")
_PAREN = re.compile(r"[（(]([^（）()]{1,20})[）)]")


def _in_paren_only(content: str, spans: list[tuple[int, int]]) -> bool:
    ps = [m.span() for m in _PAREN.finditer(content)]
    return bool(spans) and all(any(ps_ <= s and e <= pe for ps_, pe in ps) for s, e in spans)


def _strip_parens(content: str, g: Glossary, bad: set[str]) -> str:
    def rep(m):
        inner = m.group(1)
        return "" if set(mentions(g, inner)) & bad else m.group(0)
    return _PAREN.sub(rep, content)


def _group_of_subject(g: Glossary, subject: str) -> str | None:
    s = (subject or "").lower()
    if s in g.subj:
        return g.subj[s]
    for k, al in g.groups.items():
        if any(a.lower() == s for a in al):
            return k
    return None


@dataclass
class Verdict:
    action: str                    # ok / strip / review / reject
    reason: str = ""
    content: str = ""


def judge(content: str, sources: list[str], context: str = "", subject: str = "",
          g: Glossary | None = None, actors: tuple | list | set = ()) -> Verdict:
    """content：事实正文；sources：引文 + 证据原话（直接依据）；context：片段上下文（只用于外部语境判断和兜底依据）。"""
    g = g if g is not None else load()
    if g is None or not g.groups:
        return Verdict("ok", content=content)
    direct = "\n".join(s for s in sources if s)
    src = set(mentions(g, direct))
    for a in actors or ():                              # 说话人本人就是依据（GPT 自己说的话，事实写 GPT）
        ag = _group_of_subject(g, str(a or ""))
        if ag:
            src.add(ag)
    ctx = set(mentions(g, context)) if context else set()
    foreign_ctx = bool(g.foreign) and any(w and w in (direct + "\n" + (context or "")) for w in g.foreign)
    cm = mentions(g, _SPEAKER.sub(lambda m: " " * len(m.group(0)), content, count=1))   # 「某助手（08-03 20:26）：」是说话人标签（来自证据元数据），不算内容里的实体
    sg = _group_of_subject(g, subject)
    need = dict(cm)
    if sg and sg not in need:
        need[sg] = []                                  # 主语本身就是某个 AI：同样要有依据
    bad_strip, review, reject = set(), [], []
    src_loose = src | _single_char_support(g, direct)    # 单字别名（「一条鱼」）只当依据用，不当内容里的点名
    for k, spans in need.items():
        if k in src_loose:
            continue
        par = g.parent.get(k)
        if par and par in src_loose:                   # 原话只说了父组（某外号），事实写成子组（管家）
            if foreign_ctx:
                review.append(f"{k}←{par}（外部语境）")
            continue
        if k in ctx:                                   # 依据在片段上下文：原话没点名（「他」）就放行；原话点了别的对象 → 送审
            if (src - {k}) or (par and foreign_ctx):
                review.append(f"{k}（只在上下文，原话说的是 {'/'.join(sorted(src - {k})) or '代词'}）")
            continue
        if par and par in ctx:
            review.append(f"{k}←{par}（只在上下文）")
            continue
        conflict = sorted(src - {k})
        why = f"{k} 原话和上下文里都没有" + (f"，原话说的是 {'/'.join(conflict)}" if conflict else "")
        if spans and _in_paren_only(content, spans) and sg != k:
            bad_strip.add(k)                           # 只是括号注解：去掉注解（「外号（GPT）」）
            continue
        if spans and conflict:
            reject.append(why)                         # 原话点名的是另一个对象，事实却写成这个：认错人
            continue
        review.append(why)                             # 原话里没点名任何对象（代词/省略）：拿不准，送审，不硬归
    if reject:
        return Verdict("reject", "entity:unsupported：" + "；".join(reject), content)
    new = _strip_parens(content, g, bad_strip) if bad_strip else content
    if review:
        return Verdict("review", "entity:review：" + "；".join(review), new)
    if bad_strip:
        return Verdict("strip", "entity:annotation-stripped：" + "/".join(sorted(bad_strip)), new)
    return Verdict("ok", content=content)


def send_to_review(con, fid: int | None, ok: dict) -> bool:
    """check() 输出里带 review 的：新事已入库 → 置 uncertain（可恢复，不删）。返回是否送审。"""
    if not fid or not ok.get("review"):
        return False
    from store.store import _bump
    ver = con.execute("SELECT version FROM facts WHERE id=?", (fid,)).fetchone()[0]
    _bump(con, fid, ver, status="uncertain")
    return True
