"""密钥打码：存进库之前、发给模型之前、注入对话之前，都过这一道。**没有关闭开关**（用户和助手定的）。

命中的串换成 〔已打码:类型:前4…后2〕——只留头 4 尾 2 个字符，够人认出是哪一把，不够还原。
日志只记「哪里、哪类、几个」，**从不记原文**；日志写不了也不影响调用，也绝不 print（钩子的 stdout 会进对话上下文）。

识别的类型：
  sk       sk-… 形态的 API key（DeepSeek / OpenAI / Anthropic 一类）
  github   github_pat_… / ghp_ gho_ ghu_ ghs_ ghr_
  slack    xoxb- xoxp- xoxa- … 令牌
  aws      AKIA… / ASIA… 访问密钥 id
  tg       Telegram bot token（8~10 位数字:35 位）
  bark     URL 里 Bark 的 key：主机名含 bark / day.app，或路径里有 /bark…/，后面那一段 ≥16 位
  bearer   Authorization: Bearer <token>（只打码 token 本身）
  kv       「密钥/令牌/绑定码/token/secret/password/api_key = 一串」后面那一串（要混合字母和数字）
  entropy  兜底：≥32 位、字母数字混合、熵够高的长串

刻意不打码（实测会误伤）：纯十六进制串（sha256/git 哈希/md5）、UUID、全字母或全数字的串、
像标识符的长串（snake_case / kebab-case，且大半是单词和纯数字段）、路径和 URL 的其余部分。
纯十六进制的 key 只有紧跟「密钥/token…」这类词时才会被 kv 一类抓到。
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "redact.log"

_TOK = r"[A-Za-z0-9_\-]"

_RULES: list[tuple[str, re.Pattern, int]] = [
    # (类型, 正则, 要打码的分组号；0＝整段)
    ("github", re.compile(r"(?<![A-Za-z0-9_])(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"), 0),
    ("sk", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}"), 0),
    ("slack", re.compile(r"(?<![A-Za-z0-9])xox[abprsoe]?-[A-Za-z0-9\-]{10,}"), 0),
    ("aws", re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"), 0),
    ("tg", re.compile(r"(?<![0-9A-Za-z])\d{8,10}:[A-Za-z0-9_\-]{35}(?![A-Za-z0-9_\-])"), 0),
    ("bark", re.compile(
        r"(https?://(?:[^\s/]*bark[^\s/]*|[^\s/]*day\.app)/|https?://[^\s]*?/bark[^\s/]*/)"
        r"([A-Za-z0-9_\-]{16,})", re.I), 2),
    ("bearer", re.compile(r"(\bBearer\s+)([A-Za-z0-9._~+/=\-]{16,})", re.I), 2),
]
# 关键词 + 赋值：「密钥是 xxx」「token: xxx」「api_key=xxx」
_KV = re.compile(
    r"((?:api[_\-]?key|secret|token|passw(?:or)?d|access[_\-]?key|密钥|令牌|绑定码|口令|密码)"
    r"[A-Za-z0-9_]{0,12}\s*(?:[:=：]|是|为|\s)\s*[`\"'“‘]?)"
    r"([A-Za-z0-9._~+/=\-]{16,})", re.I)
_LONG = re.compile(r"(?<![A-Za-z0-9_\-])[A-Za-z0-9_\-]{32,}={0,2}(?![A-Za-z0-9_\-])")
_HEX = re.compile(r"^[0-9a-fA-F]+$")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_MASKED = "〔已打码:"


def _entropy(s: str) -> float:
    n = len(s)
    if not n:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum(c / n * math.log2(c / n) for c in freq.values())


def _identifier_like(s: str) -> bool:
    """snake/kebab 标识符、带日期的文件名：拆开后大半是单词或纯数字段。"""
    parts = [p for p in re.split(r"[_\-]", s) if p]
    if len(parts) < 3:
        return False
    good = sum(len(p) for p in parts if p.isdigit() or (p.isalpha() and (len(p) >= 2)))
    return good / max(1, sum(len(p) for p in parts)) >= 0.7


def _secretish(s: str, *, min_len: int, min_entropy: float) -> bool:
    s = s.rstrip("=")
    if len(s) < min_len or _UUID.match(s) or _HEX.match(s):
        return False
    if s.isalpha() or s.isdigit() or not any(c.isdigit() for c in s) or not any(c.isalpha() for c in s):
        return False
    if _identifier_like(s):
        return False
    return _entropy(s) >= min_entropy


_PUBKEY_BEFORE = re.compile(r"(?:ssh-(?:ed25519|rsa|dss)|ecdsa-sha2-nistp\d+|sk-ssh-ed25519@openssh\.com)\s+$")
_FILEID_BEFORE = re.compile(r"file(?:_unique)?_id\s*[=:]\s*[\"'`]?$", re.I)
_EXT_AFTER = re.compile(r"\.[A-Za-z0-9]{2,5}(?![A-Za-z0-9])")


def _benign_context(text: str, m) -> bool:
    """长串但不是凭据：SSH 公钥、Telegram file_id、路径里带扩展名的文件名（…/inbox/<随机名>.jpg）。"""
    before = text[max(0, m.start() - 40):m.start()]
    if _PUBKEY_BEFORE.search(before) or _FILEID_BEFORE.search(before):
        return True
    if before.endswith("/") and _EXT_AFTER.match(text, m.end()):
        return True
    return False


def _mask(kind: str, raw: str) -> str:
    return f"〔已打码:{kind}:{raw[:4]}…{raw[-2:]}〕"


def redact(text):
    """返回 (打码后的文本, {类型: 个数})。非字符串原样返回。"""
    if not isinstance(text, str) or len(text) < 16:
        return text, {}
    counts: dict[str, int] = {}

    def bump(k: str) -> None:
        counts[k] = counts.get(k, 0) + 1

    for kind, rx, grp in _RULES:
        def rep(m, kind=kind, grp=grp):
            bump(kind)
            if grp == 0:
                return _mask(kind, m.group(0))
            return m.group(1) + _mask(kind, m.group(2))
        text = rx.sub(rep, text)

    def rep_kv(m):
        if _secretish(m.group(2), min_len=16, min_entropy=2.8) or (
                _HEX.match(m.group(2)) and len(m.group(2)) >= 20
                and any(c.isdigit() for c in m.group(2)) and any(c.isalpha() for c in m.group(2))):
            bump("kv")
            return m.group(1) + _mask("kv", m.group(2))
        return m.group(0)
    text = _KV.sub(rep_kv, text)

    def rep_long(m):
        s = m.group(0)
        if _secretish(s, min_len=32, min_entropy=4.0) and not _benign_context(m.string, m):
            bump("entropy")
            return _mask("entropy", s)
        return s
    text = _LONG.sub(rep_long, text)
    return text, counts


def note(where: str, counts: dict) -> None:
    """只记计数。日志写失败静默。"""
    if not counts:
        return
    try:
        line = f"{time.strftime('%F %T')} {where} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:                                        # noqa: BLE001
        pass


def scrub(text, where: str = ""):
    """打码 + 记一笔计数。给「出口」用：注入前、送模型前、入库前。"""
    out, counts = redact(text)
    if counts:
        note(where, counts)
    return out


def contains_secret(text) -> bool:
    return bool(redact(text)[1])
