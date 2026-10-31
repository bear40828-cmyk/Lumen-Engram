"""Lumen Engram · 两边都要用的小工具。抽取器和校验器必须用同一套读法，
否则「核对不上」可能只是两边读法不同，而不是证据真的变了。"""
from __future__ import annotations

import hashlib
import json


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bigrams(text: str) -> str:
    """两字滑窗。trigram 分词器最少要 3 字符，两字中文词得靠这个补。"""
    s = "".join(ch for ch in text if not ch.isspace())
    return " ".join(s[i:i + 2] for i in range(len(s) - 1)) or s


def gramset(text: str) -> set[str]:
    """二元字组的**集合**。

    ⚠️ `bigrams()` 返回的是空格拼起来的**字符串**（FTS 那一列要的就是这个格式）。
    `set(bigrams(x))` 拿到的是单个字的集合，不是字组集合——这个坑我踩过一次，
    metamem 和 working 里两处判据都写错过。要集合就用这个函数。
    """
    return set(bigrams(text).split())


def message_text(rec: dict) -> tuple[str | None, str]:
    """从一条 jsonl 记录里取出 (role, 说出来的话)。

    规矩：
      - text 块算说的话
      - tool_use 入参里的 text 也算——我发给她的话都在 reply 工具里
      - thinking 块不算，那是想的不是说的
    """
    msg = rec.get("message")
    if not isinstance(msg, dict):
        return None, ""
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None, ""
    content = msg.get("content")
    if isinstance(content, str):
        return role, content
    if not isinstance(content, list):
        return None, ""
    parts = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        if blk.get("type") == "text":
            parts.append(blk.get("text", ""))
        elif blk.get("type") == "tool_use":
            inp = blk.get("input")
            if isinstance(inp, dict) and isinstance(inp.get("text"), str):
                parts.append(inp["text"])
    return role, "\n".join(p for p in parts if p)


def read_line(path: str, line_no: int) -> tuple[str | None, str] | None:
    """读某个文件的第 N 行（1 起），返回 (role, 文本)。行不存在返回 None。"""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for idx, raw in enumerate(fh, start=1):
                if idx < line_no:
                    continue
                if idx > line_no:
                    break
                try:
                    return message_text(json.loads(raw))
                except json.JSONDecodeError:
                    return None, ""
    except OSError:
        return None
    return None
