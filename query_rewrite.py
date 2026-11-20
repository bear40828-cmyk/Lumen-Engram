#!/usr/bin/env python3
"""Lumen Engram · 查询改写（召回改造·第 3 层，2026-11-04）

当前句没有明确专名（「那个旅行计划的事」「你还记得之前说的那个吗」）时，先让便宜模型把它改写成
2~3 个**具体的检索问句**，FTS 和向量各拿这几句再查一遍，候选并进同一个池子，交给精排。

借鉴：
  · agentmemory（Apache-2.0）src/functions/query-expansion.ts：系统提示词「query expansion engine for a memory
    retrieval system」+ 多条不同角度的改写（paraphrase / 具体化 / 抽象化）、顺便抽出实体、时间词具体化；
    模型失败时安静退回「不改写」。这里照它的结构改成中文、改成 JSON 输出、加上最近几轮对话做指代消解。
  · memobase（Apache-2.0）controllers/context.py `pack_latest_chat`：检索用的 query 带上最近 3 条对话——
    这里的 context 参数就是这个思路（最近 3~5 轮）。
  · 例子（Claude 的规格）：「那个旅行计划的事」→「旅行计划去哪里」「什么时候出发」。

只在**需要**时调（见 should_rewrite）：命中专名（直通）的不改写；太短/纯表情的不改写（FTS+向量照跑，只是不花这次钱）。
失败/超时/没配模型 → 返回 []，调用方照旧只用原句查。**绝不 print**（钩子 stdout 会注入对话）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TIMEOUT_S = 6
MAX_QUESTIONS = 3
MIN_CHARS = 4
EMOJI_ONLY = re.compile(r"^[\s\W_\U0001F000-\U0001FAFF\u2600-\u27BF]*$")

PROMPT = """你是长期记忆检索系统的「查询改写器」。用户刚说了一句话，它可能很含糊（指代、省略、只有感受没有对象）。
请结合最近的对话，把它改写成 2~3 个**具体的、能直接拿去搜记忆库的问句**，每句换一个角度：
一句还原它真正在指的人/事/物，一句换成库里可能的写法（同义词、上位词、「抢救」→「住院」这类），
必要时一句补上时间或人物。

规矩：
- 每句 ≤ 25 字，像记忆库里会写的陈述或问题，不要空话；
- 只用对话里出现过的人、事、物，不要编造；
- 如果这句话根本不是在提旧事（撒娇、寒暄、对眼前事的即时反应），输出空列表；
- 只输出 JSON：{{"questions": ["…", "…"], "entities": ["从句子里抽出的人名/物名，没有就空"]}}

最近的对话（旧→新，可能为空）：
{context}

用户当前这句：
{query}"""

_CACHE: dict = {}


def should_rewrite(text: str, *, has_direct_entity: bool = False) -> bool:
    """要不要花这一次调用。这不是否决权——不改写只是少一路检索词，FTS+向量照跑。"""
    if has_direct_entity:
        return False
    t = (text or "").strip()
    if len(re.sub(r"\s+", "", t)) < MIN_CHARS or EMOJI_ONLY.match(t):
        return False
    return True


def parse(raw: str) -> dict:
    """容忍 ```json 围栏和前后废话；失败返回空结果。"""
    try:
        m = re.search(r"\{.*\}", raw or "", re.S)
        d = json.loads(m.group(0) if m else raw)
        qs = [str(q).strip() for q in (d.get("questions") or []) if str(q).strip()]
        ents = [str(e).strip() for e in (d.get("entities") or []) if str(e).strip()]
        return {"questions": qs[:MAX_QUESTIONS], "entities": ents[:4]}
    except Exception:                                        # noqa: BLE001
        return {"questions": [], "entities": []}


def rewrite(text: str, context: list[str] | None = None, *, chat=None) -> dict:
    """返回 {"questions": [...], "entities": [...]}；任何失败 → 都是空。chat 可注入（测试/回放）。"""
    ctx = [c for c in (context or []) if c][-5:]
    key = (text, tuple(ctx))
    if key in _CACHE:
        return _CACHE[key]
    out = {"questions": [], "entities": []}
    try:
        if chat is None:
            from llm import chat as _chat
            chat = _chat
        raw = chat([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": PROMPT.format(
                context="\n".join(f"- {c[:120]}" for c in ctx) or "（无）", query=text[:300])},
        ], purpose="rewrite", max_tokens=160, timeout=TIMEOUT_S)
        out = parse(raw)
    except Exception:                                        # noqa: BLE001
        pass
    _CACHE[key] = out
    return out
