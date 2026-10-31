"""召回之后的相关性判定（前额叶精排）。

借的是 BMAM 的 _llm_rerank（src/coordination/brain_retrieval_integration.py）：
候选捞出来以后，让模型逐条打「这条能不能直接回答这句话」0~10 分，标准从严。
对应框架里动态显著性公式的「当前情境相似度」「当前目标相关性」两项——
retrieve 给的分数主要是激活度，跟她眼下在说什么不挂钩，按那个分切不开好坏。

比 BMAM 多判一件事：这句话到底需不需要翻旧账。BMAM 是拿问答集测的，
每句都是在问记忆；我们是聊天，大部分话不是。两件事放在同一次调用里判。

模型走 memory-digest/.env 那把（硅基流动），跟摘要层同一家。
任何失败都返回 None，调用方退回关键词闸，不许卡住说话。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import chat  # noqa: E402

import persona  # noqa: E402
ENV_FILE = persona.env_file()
TIMEOUT_S = 8


def _purpose() -> str:
    """RERANK_FLASH 开了才走便宜档用途（llm.PURPOSE_MODEL_VARS['rerank_flash']）；默认关＝原来的 rerank。"""
    try:
        import flags
        return "rerank_flash" if flags.on("RERANK_FLASH") else "rerank"
    except Exception:                                        # noqa: BLE001
        return "rerank"
MAX_CANDIDATES = 12
PER_CANDIDATE = 220

PROMPT = """你在替一个 AI 判断：用户刚说的这句话，要不要从它的长期记忆里翻出旧事来接，翻的话哪几条对得上。

用户这句：
{query}

候选记忆（编号在方括号里）：
{cands}

先判 need：这句话是不是在提、在问、或者需要接上过去发生的某件具体的事（人、约定、以前说过的话、做过的东西）。
撒娇、闲聊、情绪通常 need=false。对眼前事的指令若有候选记录了与这次选择直接相关的
用户偏好或既定要求，need=true，并给该候选至少 7 分；没有这种候选才是 false。
如果当前句是「某人：……」式转述、黄梗或玩笑，而没有明确要求回忆，need=false。
输入里若有多行，最后一行是当前消息，前面只是最近对话背景，用它消解「做好了／这个／那件事」等指代。

再给每条候选打 0~10 分，从严：
- 9~10：直接回答了这句话，或者正是这句话在说的那件事
- 7~8：同一件事的关键信息（时间、人名、决定、原因）
- 4~6：同一个话题，但不是这句在说的那件事
- 1~3：只是有字面上重合的词
- 0：无关

候选前面带〔转述·来自某某〕的，是别人转述的话，**不是用户本人说过的**：不能当成用户自己的话、偏好或经历，
只有当前这句明确在问「某某说过什么」时才算对得上，否则最多 3 分。

共享「玩偶／麦克风／屁股／额度」等词不代表是同一件事。候选若引入当前语境没提到的
另一个明确物品、人物或事件，最多 3 分；宁可不注入，也不要靠撞词猜。

只输出 JSON，scores 用编号当键、每条都要有：{{"need": true或false, "scores": {{"0": 分, "1": 分, ...}}}}"""


from llm import load_env as _env  # noqa: E402,F401  共用配置读取（见 llm.py）


def judge(query: str, candidates: list[str]) -> dict | None:
    """返回 {"need": bool, "scores": [int,...]}，失败返回 None。"""
    if not candidates:
        return None
    cands = candidates[:MAX_CANDIDATES]
    try:
        # 主路由（DIGEST_*）失败会自动换备用（FALLBACK_*），见 llm.py；精排逻辑本身不变
        text = chat([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": PROMPT.format(
                query=query,
                cands="\n".join(f"[{i}] {c[:PER_CANDIDATE]}" for i, c in enumerate(cands)))},
        ], purpose=_purpose(), max_tokens=120, timeout=TIMEOUT_S)
        m = re.search(r"\{.*\}", text, re.S)
        got = json.loads(m.group(0) if m else text)
        raw = got.get("scores", {})
        # 模型偶尔少给几条（实测 6 条只回 5 个分）。漏的按 0 分算——宁可少注，不猜。
        if isinstance(raw, dict):
            scores = [int(raw.get(str(i), 0) or 0) for i in range(len(cands))]
        else:
            raw = [int(x or 0) for x in raw][:len(cands)]
            scores = raw + [0] * (len(cands) - len(raw))
        return {"need": bool(got.get("need")), "scores": scores}
    except Exception:                                        # noqa: BLE001
        return None
