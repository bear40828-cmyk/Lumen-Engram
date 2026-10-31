#!/usr/bin/env python3
"""Lumen Engram · 功能开关（任务二，2026-10-28）

所有「新路径 / 冻结的旧路径」的开关都在这里登记，**默认值写在这一处**，改默认只改这里。
读法：先看环境变量 ENGRAM_<名字>，再看 .env 文件（llm.load_env，同一个 .env）里的同名项；
都没有就用默认。值为 1/true/on/yes 算开，其余算关。每次调用现读，不缓存（测试好切，
钩子是一次性进程所以没有性能问题）。

    名字              默认   管什么
    METAMEM           关     召回时是否用 metamem 的 FOK/TOT「够不够」判据 + 证据体检（冲突提示）。
                             关＝路由 + 精排两层决定「要不要翻、翻哪几条」，不再有第三层来判够不够。
                             （2026-10-29）问句里的「我/她」＝用户、「你」＝助理，只认那个人的事；混合召回里把她的事排前面。
    LAYERED_RETRIEVE  关     是否走旧的「六类分层路由 + 逐层升级」召回。
                             关＝混合召回（FTS 关键词 + 向量，合并后交精排）；开＝回到旧路径（回滚用）。
    CONSOLIDATE       关     consolidate.py（巩固：把多次出现的事实升级）入口。关＝直接退出，什么都不写。
    INVALIDATE_CHAIN  关     （2026-10-29 降级）新事让同槽旧事「可能过时」的软标记（stale.py）：只在旁表 fact_stale_marks
                             记一行，不封旧事、不改 status、不删；代码硬卡「新的必须是她本人原话」。召回时旧的排序×0.5
                             并标「可能已过时」。要真正生效还需 ENGRAM_TIMELINE=1（负责判）。
                             store.supersede / invalidate（硬封）只剩手工 hard=True 才能用，管线不再调用。
                             关＝不写标记、不读标记。confirm/pin 不受影响。
    TIMELINE          关     extract/timeline.py 的新旧判断（抽取时让模型判「新事是否顶替旧事」）。
                             关＝不判（等同模型没配），抽取照常入库。
    TIMELINE_CONFIRM  关     （2026-10-29）TIMELINE 判中一对后，再让 TIMELINE_CONFIRM_MODEL（没配退回 EXTRACT_MODEL，
                             通常配成更强的模型）用同一套标准只重判这一对，两遍都说「顶替」才算。只多花命中那几对的钱。
                             大样本实测：只靠便宜模型 48% 精度，加强模型复核后 83%（见 TALK）。
    HANDOFF_INJECT    关     （2026-10-29）hooks_handoff_inject.py（SessionStart 钩子）：开新窗口时把交接只读镜像里最新一份笔记的
                             「摘要 / 悬着的事 / 她的状态」三段（带标签、限长、失败静默）注入上下文。钩子由助手在 settings 里注册。
    UNIFIED_GATE      关     抽取时用「一次 LLM 调用」同时出 promote/fact_type/subject/fact_key/slots
                             （取代 gate+classify+keys 三套规则）。关＝仍走原规则。
    RAW_ONLY          关     （任务三）只存原话：unified.decide 一律不晋升、from_channel 全员只存证据、
                             ingest.py（每日摘要灌库）冻结。关＝跟现在一字不差。要和 DISTILL_SLICE 一起开才有意义。
    DISTILL_SLICE     关     （任务三）hourly/auto-ingest.sh 走 collect → distill_slice（会话安静 ≥3 小时才整理，
                             带游标/退避/死信箱）。关＝仍跑旧的 75 分钟窗口 from_evidence。
    SELF_REMEMBER     关     （任务三）MCP 的 engram_remember 工具。关＝tools/list 里没有它，调用得到「没有这个工具」。
    RERANK_FLASH      关     （任务三）精排（rerank.py）改用 RERANK_FLASH_MODEL（便宜档）；没配该变量就仍用原模型。
                             新旧比对（newer.py）不受影响。
    VALUE_GATE        关     （任务三后续；已部署到正式库，默认关）「值不值得记」闸：确定性预过滤（心跳/应答/工具输出/JSON/Error/太短）+
                             封闭类别抽取提示词 + 角色规则 + 后检 + 置信度分档，见 value_gate.py。关＝跟现在一字不差。
    VALUE_GATE_UNION  关     （已部署到正式库，默认关；要 VALUE_GATE 开才有意义）召回补偿：闸开时同一片段再用旧抽取器提示词抽一遍，
                             候选（_old）过同一套后检、与新提示词输出去重合并；旧候选只收「大事」（她的决定/规矩/搬家考研这类、
                             助手的承诺），每片段仍 3 条封顶，大事可超到 5 条。多一次模型调用。
    CLUSTER_SUMMARY   关     （任务四·步骤3，2026-10-29）夜间向量聚类归纳 maintenance/cluster_summarize.py：facts 向量按余弦≥0.75、
                             average linkage 聚簇，只拿她本人原话让模型写一句〔归纳〕，代码后检（簇内矛盾不写、「多次/经常」要多个事件撑）。
                             归纳是新的派生事实，原事实一行不改。关＝脚本直接退出。召回里〔归纳〕排在原话后面。
                             numpy 只装在 venv（.venv），脚本用 .venv/bin/python 跑；crontab 由助手加。
    CONTEXT_ROUTE     关     （任务四·步骤5，2026-10-29）召回按上下文分发到「人」：钩子传进来的检索文本（含最近对话）里恰好指向
                             一个人（我/她→用户，你→助理，助手乙→GPT 侧…）时，那个人的事实（和 us）在混合召回里小幅加分
                             （+0.01，只动顺序、不过滤、不丢）。who 显式传了就以 who 为准。关＝跟现在一字不差。
    ATOMIC_SPLIT      关     （任务四·步骤5）maintenance/split_atomic.py：把旧的「一条里塞好几件事」的事实拆成原子事实（新派生事实，
                             原事实不改）。关＝脚本直接退出。**不要对正式库跑**，先在副本上看。
    （另：llm_usage 记账不是开关，默认开，只记 token、失败静默；设环境变量 ENGRAM_LLM_USAGE=0 关掉。）
"""
from __future__ import annotations

import os

DEFAULTS = {
    "METAMEM": False,
    "LAYERED_RETRIEVE": False,
    "CONSOLIDATE": False,
    "INVALIDATE_CHAIN": False,
    "TIMELINE": False,
    "TIMELINE_CONFIRM": False,
    "HANDOFF_INJECT": False,
    "UNIFIED_GATE": False,
    "RAW_ONLY": False,
    "DISTILL_SLICE": False,
    "SELF_REMEMBER": False,
    "RERANK_FLASH": False,
    "VALUE_GATE": False,
    "VALUE_GATE_UNION": False,
    "CLUSTER_SUMMARY": False,
    "CONTEXT_ROUTE": False,
    "ATOMIC_SPLIT": False,
}
_TRUE = {"1", "true", "on", "yes"}


def on(name: str) -> bool:
    if name not in DEFAULTS:
        raise KeyError(f"未登记的开关：{name}")
    v = os.environ.get(f"ENGRAM_{name}")
    if v is None:
        try:
            from llm import load_env
            v = load_env().get(f"ENGRAM_{name}")
        except Exception:                                    # noqa: BLE001
            v = None
    if v is None:
        return DEFAULTS[name]
    return v.strip().lower() in _TRUE


class FrozenError(RuntimeError):
    """入口被开关冻结了。"""


def require(name: str, what: str) -> None:
    """冻结入口用：开关没开就抛 FrozenError（不改任何东西）。"""
    if not on(name):
        raise FrozenError(f"{what} 已冻结（开关 ENGRAM_{name}=1 才启用）；见 flags.py")
