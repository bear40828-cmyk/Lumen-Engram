#!/usr/bin/env python3
"""Lumen Engram · 版本号集中在这儿

GPT 第三轮第三颗螺丝：派生层必须带版本。
不然半年后看到「preference, confidence=0.82」，根本不知道这是最早那个
会把「他妈」认成「他」的分类器打的分，还是修好之后的版本打的。

改了对应模块的判据，就把这里的号往上抬一位。
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CLASSIFIER_VERSION = "classify-0.2"     # 分句 + experiencer + 报道框架 + 置信度回退
ROUTER_VERSION = "router-0.3"           # skip / light / deep 三档
METAMEM_VERSION = "metamem-0.2"         # 证据化充分性：来源等级 + 有效期 + 冲突门禁
RETRIEVE_VERSION = "retrieve-0.2"       # L1 薄路由 + 分层召回 + 工作记忆分发 + 三档深度
SALIENCE_VERSION = "salience-0.2"       # 按类型分化的 recency 半衰期
REGISTRY_VERSION = "keys-0.2"           # project.status 收紧 + requires_self/polarity/digit_boundary
CARDS_VERSION = "cards-0.3"             # 最多两个命中窗口
GATE_VERSION = "gate-0.2"               # 事实槽位，长度只当弱特征


@lru_cache(maxsize=1)
def git_commit() -> str:
    """当前代码的提交号。日志要能对回是哪套代码跑出来的。"""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def config_version() -> str:
    return "+".join((GATE_VERSION, CLASSIFIER_VERSION, SALIENCE_VERSION,
                     REGISTRY_VERSION, CARDS_VERSION, ROUTER_VERSION,
                     METAMEM_VERSION, RETRIEVE_VERSION))


if __name__ == "__main__":
    print("git      ", git_commit())
    print("config   ", config_version())
    for name in ("CLASSIFIER", "ROUTER", "SALIENCE", "REGISTRY", "CARDS", "GATE",
                 "METAMEM", "RETRIEVE"):
        print(f"{name.lower():<10}", globals()[f"{name}_VERSION"])
