#!/usr/bin/env python3
"""Lumen Engram · 动态显著性

一句话：**决定一条记忆有多容易被想起来，绝不决定它是否存在。**

    激活度 = 初始重要度 × 时间衰减
           + 重复强化（被召回过几次）
           + 情绪重新激活（很小的权重）
           + 钉住加权（她要求留的永远浮在上面）

时间衰减用半衰期，不做硬截断——半年前的事只是变淡，不会消失。
`user_pinned` 的东西不参与衰减，也不参与冷层迁移。

冷热迁移只在 active ↔ cold 之间搬，**永远不删**。冷层的东西照样搜得到，
只是排在后面——这就是「遗忘是不再容易被叫醒」的实现。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import half_life  # noqa: E402

HALF_LIFE_DAYS = 45.0        # 默认半衰期（episodic）；其他类型见 classify.HALF_LIFE
COLD_THRESHOLD = 0.55        # 激活度低于这个搬去冷层
COLD_WEIGHT = 0.5            # 冷层的东西检索时激活度打五折（只影响排序，不影响是否存在）
WARM_THRESHOLD = 0.75        # 高于这个从冷层搬回来（留个滞回区，别来回抖）


def decay(age_days: float | None, fact_type: str | None = None) -> float:
    """时间衰减系数，0~1。半衰期按记忆类型走：偏好淡得最慢，一次性事件最快。

    没有时间的按 0.5 算，不给它免费的新鲜度。
    """
    if age_days is None or age_days < 0:
        return 0.5
    return math.pow(0.5, age_days / half_life(fact_type))


def activation(*, importance: float | None, confidence: float | None,
               emotion: float | None, access_count: int, age_days: float | None,
               user_pinned: bool, fact_type: str | None = None,
               context_hit: float = 0.0) -> float:
    """context_hit 是当前话题相似度，0~1，由 router 传进来。"""
    if user_pinned:
        return 10.0                                    # 钉住的一律浮在最上面
    base = (importance if importance is not None else 0.5)
    fresh = decay(age_days, fact_type)
    repeat = min(access_count, 8) * 0.06               # 被想起过的更容易再想起
    conf = (confidence if confidence is not None else 0.5) * 0.4
    emo = (emotion if emotion is not None else 0.0) * 0.15
    return round(base * (0.4 + 0.6 * fresh) + repeat + conf + emo + context_hit * 0.8, 4)


def register(con) -> None:
    """把衰减和激活度注册成 SQL 函数，让排序能在库里算。"""
    con.create_function("engram_decay", 2, decay)

    def _act(importance, confidence, emotion, access_count, age_days, pinned, fact_type=None):
        return activation(importance=importance, confidence=confidence, emotion=emotion,
                          access_count=access_count or 0, age_days=age_days,
                          user_pinned=bool(pinned), fact_type=fact_type)
    con.create_function("engram_activation", 6, _act)
    con.create_function("engram_activation", 7, _act)


def migrate_layers(con, *, dry_run: bool = False) -> dict:
    """按激活度搬冷热。只搬 layer，不动状态、不删东西、跳过钉住的。"""
    rows = con.execute("""
        SELECT id, ext_id, layer, user_pinned,
               engram_activation(importance, confidence, emotion, access_count,
                                 julianday('now') - julianday(COALESCE(valid_from, created_at)), user_pinned,
                                 fact_type) AS act
        FROM facts
        WHERE status IN ('candidate','confirmed') AND user_pinned = 0
    """).fetchall()

    to_cold = [r for r in rows if r["layer"] == "active" and r["act"] < COLD_THRESHOLD]
    to_warm = [r for r in rows if r["layer"] == "cold" and r["act"] >= WARM_THRESHOLD]

    if not dry_run:
        for r in to_cold:
            con.execute("UPDATE facts SET layer='cold', updated_at=datetime('now'),"
                        " version=version+1 WHERE id=?", (r["id"],))
        for r in to_warm:
            con.execute("UPDATE facts SET layer='active', updated_at=datetime('now'),"
                        " version=version+1 WHERE id=?", (r["id"],))
        con.commit()

    return {"看了": len(rows), "搬去冷层": len(to_cold), "搬回活跃": len(to_warm)}
