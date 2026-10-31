#!/usr/bin/env python3
"""Lumen Engram · 巩固（consolidation replay）

GPT 第二轮把这件事的边界划得很清楚，照办：

  **巩固是抽象，不是覆盖。**
  由若干 episodic 推出一条 semantic，那几条 episode 原样留着，
  用 fact_relations 的 derived_from 连过去，绝不 supersede 它们。
  supersede 只用于「观点或事实真的变了」（semantic v1 → v2）。

触发条件也按它给的三个维度，不用「出现 N 次就合并」：
  1. 独立支撑 ≥ 3 条
  2. 来自 ≥ 2 个不同日期（同一天连说三遍不算三份独立支撑）
  3. 方向一致（同一 subject + 同一 fact_key + 同一归一化值）

只对**有 fact_key 的**自动巩固。自由文本不上字符串相似度硬炼语义事实——
那是等 embedding 上了再说的事。

    python3 consolidate.py            # 干看有哪些够条件
    python3 consolidate.py --apply    # 真写进去
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from keys import by_name                                          # noqa: E402
from store.store import connect, promote, record_class, relate    # noqa: E402

MIN_SUPPORT = 3.0        # 支撑票数下限（按 origin_root 去重后计票）
MIN_DAYS = 2             # 至少来自两个不同日子
ECHO_WEIGHT = 1 / 3      # 别人转述当事人的事：算三分之一票
MIN_OWNER_VOTES = 1      # 当事人自己至少得说过一次，纯靠转述不许巩固

# GPT 第三轮第一颗螺丝：防「自己给自己造共识」。
# 用户说一次「考某大学」，我复述一次，GPT再复述一次——若三条都算独立支撑，
# 系统就会凭 AI 的回声把它判成稳定事实。所以只数：
#   1. tier = primary（本人原话，不是转述／摘要／系统生成）
#   2. evidence.source_actor == fact.subject（说这话的就是这条事实的主角）
#   3. origin_root 去重（同一句原话衍生出的一串，永远只算一票）

WHO = {persona.USER: persona.N_USER, "claude": persona.N_CLAUDE, "gpt": persona.N_GPT, persona.AGENT_C: persona.N_C}


def _actors(con, r) -> set:
    """这条证据实际算谁的话：说话人标注修正层（actor_fix）有改判就用改判，没有就是 source_actor。"""
    try:
        import actor_fix
        return actor_fix.actors_of(actor_fix.lookup(con, [r["evidence_id"]]).get(r["evidence_id"]), r["source_actor"])
    except Exception:                                        # noqa: BLE001
        return {r["source_actor"]}


def candidates(con) -> list[dict]:
    """按 (subject, fact_key, value) 聚，挑出够条件的。"""
    groups: dict[tuple, list] = defaultdict(list)
    for r in con.execute("""
            SELECT f.id, f.subject, f.content, f.valid_from, f.fact_type,
                   m.fact_key, m.value, m.polarity, t.origin_root,
                   e.source_actor, e.id AS evidence_id
            FROM fact_key_map m
            JOIN facts f ON f.id = m.fact_id
            JOIN fact_evidence fe ON fe.fact_id = f.id
            JOIN evidence e ON e.id = fe.evidence_id
            JOIN evidence_tier t ON t.evidence_id = e.id
            WHERE f.status IN ('candidate','confirmed') AND f.valid_to IS NULL
              AND m.value IS NOT NULL AND m.eligible = 1
              AND t.tier = 'primary'
              """):
        r = dict(r)
        spec = by_name(r["fact_key"])
        if spec is not None and not spec.consolidatable:
            continue
        # 「这条事实是关于谁的」跟「这句话是谁说的」分开：
        # registry 里标了 owner 的槽位，主语按 owner 算（考研是用户的事，
        # 哪怕开口的人是我）；没标 owner 的退回旧口径，用 fact.subject。
        owner = (spec.owner if spec is not None else None) or r["subject"]
        # 防回声：当事人自己说的算整票，别人转述只算三分之一票，
        # 而且没有当事人自己的那一票，光靠转述永远巩固不出来。
        r["weight"] = 1.0 if owner in _actors(con, r) else ECHO_WEIGHT
        r["subject"] = owner
        groups[(owner, r["fact_key"], r["value"], r["polarity"])].append(r)

    out = []
    for (subject, key, value, polarity), rows in groups.items():
        # 同源只算一票：一句原话被摘要抽一次、直抽再抽一次，仍然是 1 票。
        # 我复述、GPT复述都不在这儿——上面 SQL 已经按 source_actor 过滤掉了。
        by_root: dict[str, dict] = {}
        for r in rows:
            cur = by_root.get(r["origin_root"])
            # 同一句原话衍生出的一串只算一票，取其中权重最高的那条
            if cur is None or r["weight"] > cur["weight"]:
                by_root[r["origin_root"]] = r
        rows = list(by_root.values())
        votes = sum(r["weight"] for r in rows)
        owner_votes = sum(1 for r in rows if r["weight"] == 1.0)
        days = {(r["valid_from"] or "")[:10] for r in rows if r["valid_from"]}
        if votes < MIN_SUPPORT or len(days) < MIN_DAYS or owner_votes < MIN_OWNER_VOTES:
            continue
        # 已经巩固过就跳过（同一条抽象只做一次）
        already = con.execute("""
            SELECT 1 FROM fact_relations r JOIN fact_key_map m ON m.fact_id = r.parent_fact_id
            WHERE m.fact_key=? AND m.value=? LIMIT 1""", (key, value)).fetchone()
        if already:
            continue
        out.append({"subject": subject, "key": key, "value": value,
                    "polarity": polarity, "rows": rows, "days": sorted(days),
                    "votes": votes, "owner_votes": owner_votes})
    return out


def phrase(subject: str, key: str, value: str, n: int, days: list[str],
           polarity: str = "pos") -> str:
    who = WHO.get(subject, subject)
    span = f"{days[0]} 到 {days[-1]}" if len(days) > 1 else days[0]
    topic = {
        "preference.color": "在颜色上稳定偏向",
        "preference.food": "口味上反复提到",
        "education.target_school": "考研目标院校是",
        "education.target_major": "考研目标专业是",
        "education.exam_subject": "备考科目里反复出现",
        "education.graduation": "毕业时间是",
        "person.hometown": "老家是",
        "person.pet": "养的是",
        "comm.format_rule": "沟通格式上的固定要求是",
        "project.status": "项目状态反复被记为",
    }.get(key, f"在 {key} 上反复提到")
    neg = "反向（明确否定）" if polarity == "neg" else ""
    return (f"{who}：{topic}「{value}」{neg}——{span} 之间有 {n} 次独立支撑，由巩固得出")


def run(con, *, apply: bool) -> int:
    cands = candidates(con)
    if not cands:
        print(f"没有够条件的（要求：≥{MIN_SUPPORT} 条独立支撑、来自 ≥{MIN_DAYS} 个不同日子）")
        return 0

    for c in cands:
        spec = by_name(c["key"])
        content = phrase(c["subject"], c["key"], c["value"], len(c["rows"]),
                         c["days"], c["polarity"])
        content += f"（其中本人原话 {c['owner_votes']} 次）"
        print(f"\n{content}")
        print(f"  key={c['key']}  cardinality={spec.cardinality if spec else '?'}"
              f"  conflict={spec.conflict_policy if spec else '?'}")
        for r in c["rows"][:4]:
            print(f"    ← {(r['valid_from'] or '')[:10]}  {r['content'][:52]}")
        if len(c["rows"]) > 4:
            print(f"    …… 还有 {len(c['rows']) - 4} 条支撑")

        if not apply:
            continue

        fid = promote(con, subject=c["subject"], fact_type="semantic", content=content,
                      valid_from=c["days"][0], importance=0.8, confidence=0.85)
        if fid is None:
            print("    （库里已有同样的抽象，跳过）")
            continue
        record_class(con, fid, "semantic", 0.85,
                     f"由 {len(c['rows'])} 条支撑巩固而来，跨 {len(c['days'])} 天")
        con.execute("INSERT OR REPLACE INTO fact_key_map (fact_id, fact_key, value, built_at)"
                    " VALUES (?,?,?,datetime('now'))", (fid, c["key"], c["value"]))
        for r in c["rows"]:
            relate(con, fid, r["id"], "derived_from")
        con.commit()
        print(f"    ✓ 写入 {len(c['rows'])} 条 derived_from，原始那几条一个字没动")

    print(f"\n共 {len(cands)} 组够条件" + ("" if apply else "（干看，没写库）"))
    return 0


def main() -> int:
    # 2026-10-28（任务二·步骤4）冻结：巩固只对「有 fact_key 且 ≥3 条独立支撑」的事实出手，
    # 实际库里几乎没有够格的，跑了也是空转；而它一旦 --apply 就会往库里写 semantic 事实。
    # 默认不跑（不连库、不写任何东西）。ENGRAM_CONSOLIDATE=1 恢复。代码原样保留。
    import flags
    if not flags.on("CONSOLIDATE"):
        print("巩固已冻结（ENGRAM_CONSOLIDATE=1 才运行；见 flags.py），未做任何事。")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    return run(connect(), apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
