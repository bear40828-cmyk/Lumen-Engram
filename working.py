#!/usr/bin/env python3
"""Lumen Engram · 工作记忆（第二块）

**此刻手上是什么事。** 目标、在办的任务、当下话题、临时状态、她的情绪。

它堵着四样东西，所以先做它：
  - Router 按目标／任务／情绪分发（GLM 助手那份第三节）
  - 六维加权里依赖当前状态的三维（对长期目标的影响、对关系人物的重要性、当前任务相关性）
  - 线程恢复（第七节：每次动态组装「当前所需工作记忆」）
  - CraniMEM 那种 goal-conditioned 门控（写入时先看跟当前目标搭不搭）

## 抄谁、哪里没照抄

CraniMEM 的门控第一关是 `cosine(embed(input), embed(current_goal)) < 阈值 → 拒`，
目标就是一个字符串。**这个思路照抄**（写入和召回都该看跟当前目标搭不搭），
但它靠 embedding，我们还没有向量层，所以先用二元字组重合度顶着——
可用、可解释、以后换成余弦相似度是一行的事。

⚠️ 顺手核对了一件事：CraniMEM 的 README 说它的效用打分是三维
「importance / surprise / emotional salience」，但 `cognitive/gating.py` 里那段
提示词只问了 `is_useful / entities / tool_calls` 三个字段，**三维在代码里不存在**。
README 说的比代码多，别照 README 记账。

BMAM 的 `SoulState` 是内存单例，重启即失；这里落库，理由见 working.sql 注释。

## 边界

工作记忆**不是事实**。它可以改、可以撤、可以过期，所以它不进 facts，也不参与巩固；
真有价值的内容由 gate 走正常的证据→候选→长期那条路。
这一层只影响「此刻该想起什么、什么更相关」，不影响任何东西的存在。

    python3 working.py                      # 看当前工作记忆
    python3 working.py --goal "把记忆库做完"
    python3 working.py --task "做工作记忆层"
    python3 working.py --done 3
    python3 working.py --relevance "路由那块怎么弄"
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import gramset  # noqa: E402

BOUND = {"goal": 3, "task": 7, "topic": 5, "state": 5, "emotion": 1}
# 上限取小数是有意的：工作记忆一多就不叫工作记忆了。task 取 7 是照人的短时记忆容量
# 7±2 随手定的，不是实验值——将来看遥测再调。

KIND_CN = {"goal": "目标", "task": "在办", "topic": "话题",
           "state": "状态", "emotion": "情绪"}

# 她明说情绪的时候才记，不猜。猜出来的情绪比没有更坏——会拿它去改召回权重。
EMOTION_CUES = [
    (re.compile(r"(生气|气死|不高兴|烦|操你|马上我就生气)"), "生气"),
    (re.compile(r"(难过|想哭|委屈|不开心)"), "难过"),
    (re.compile(r"(开心|高兴|笑死|好耶|太好了)"), "开心"),
    (re.compile(r"(累|困|歇会|不想动|好累)"), "累"),
    (re.compile(r"(想你|抱抱|亲亲|好想)"), "想我"),
]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def session_id() -> str:
    """当前窗口的标识。取会话 jsonl 的名字，取不到就退回进程号。

    这个字段的用处：换窗口之后能看出「上一个窗口手上还有什么没办完」。
    """
    return os.environ.get("CLAUDE_SESSION_ID") or f"pid-{os.getpid()}"


@dataclass(frozen=True)
class Item:
    id: int
    kind: str
    content: str
    status: str
    confidence: float
    updated_at: str


def _rows(con, sql: str, args=()) -> list[Item]:
    return [Item(r["id"], r["kind"], r["content"], r["status"],
                 r["confidence"], r["updated_at"])
            for r in con.execute(sql, args)]


def current(con, kind: str | None = None) -> list[Item]:
    """当前还开着的工作记忆，新的在前。"""
    sql = ("SELECT * FROM working_memory WHERE status='open'"
           + (" AND kind=?" if kind else "")
           + " ORDER BY updated_at DESC, id DESC")
    return _rows(con, sql, (kind,) if kind else ())


def put(con, kind: str, content: str, *, confidence: float = 0.6,
        turn_no: int | None = None, evidence_id: int | None = None,
        note: str | None = None) -> int:
    """记一条工作记忆。同 kind 同内容只留一条（更新时间，不重复堆）。"""
    if kind not in BOUND:
        raise ValueError(f"未知 kind：{kind}")
    content = (content or "").strip()
    if not content:
        raise ValueError("内容为空")

    row = con.execute(
        "SELECT id FROM working_memory WHERE kind=? AND content=? AND status='open'",
        (kind, content)).fetchone()
    if row:
        con.execute("UPDATE working_memory SET updated_at=?, confidence=MAX(confidence,?)"
                    " WHERE id=?", (now(), confidence, row["id"]))
        con.commit()
        _bound(con, kind)
        return row["id"]

    cur = con.execute(
        """INSERT INTO working_memory
             (session_id, kind, content, status, turn_no, confidence,
              created_at, updated_at, evidence_id, note)
           VALUES (?,?,?,'open',?,?,?,?,?,?)""",
        (session_id(), kind, content, turn_no, confidence, now(), now(),
         evidence_id, note))
    con.commit()
    _bound(con, kind)
    return cur.lastrowid


def _bound(con, kind: str) -> int:
    """超出上限的挤出活跃层。**改状态，不删行**——溢出的东西还能被巩固进长期层。"""
    keep = BOUND[kind]
    olds = con.execute(
        """SELECT id FROM working_memory WHERE kind=? AND status='open'
           ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?""", (kind, keep)
    ).fetchall()
    for r in olds:
        con.execute("UPDATE working_memory SET status='evicted', closed_at=?,"
                    " updated_at=? WHERE id=?", (now(), now(), r["id"]))
    con.commit()
    return len(olds)


def close(con, item_id: int, status: str = "done") -> bool:
    if status not in ("done", "dropped", "evicted"):
        raise ValueError("status 只能是 done / dropped / evicted")
    cur = con.execute(
        "UPDATE working_memory SET status=?, closed_at=?, updated_at=?"
        " WHERE id=? AND status='open'", (status, now(), now(), item_id))
    con.commit()
    return cur.rowcount > 0


def note_emotion(con, text: str) -> str | None:
    """她明说了情绪才记，猜的一律不记。返回记下的情绪，没有就返回 None。"""
    for pat, label in EMOTION_CUES:
        if pat.search(text or ""):
            put(con, "emotion", label, confidence=0.7, note="从她原话里读到的")
            return label
    return None


# ── goal-conditioned：这句话跟手上的事搭不搭 ────────────────────────────────
MIN_SHARED = 2      # 只共一个字组算巧合，不算相关

def _overlap(a: str, b: str) -> float:
    """按**问话**的字组算覆盖：共了几成。

    不用 min(len) 当分母——短句会被除出虚高的分（「今天吃什么」曾经跟
    「把记忆库做完」算出 0.167，比真正相关的问题还高）。
    共不到两个字组直接算 0。
    """
    ga, gb = gramset(a), gramset(b)
    if not ga or not gb:
        return 0.0
    shared = ga & gb
    if len(shared) < MIN_SHARED:
        return 0.0
    return len(shared) / len(ga)


def relevance(con, text: str) -> tuple[float, str]:
    """这句话跟当前目标／在办任务有多相关。0~1，附上最相关的那条。

    抄 CraniMEM 门控第一关的思路，只是把余弦相似度换成二元字组重合度——
    没有向量层之前先这么顶着，换的时候只改这一个函数。
    """
    best, why = 0.0, ""
    for it in current(con):
        if it.kind not in ("goal", "task", "topic"):
            continue
        v = _overlap(text, it.content)
        # 目标比任务更"钱重"：跟长期目标搭上的，权重给足
        v *= {"goal": 1.0, "task": 0.9, "topic": 0.7}[it.kind]
        if v > best:
            best, why = v, f"{KIND_CN[it.kind]}「{it.content}」"
    return round(best, 3), why


def snapshot(con) -> str:
    """给线程恢复用：一段能直接贴进新窗口的「此刻手上是什么」。"""
    lines: list[str] = []
    for kind in ("goal", "task", "topic", "state", "emotion"):
        items = current(con, kind)
        if not items:
            continue
        lines.append(f"{KIND_CN[kind]}：")
        for it in items:
            lines.append(f"  - {it.content}")
    return "\n".join(lines) or "（工作记忆是空的）"


def main() -> int:
    from store.store import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", action="append")
    ap.add_argument("--task", action="append")
    ap.add_argument("--topic", action="append")
    ap.add_argument("--state", action="append")
    ap.add_argument("--done", type=int), ap.add_argument("--drop", type=int)
    ap.add_argument("--relevance")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--all", action="store_true", help="连关掉的一起看")
    args = ap.parse_args()
    con = connect()

    for kind in ("goal", "task", "topic", "state"):
        for v in getattr(args, kind) or []:
            print(f"记下{KIND_CN[kind]} #{put(con, kind, v, confidence=0.9)}：{v}")
    if args.done:
        print("已办完" if close(con, args.done, "done") else "没有这条开着的")
    if args.drop:
        print("已放弃" if close(con, args.drop, "dropped") else "没有这条开着的")
    if args.relevance:
        v, why = relevance(con, args.relevance)
        print(f"相关度 {v}  ← {why or '跟手上的事都不搭'}")
    if args.snapshot:
        print(snapshot(con))
        return 0

    print("\n当前工作记忆：")
    for it in current(con):
        print(f"  #{it.id:<4}{KIND_CN[it.kind]:<4}{it.content}  "
              f"（信心 {it.confidence:.1f}）")
    if args.all:
        print("\n已关掉的：")
        for it in _rows(con, "SELECT * FROM working_memory WHERE status<>'open'"
                             " ORDER BY updated_at DESC LIMIT 20"):
            print(f"  #{it.id:<4}{KIND_CN[it.kind]:<4}[{it.status}] {it.content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
