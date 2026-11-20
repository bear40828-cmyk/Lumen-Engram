#!/usr/bin/env python3
"""Lumen Engram · 召回回放（只读副本上跑；不连模型除非 --llm prod）

把最近 N 句她的真实消息，各自原样喂给钩子（in-process），旧路径 vs 召回改造路径（ENGRAM_RECALL_V2=1）各一遍，
记下每句注入了哪几条。用来看「误注入有没有变多」。

    python3 replay_recall.py --db /path/COPY.db -n 100 --out replay.json          # 精排不可用的确定性对照（零成本）
    python3 replay_recall.py --db /path/COPY.db -n 100 --llm prod --out replay.json  # 真精排（花钱！要先报预算）

安全：只许在**副本**上跑（会写遥测表）；--db 指向 engram.db 同名文件会拒绝。精排默认关（--llm none → 精排返回 None，
走关键词闸退路）；--embed MODULE:FUNC 可以换嵌入函数（比如离线 bge-m3），缺省用 vectors.embed（要 Cloudflare 凭据）。
"""
from __future__ import annotations

import persona  # noqa: E402
import argparse
import contextlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ENGRAM_LLM_USAGE", "0")


def real_messages(con, n: int, ctx_turns: int = 6) -> list[dict]:
    """最近 n 句她的消息 + 各自之前的几轮对话（同一会话文件里的她和助手）。"""
    rows = con.execute(
        "SELECT id, source_path, content, occurred_at FROM evidence WHERE source_kind='chat' AND source_actor=?"
        " AND length(content) >= 2 ORDER BY id DESC LIMIT ?", (persona.USER, n * 3)).fetchall()
    out = []
    for r in rows:
        t = (r["content"] or "").strip()
        if re.match(r"^\s*(【|<task-notification|<system-reminder|\[tool|\(document)", t):
            continue
        prev = con.execute(
            "SELECT source_actor, content FROM evidence WHERE source_path=? AND source_kind='chat' AND id<? "
            "AND source_actor IN (?,'claude') ORDER BY id DESC LIMIT ?", (r["source_path"], r["id"], persona.USER, ctx_turns)).fetchall()
        out.append({"id": r["id"], "text": t, "at": r["occurred_at"],
                    "context": [(p["source_actor"], (p["content"] or "")[:260]) for p in reversed(prev)]})
        if len(out) >= n:
            break
    return list(reversed(out))


def write_transcript(ctx) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="replay-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for actor, body in ctx:
            fh.write(json.dumps({"type": "user" if actor == persona.USER else "assistant",
                                 "message": {"content": body}}, ensure_ascii=False) + "\n")
    return path


def run_hook(hook, text: str, ctx) -> str:
    tp = write_transcript(ctx)
    try:
        sys.stdin = io.StringIO(json.dumps({"prompt": text, "transcript_path": tp}, ensure_ascii=False))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            hook.main()
        return buf.getvalue()
    finally:
        sys.stdin = sys.__stdin__
        os.unlink(tp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="副本库路径（不许是 engram.db）")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--out", default="replay.json")
    ap.add_argument("--llm", choices=["none", "prod"], default="none")
    ap.add_argument("--embed", help="MODULE:FUNC，替换 vectors.embed（离线嵌入）")
    ap.add_argument("--new-mode", default="1", help="新路径的 ENGRAM_RECALL_V2 取值（1 或 hybrid）")
    ap.add_argument("--rewrite", action="store_true", help="新路径同时开查询改写（--llm prod 才有意义）")
    a = ap.parse_args()
    if Path(a.db).name == "engram.db":
        print("拒绝：只许在副本上回放"); return 2
    os.environ["ENGRAM_HOME"] = str(ROOT)
    import store.store as S
    S.connect.__defaults__ = (a.db,)
    import vectors
    if a.embed:
        mod, fn = a.embed.split(":")
        vectors.embed = getattr(importlib.import_module(mod), fn)
    import rerank
    if a.llm == "none":
        rerank.judge = lambda *args, **kw: None
    import newer
    newer.superseded_by = lambda *args, **kw: None
    import hooks_engram_recall as hook
    hook.LOG = os.devnull
    hook.recently_injected = lambda con, minutes=30: set()
    import sqlite3
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row
    msgs = real_messages(con, a.n)
    res = []
    for i, m in enumerate(msgs, 1):
        row = {"id": m["id"], "text": m["text"], "context": m["context"][-2:]}
        for tag, v2 in (("old", "0"), ("new", a.new_mode)):
            os.environ["ENGRAM_RECALL_V2"] = v2
            os.environ["ENGRAM_RECALL_REWRITE"] = "1" if (v2 != "0" and a.rewrite) else "0"
            out = run_hook(hook, m["text"], m["context"])
            row[tag] = out.strip()
        res.append(row)
        print(f"{i:3d}/{len(msgs)} old={'注' if row['old'] else '—'} new={'注' if row['new'] else '—'} | {m['text'][:30]}", file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    inj = lambda k: sum(1 for r in res if r[k] and r[k].startswith("【记忆库·"))
    print(json.dumps({"messages": len(res), "old_injected": inj("old"), "new_injected": inj("new")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
