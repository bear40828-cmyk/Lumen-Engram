#!/usr/bin/env python3
"""LongMemEval 评测：Lumen Engram 召回 vs. BM25 对照。

每道题建一个临时库，把题目自带的历史会话逐轮写入证据层，再把用户每一轮直接升为候选事实
（晋升门按中文对话写，英文原话进不了事实层，所以这里绕过它；本评测只测路由、召回与排序）。
两组都取前 k 条片段（带日期）交给同一个答题模型，再由同一个判分模型对照标准答案判对错。

召回模式（--mode）：
  legacy   旧路径：retrieve.retrieve（规则路由 + 分层检索），即 ENGRAM_RECALL_V2 不设时钩子用的检索
  hybrid   新路径候选池：recall_flow.topk（专名直通 + FTS + 向量 + RRF），可选 --rewrite 查询改写
  1        新路径 + 交叉编码器：候选池（≤40）经 bge-reranker-v2-m3 重排后取前 k
  bm25     只跑 BM25 对照组
--arms 决定跑哪几组，默认「所选模式 + bm25」。

用法见同目录 README.md。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


# ── 数据 ───────────────────────────────────────────────────────────────────
def load_items(path: str, qtypes: list[str] | None, n: int, seed: int) -> list[dict]:
    items = json.load(open(path, encoding="utf-8"))
    if qtypes:
        items = [x for x in items if x.get("question_type") in qtypes]
    items = sorted(items, key=lambda x: x["question_id"])
    rng = random.Random(seed)
    if 0 < n < len(items):
        items = rng.sample(items, n)
    return items


def parse_date(s: str) -> str:
    """'2023/07/15 (Sat) 13:17' → ISO（UTC）。"""
    day, rest = s.split(" (")[0], s.split(") ")[1]
    return dt.datetime.strptime(f"{day} {rest}", "%Y/%m/%d %H:%M").replace(tzinfo=dt.timezone.utc).isoformat()


# ── Lumen 组 ───────────────────────────────────────────────────────────────
def build_db(item: dict, db: str, vectors: bool):
    import persona
    from classify import classify
    from store.store import add_evidence, connect, promote, record_tier
    con = connect(db)
    user = persona.USER
    for sess, date, sid in zip(item["haystack_sessions"], item["haystack_dates"], item["haystack_session_ids"]):
        occ = parse_date(date)
        for ti, turn in enumerate(sess):
            actor = user if turn["role"] == "user" else "claude"
            ev = add_evidence(con, source_kind="chat", source_actor=actor, source_path=sid,
                              source_locator=f"{sid}#t{ti}", content=turn["content"], occurred_at=occ,
                              verification_status="verified")
            record_tier(con, ev, "primary", f"{sid}#t{ti}")
            if actor == user:
                ft, conf, _ = classify(turn["content"])
                promote(con, subject=user, fact_type=ft, content=turn["content"], evidence_ids=[ev],
                        valid_from=occ, confidence=conf)
    con.commit()
    n_vec = 0
    if vectors:
        import vectors as V
        n_vec = V.build(con)
        con.commit()
    return con, n_vec


def _date(h: dict) -> str:
    return (h.get("valid_from") or h.get("occurred_at") or "")[:10]


def lumen_hits(item: dict, mode: str, k: int, *, vectors: bool, rewrite: bool, xscore) -> tuple[list, dict]:
    fd, db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        con, n_vec = build_db(item, db, vectors)
        meta = {"n_vectors": n_vec} if vectors else {}
        q = item["question"]
        if mode == "legacy":
            import persona
            import retrieve as R
            r = R.retrieve(con, q, who=persona.USER)
            hits = r.hits[:k]
            meta.update({"should_recall": r.should_recall, "depth": r.depth, "terms": list(r.terms),
                         "n_pool": len(r.hits)})
        else:
            import recall_flow as RF
            pool = RF.topk(con, q, k=RF.FULL_POOL_CAP if mode == "1" else k, rewrite=rewrite,
                           autobuild_entities=True)
            meta["n_pool"] = len(pool)
            if mode == "1":
                s = xscore(q, [(h.get("content") or "") for h in pool]) if pool else []
                if s is None:
                    raise RuntimeError("交叉编码器不可用：配置 XRERANK_URL / XRERANK_API_KEY，或 --xrerank local，或 --mock")
                order = RF.order_by_xscore(pool, s)
                pool = [dict(pool[i], xscore=round(s[i], 4)) for i in order]
                meta["top_xscores"] = [h["xscore"] for h in pool[:k]]
            hits = pool[:k]
        con.close()
        out = []
        for h in hits:
            c = h.get("content") or ""
            if h.get("superseded_by"):
                c = "【较早版本，可能已被更新】" + c
            out.append((_date(h), c))
        return out, meta
    finally:
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(db + suf)
            except OSError:
                pass


# ── BM25 对照组 ────────────────────────────────────────────────────────────
def bm25_hits(item: dict, k: int) -> tuple[list, dict]:
    con = sqlite3.connect(":memory:")
    con.execute("create virtual table t using fts5(content, date unindexed)")
    for sess, date in zip(item["haystack_sessions"], item["haystack_dates"]):
        for turn in sess:
            if turn["role"] == "user":
                con.execute("insert into t values (?,?)", (turn["content"], parse_date(date)[:10]))
    words = [w for w in re.findall(r"[A-Za-z0-9&']+", item["question"]) if len(w) > 2]
    q = " OR ".join('"%s"' % w.replace('"', "") for w in words) or '""'
    try:
        rows = con.execute("select date, content from t where t match ? order by bm25(t) limit ?", (q, k)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    return [(d, c) for d, c in rows], {"query": q}


# ── 交叉编码器 ─────────────────────────────────────────────────────────────
def make_xscore(kind: str):
    if kind == "mock":
        def mock(q, docs):
            qs = set(re.findall(r"[a-z0-9]+", q.lower()))
            return [len(qs & set(re.findall(r"[a-z0-9]+", d.lower()))) / (1 + len(qs)) for d in docs]
        return mock
    if kind == "local":
        from sentence_transformers import CrossEncoder   # pip install sentence-transformers
        model = CrossEncoder(os.environ.get("XRERANK_MODEL") or "BAAI/bge-reranker-v2-m3", max_length=512)

        def local(q, docs):
            import xrerank as _xr
            raw = model.predict([(q, _xr.best_window(d, q)) for d in docs])
            return [1 / (1 + math.exp(-float(x))) if not 0 <= float(x) <= 1 else float(x) for x in raw]
        return local
    import xrerank
    return xrerank.scores


# ── 答题与判分 ─────────────────────────────────────────────────────────────
def chat(prompt: str) -> str:
    url = os.environ.get("EVAL_BASE_URL") or os.environ.get("DIGEST_BASE_URL")
    key = os.environ.get("EVAL_API_KEY") or os.environ.get("DIGEST_API_KEY")
    model = os.environ.get("EVAL_MODEL") or os.environ.get("DIGEST_MODEL")
    if not (url and key and model):
        raise SystemExit("缺 EVAL_BASE_URL / EVAL_API_KEY / EVAL_MODEL（或 --mock）")
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 300}).encode()
    err = None
    for i in range(4):
        try:
            req = urllib.request.Request(url, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip()
        except Exception as e:                           # noqa: BLE001
            err = e
            time.sleep(3 * (i + 1))
    return f"[llm error: {err}]"


def strip_thinking(text: str) -> str:
    """去掉推理模型输出里的思考标记，只留最终答案。

    DeepSeek-V4-Flash 这类推理模型常在答案前输出思考过程，判分前必须去掉，
    否则判分模型会被思考里的旧值、新值混淆：
      · 成对的 <think>...</think> 整段删（可多段）；
      · 只有孤立的 </think> 时，删掉它之前的内容（那是没打 <think> 的思考）；
      · 只有孤立的 <think> 时，删掉它及之后的内容（没闭合的思考，没有最终答案）。
    没有标记时原样返回（只去首尾空白）。"""
    if not text:
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    low = out.lower()
    if "</think>" in low:
        idx = low.rindex("</think>")
        out = out[idx + len("</think>"):]
        low = out.lower()
    if "<think>" in low:
        out = out[:low.index("<think>")]
    return out.strip()


def answer(item: dict, hits: list, mock: bool) -> str:
    if mock:
        return strip_thinking(hits[-1][1][:200]) if hits else "I don't know."
    ctx = "\n\n".join(f"[{d}] {c[:1500]}" for d, c in sorted(hits)) or "(nothing retrieved)"
    return strip_thinking(chat("You are answering a question about the user's own past, using only these retrieved memory snippets "
                "(the user's past messages, with dates). If facts changed over time, use the most recent. "
                f"Today is {item['question_date']}.\n\nSnippets:\n{ctx}\n\nQuestion: {item['question']}\n"
                "Answer concisely. If the snippets don't contain it, say you don't know."))


def judge(item: dict, pred: str, mock: bool) -> bool:
    if mock:
        return str(item["answer"]).lower() in pred.lower()
    extra = (" This is a knowledge-update question: the model must give the UPDATED (latest) value."
             if item.get("question_type") == "knowledge-update" else "")
    v = chat(f"Question: {item['question']}\nGold answer: {item['answer']}\nModel answer: {pred}\n\n"
             f"Does the model answer match the gold answer in meaning?{extra} Reply with only yes or no.")
    return v.lower().startswith("yes")


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                        text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:                                    # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="LongMemEval json（如 longmemeval_s_cleaned.json）")
    ap.add_argument("--mode", choices=["legacy", "hybrid", "1", "bm25"], default="1")
    ap.add_argument("--arms", default=None, help="逗号分隔，默认「mode,bm25」")
    ap.add_argument("--types", default="", help="题型，逗号分隔，如 knowledge-update,temporal-reasoning；空＝全部")
    ap.add_argument("-n", type=int, default=30, help="抽多少题（0＝全部）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--vectors", action="store_true", help="用 bge-m3 向量（需 CF_API_TOKEN / CF_ACCOUNT_ID）")
    ap.add_argument("--rewrite", action="store_true", help="hybrid 模式下开查询改写（会调模型）")
    ap.add_argument("--xrerank", choices=["api", "local"], default="api", help="mode=1 的交叉编码器来源")
    ap.add_argument("--mock", action="store_true", help="不调任何模型：答题取最后一条片段、判分做子串匹配、交叉编码器用词重合")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    os.environ.setdefault("ENGRAM_LLM_USAGE", "0")   # 评测不往用量表记账

    arms = [x for x in (a.arms or (a.mode if a.mode == "bm25" else f"{a.mode},bm25")).split(",") if x]
    items = load_items(a.data, [t for t in a.types.split(",") if t], a.n, a.seed)
    xscore = make_xscore("mock" if a.mock else a.xrerank) if "1" in arms else None
    vector_model = None
    if a.vectors:
        import vectors as _V
        vector_model = _V.MODEL
    meta = {"data": os.path.basename(a.data), "types": a.types or "all", "n": len(items), "seed": a.seed, "k": a.k,
            "arms": arms, "vectors": a.vectors, "vector_model": vector_model, "rewrite": a.rewrite,
            "xrerank": ("mock" if a.mock else a.xrerank) if "1" in arms else None,
            "xrerank_model": os.environ.get("XRERANK_MODEL") or "BAAI/bge-reranker-v2-m3",
            "answer_judge_model": "mock" if a.mock else (os.environ.get("EVAL_MODEL") or os.environ.get("DIGEST_MODEL")),
            "git_commit": git_commit(), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "question_ids": [x["question_id"] for x in items]}
    rows = []
    for i, it in enumerate(items):
        row = {"id": it["question_id"], "type": it.get("question_type"), "q": it["question"], "gold": it["answer"]}
        for arm in arms:
            t0 = time.time()
            if arm == "bm25":
                hits, m = bm25_hits(it, a.k)
            else:
                hits, m = lumen_hits(it, arm, a.k, vectors=a.vectors, rewrite=a.rewrite, xscore=xscore)
            pred = answer(it, hits, a.mock)
            row[arm] = {"pred": pred, "ok": judge(it, pred, a.mock), "n_hits": len(hits), "meta": m,
                        "secs": round(time.time() - t0, 1), "hits": hits}
        rows.append(row)
        print(i, it.get("question_type"), " ".join(f"{arm}={row[arm]['ok']}" for arm in arms), "|", it["question"][:60],
              flush=True)
        summary = {arm: {"correct": sum(r[arm]["ok"] for r in rows), "total": len(rows),
                         "by_type": {t: f"{sum(r[arm]['ok'] for r in rows if r['type'] == t)}/"
                                        f"{sum(1 for r in rows if r['type'] == t)}"
                                     for t in sorted({r['type'] for r in rows})}} for arm in arms}
        json.dump({"meta": meta, "summary": summary, "results": rows}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    for arm in arms:
        s = summary[arm] if rows else {"correct": 0, "total": 0}
        print(arm, s["correct"], "/", s["total"], s.get("by_type", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
