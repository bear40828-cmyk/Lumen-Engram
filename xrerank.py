#!/usr/bin/env python3
"""Cross-encoder rerank for chat recall (full-on path).

Pattern copied from Graphiti's BGERerankerClient / mem0's reranker step:
hybrid retrieval (BM25 + vector, RRF) -> cross-encoder scores every
(query, candidate) pair -> keep results above a min score (Graphiti
`reranker_min_score`), top-k. No LLM judge, no "need" flag: if nothing
clears the threshold, nothing is injected.

Model: BAAI/bge-reranker-v2-m3 through the OpenAI-style /v1/rerank endpoint
of the provider already configured in DIGEST_BASE_URL (same key). Override
with XRERANK_MODEL / XRERANK_URL. Returns None on any failure.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

import llm

MODEL = "BAAI/bge-reranker-v2-m3"
TIMEOUT_S = 3.0
PER_CANDIDATE = 300


def best_window(doc: str, query: str, limit: int = PER_CANDIDATE) -> str:
    """给交叉编码器的片段：取含查询词最密的 limit 字窗口，而不是前 limit 字。

    事实常是一大段话，关键的新值（如 "for 9 months"）可能埋在 300 字以后；
    直接截前 300 字会把新值截掉，精排就永远看不到它。先按句切分，取查询词
    命中最多的那句，再前后补足到 limit 字；没有命中时回退到前 limit 字。"""
    if len(doc) <= limit:
        return doc
    terms = {w for w in re.findall(r"[A-Za-z0-9']+", query.lower()) if len(w) > 2}
    sents = [s for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]
    if not sents:
        return doc[:limit]

    def hits(s: str) -> int:
        low = s.lower()
        return sum(low.count(t) for t in terms)
    scored = [(hits(s), i) for i, s in enumerate(sents)]
    best_hits, best_i = max(scored)
    if best_hits == 0:
        return doc[:limit]
    out, n = sents[best_i], len(sents[best_i])
    j = best_i + 1
    while j < len(sents) and n + 1 + len(sents[j]) <= limit:
        out += " " + sents[j]
        n += 1 + len(sents[j])
        j += 1
    j = best_i - 1
    while j >= 0 and n + 1 + len(sents[j]) <= limit:
        out = sents[j] + " " + out
        n += 1 + len(sents[j])
        j -= 1
    return out


def _endpoint(env: dict) -> tuple[str | None, str | None, str]:
    key = llm._get(env, "XRERANK_API_KEY") or llm._get(env, "DIGEST_API_KEY")
    url = llm._get(env, "XRERANK_URL")
    if not url:
        base = llm._get(env, "DIGEST_BASE_URL") or ""
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        url = base + "/rerank" if base else None
    return key, url, llm._get(env, "XRERANK_MODEL") or MODEL


def scores(query: str, docs: list[str], *, timeout: float = TIMEOUT_S) -> list[float] | None:
    """relevance in [0,1] per doc, same order; None on failure."""
    if not docs:
        return []
    env = llm.load_env()
    key, url, model = _endpoint(env)
    if not (key and url):
        return None
    body = json.dumps({"model": model, "query": query[-1000:],
                       "documents": [best_window(d, query) for d in docs],
                       "return_documents": False}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with llm._urlopen(req, timeout) as r:
            got = json.loads(r.read().decode())
        out = [0.0] * len(docs)
        for item in got.get("results", []):
            out[int(item["index"])] = float(item["relevance_score"])
        tok = (got.get("meta") or {}).get("tokens") or {}
        llm._usage("xrerank", "primary", model,
                   {"prompt_tokens": tok.get("input_tokens", 0), "completion_tokens": 0,
                    "total_tokens": tok.get("input_tokens", 0)}, True)
        return out
    except Exception as e:                                   # noqa: BLE001
        llm._usage("xrerank", "primary", model, None, False, type(e).__name__)
        return None
    finally:
        _ = time.perf_counter() - t
