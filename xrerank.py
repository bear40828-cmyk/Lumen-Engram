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
import time
import urllib.request

import llm

MODEL = "BAAI/bge-reranker-v2-m3"
TIMEOUT_S = 3.0
PER_CANDIDATE = 300


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
                       "documents": [d[:PER_CANDIDATE] for d in docs],
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
