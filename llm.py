#!/usr/bin/env python3
"""Lumen Engram · 共用的模型调用（主路由 + 备用路由）

之前 from_evidence / timeline / rerank / newer / corrections 各抄了一份「读 .env → 拼 body →
urlopen」。硅基流动余额用完（HTTP 402）那天，抽取整夜空跑，钩子的精排悄悄退回关键词闸。
现在所有调模型的地方都走这一个函数，主路由不行就自动换备用，并记下是哪条路由出的结果。

配置（环境变量优先，其次 .env 文件；本模块**不写、不打印、不复制任何 key**）：
  主路由：DIGEST_API_KEY / DIGEST_BASE_URL（完整的 chat/completions 地址） / 模型名
          模型名按用途取：extract → EXTRACT_MODEL，rerank → RERANK_MODEL，缺了用 DIGEST_MODEL
  备用：  FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL（OpenAI 兼容；
          BASE_URL 可以只写到域名或 /v1，没有 /chat/completions 会自动补）
  .env 文件路径：环境变量 ENGRAM_ENV_FILE，或 persona.env 里的 ENGRAM_ENV_FILE，缺省当前目录 .env（见 persona.py）。

换路由的条件：HTTP 401 / 402 / 403 / 408 / 429 / 5xx、超时、连不上、返回体里没有 choices。
400/404 这类「请求本身有问题」不换（换了也一样）。备用变量没配 → 行为跟以前完全一样。
只重试一次：主 → 备，不来回弹。

日志：每次调用在 llm-route.log 追加一行（时间、用途、走了哪条路由、结果）；日志写失败不影响调用。
**绝不 print 到 stdout**——钩子的 stdout 会被注入对话上下文。
"""
from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import redact  # noqa: E402
import persona  # noqa: E402
DEFAULT_ENV_FILE = persona.env_file()
LOG_FILE = ROOT / "llm-route.log"

PURPOSE_MODEL_VARS = {
    "extract": ("EXTRACT_MODEL",),
    "rerank": ("RERANK_MODEL",),
    # 夜间聚类归纳（maintenance/cluster_summarize.py）：可单独配 CLUSTER_MODEL；没配就退回 EXTRACT_MODEL / DIGEST_MODEL（--apply 时 cluster_summarize 会因此拒绝，要显式配）
    "consolidate": ("CLUSTER_MODEL", "EXTRACT_MODEL"),
    # 任务三：RERANK_FLASH 开了，rerank.py 用这个用途；没配 RERANK_FLASH_MODEL 就退回 RERANK_MODEL / DIGEST_MODEL（行为不变）
    "rerank_flash": ("RERANK_FLASH_MODEL", "RERANK_MODEL"),
    # 2026-10-29: second-opinion pass of extract/timeline.py (ENGRAM_TIMELINE_CONFIRM=1); unset -> EXTRACT_MODEL / DIGEST_MODEL
    "timeline_confirm": ("TIMELINE_CONFIRM_MODEL", "EXTRACT_MODEL"),
}
USAGE_DB = None          # 测试可以指到临时库；默认 ENGRAM_DB 或 ROOT/engram.db
USAGE_FORCE = False      # 测试里要真写时设 True（默认在 unittest 下不写，免得测试往正式库记账）
RETRY_STATUS = {401, 402, 403, 408, 429}


class LLMError(RuntimeError):
    pass


class LLMNotConfigured(LLMError):
    """主路由和备用路由都没配。"""


@dataclass
class Route:
    name: str            # primary / fallback
    url: str
    key: str = field(repr=False)
    model: str


@dataclass
class LLMResult:
    text: str
    route: str
    model: str
    usage: dict
    tried: list[str]


def load_env(path: str | None = None) -> dict:
    out: dict = {}
    try:
        with open(path or os.environ.get("ENGRAM_ENV_FILE") or DEFAULT_ENV_FILE,
                  encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
                if m:
                    out[m.group(1)] = m.group(2).strip("\"'")
    except OSError:
        pass
    return out


def _get(env: dict, name: str) -> str | None:
    return os.environ.get(name) or env.get(name) or None


def _chat_url(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def routes_for(purpose: str, env: dict | None = None) -> list[Route]:
    env = load_env() if env is None else env
    out: list[Route] = []
    model = next((m for v in PURPOSE_MODEL_VARS.get(purpose, ()) if (m := _get(env, v))), None) \
        or _get(env, "DIGEST_MODEL")
    key, url = _get(env, "DIGEST_API_KEY"), _get(env, "DIGEST_BASE_URL")
    if key and url and model:
        out.append(Route("primary", url, key, model))
    fkey, furl, fmodel = (_get(env, "FALLBACK_API_KEY"), _get(env, "FALLBACK_BASE_URL"),
                          _get(env, "FALLBACK_MODEL"))
    if fkey and furl and fmodel:
        out.append(Route("fallback", _chat_url(furl), fkey, fmodel))
    return out


def _urlopen(req, timeout):                      # 单独一层，测试时 mock 它
    return urllib.request.urlopen(req, timeout=timeout)


def _retryable(e: Exception) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in RETRY_STATUS or e.code >= 500
    if isinstance(e, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError)):
        return True
    if isinstance(e, OSError):
        return True
    return isinstance(e, _BadBody)


class _BadBody(Exception):
    pass


def _log(purpose: str, route: str, result: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%F %T')} {purpose} {route} {result}\n")
    except Exception:                            # noqa: BLE001
        pass


def _usage(purpose: str, route: str, model: str, usage: dict | None, ok: bool, err: str | None = None) -> None:
    """把一次调用的 token 追加进 llm_usage。只记账：关了/没表/锁了/任何错误都静默，绝不影响调用，绝不 print。"""
    try:
        if not USAGE_FORCE and any(k in (sys.argv[0] or "") for k in ("unittest", "pytest")):
            return
        if os.environ.get("ENGRAM_LLM_USAGE", "1").strip().lower() in ("0", "false", "off", "no"):
            return
        import sqlite3
        import datetime as _dt
        db = USAGE_DB or os.environ.get("ENGRAM_DB") or str(ROOT / "engram.db")
        u = usage or {}
        con = sqlite3.connect(db, timeout=0.5)
        try:
            con.execute(
                "INSERT INTO llm_usage (ts, purpose, route, model, prompt_tokens, completion_tokens, total_tokens, ok, error)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), purpose, route, model,
                 u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"), 1 if ok else 0,
                 (err or None)))
            con.commit()
        finally:
            con.close()
    except Exception:                            # noqa: BLE001
        pass


def _post(route: Route, messages: list[dict], *, temperature: float, max_tokens: int,
          json_mode: bool, timeout: float) -> tuple[str, dict]:
    # 送出去的每一段文字都先打码（没有开关）：库里老证据里的密钥、她当面发的 key，都不许到模型那边。
    messages = [dict(m, content=redact.scrub(m["content"], "llm")) if isinstance(m, dict) and isinstance(m.get("content"), str) else m
                for m in messages]
    body: dict = {"model": route.model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if route.name == "primary":
        body["enable_thinking"] = False          # 硅基流动特有参数；官方 DeepSeek 不带
    req = urllib.request.Request(
        route.url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {route.key}", "Content-Type": "application/json"})
    with _urlopen(req, timeout) as resp:
        data = json.load(resp)
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise _BadBody("返回体里没有 choices") from None
    usage = data.get("usage") if isinstance(data, dict) else None
    return text, usage if isinstance(usage, dict) else {}


def chat_ex(messages: list[dict], *, purpose: str = "extract", temperature: float = 0,
            max_tokens: int = 1000, json_mode: bool = True, timeout: float = 30) -> LLMResult:
    routes = routes_for(purpose)
    if not routes:
        raise LLMNotConfigured("缺 DIGEST_API_KEY / DIGEST_BASE_URL / 模型名（备用 FALLBACK_* 也没配）")
    tried: list[str] = []
    for i, r in enumerate(routes):
        try:
            text, usage = _post(r, messages, temperature=temperature, max_tokens=max_tokens,
                                json_mode=json_mode, timeout=timeout)
            tried.append(f"{r.name}:ok")
            _log(purpose, r.name, "ok" + (f" (主路由失败后改走备用：{tried[0]})" if i else ""))
            _usage(purpose, r.name, r.model, usage, True)
            return LLMResult(text, r.name, r.model, usage, tried)
        except Exception as e:                   # noqa: BLE001
            reason = (f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError)
                      else type(e).__name__)
            tried.append(f"{r.name}:{reason}")
            _usage(purpose, r.name, r.model, None, False, reason)
            last = i == len(routes) - 1
            if last or not _retryable(e):
                _log(purpose, r.name, f"FAIL {reason}" + ("" if last else "（不可重试，不换路由）"))
                raise
            _log(purpose, r.name, f"FAIL {reason} → 换{routes[i + 1].name}")
    raise LLMError("unreachable")                # pragma: no cover


def chat(messages: list[dict], **kw) -> str:
    return chat_ex(messages, **kw).text
