"""按意思找（向量检索）。治「换个说法就找不到」：用户说「买便宜点」库里写「打折」，说「这个价」库里写「233元」。

嵌入走 Cloudflare Workers AI 的 @cf/baai/bge-m3（中英都行，1024 维），钥匙在 .env.jev。
向量存在 engram.db 的 fact_vec 表（float32 blob），只给活着的 fact 做。
查询时把她这句嵌一下，跟全库算余弦，取前几条交给原来那道模型精排——向量只负责「捞」，不负责「判」。

    python3 vectors.py build        # 补齐还没向量的 fact（可重复跑，只补缺的）
    python3 vectors.py q 一句话      # 试查
任何失败都返回空，调用方照旧走关键词那条路。
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from store.store import connect  # noqa: E402
import redact  # noqa: E402

ENV = ROOT / ".env.jev"
MODEL = "@cf/baai/bge-m3"
BATCH = 50
TIMEOUT_S = 6
_cache: dict = {}


def _creds() -> tuple[str, str] | None:
    env = {}
    try:
        for line in open(ENV, encoding="utf-8"):
            m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if m:
                env[m.group(1)] = m.group(2).strip("\"'")
    except OSError:
        return None
    tok = os.environ.get("CF_API_TOKEN") or env.get("CF_API_TOKEN")
    acct = os.environ.get("CF_ACCOUNT_ID") or env.get("CF_ACCOUNT_ID")
    return (tok, acct) if tok and acct else None


def embed(texts: list[str], timeout: float = TIMEOUT_S) -> np.ndarray | None:
    c = _creds()
    if not c or not texts:
        return None
    tok, acct = c
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run/{MODEL}",
        data=json.dumps({"text": [redact.scrub(t[:500], "embed") for t in texts]}).encode(),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            v = np.asarray(json.load(resp)["result"]["data"], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        return v
    except Exception:                                        # noqa: BLE001
        return None


def ensure_table(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS fact_vec (
        fact_id INTEGER PRIMARY KEY, model TEXT NOT NULL, vec BLOB NOT NULL)""")


def build(con, limit: int = 0) -> int:
    ensure_table(con)
    rows = con.execute(
        """SELECT f.id, f.content FROM facts f LEFT JOIN fact_vec v ON v.fact_id = f.id
           WHERE v.fact_id IS NULL AND f.status IN ('candidate','confirmed') AND f.valid_to IS NULL
           ORDER BY f.id""").fetchall()
    if limit:
        rows = rows[:limit]
    done = 0
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        v = embed([r["content"] or "" for r in chunk], timeout=60)
        if v is None:
            print(f"第 {i} 批失败，跳过")
            continue
        con.executemany("INSERT OR REPLACE INTO fact_vec(fact_id, model, vec) VALUES (?,?,?)",
                        [(r["id"], MODEL, v[j].tobytes()) for j, r in enumerate(chunk)])
        con.commit()
        done += len(chunk)
        if (i // BATCH) % 20 == 0:
            print(f"已做 {done}/{len(rows)}")
    return done


def _matrix(con):
    n = con.execute("SELECT count(*) FROM fact_vec").fetchone()[0]
    if _cache.get("n") != n:
        rows = con.execute(
            """SELECT v.fact_id, v.vec FROM fact_vec v JOIN facts f ON f.id = v.fact_id
               WHERE f.status IN ('candidate','confirmed') AND f.valid_to IS NULL""").fetchall()
        _cache.update(n=n, ids=[r[0] for r in rows],
                      mat=np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
                      if rows else np.zeros((0, 1024), np.float32))
    return _cache["ids"], _cache["mat"]


def search(con, text: str, k: int = 5, floor: float = 0.5) -> list[dict]:
    """按意思找前 k 条，余弦低于 floor 的不要。返回的字段跟 retrieve 的 hits 一样。"""
    try:
        ensure_table(con)
        ids, mat = _matrix(con)
        if not ids:
            return []
        q = embed([text])
        if q is None:
            return []
        sims = mat @ q[0]
        top = np.argsort(-sims)[:k]
        out = []
        for i in top:
            if sims[i] < floor:
                break
            r = con.execute(
                """SELECT id, ext_id, subject, content, status, user_pinned, valid_from,
                          importance, confidence, layer, access_count, last_accessed
                   FROM facts WHERE id = ?""", (ids[i],)).fetchone()
            if r:
                out.append(dict(r) | {"score": float(sims[i]), "my_stance": None, "via": "vector"})
        return out
    except Exception:                                        # noqa: BLE001
        return []


if __name__ == "__main__":
    con = connect()
    if sys.argv[1:2] == ["build"]:
        print("补了", build(con, int(sys.argv[2]) if len(sys.argv) > 2 else 0))
    elif sys.argv[1:2] == ["q"]:
        for h in search(con, " ".join(sys.argv[2:]), k=8, floor=0):
            print(round(h["score"], 3), h["content"][:80])
