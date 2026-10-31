"""召回时的「新事盖旧事」提示（不改库）。

曾试过入库时自动封旧事（extract/timeline.py），判得不准就停了。
当时定的方向是：不自动封，召回时把可能更新的那条一起给，标出来让我自己看。
例：旧记录「用旧背包」还活着，
后来「背包弄脏扔了」也在——两条都该给，并且标清哪条是新的。

做法：对要注入的每条，找同一个 subject、发生得更晚、关键词沾边的几条，
让模型判「新的这条有没有让旧的不再成立」。判不出来、超时，一律当没有，照常注入。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router import keywords  # noqa: E402
from llm import chat  # noqa: E402

import persona  # noqa: E402
ENV_FILE = persona.env_file()
TIMEOUT_S = 6
MAX_NEWER = 5
STOP_GRAMS = {persona.N_USER, *(persona.N_CLAUDE[i:i + 2] for i in range(len(persona.N_CLAUDE) - 1)), "我们", "觉得", "喜欢", "现在", "之前", "因为", "所以", "已经", "还是", "一个", "这个", "那个", "可以", "没有", "不是", "就是", "她的", persona.N_USER[-1:] + "说", "时候", "什么", "自己"}
STOP = {persona.N_USER, persona.N_CLAUDE, "我们", "觉得", "喜欢", "说", "现在", "之前", "因为", "所以", "已经", "还是", "一个", "这个", "那个", "可以", "没有", "不是", "就是"}

PROMPT = """记忆库里有一条旧记录，和几条时间更晚的记录。判断：更晚的哪一条让旧记录说的事**不再成立**了。

旧记录（{old_when}）：{old}

更晚的记录：
{news}

只有这些算不再成立：状态变了（有→没了、在用→扔了、打算A→改成B）、说法被更正、同一个问题有了新答案。
只是补充、细化、同一话题的另一件事，都不算。拿不准就答 null。

只输出 JSON：{{"index": 编号或null, "why": "一句话"}}"""


from llm import load_env as _env  # noqa: E402,F401  共用配置读取（见 llm.py）


def _grams(text: str) -> set[str]:
    out = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        out |= {run[i:i + 2] for i in range(len(run) - 1)}
    return {g for g in out if g not in STOP_GRAMS}


def later_siblings(con, hit: dict) -> list[dict]:
    """同一个 subject、晚于这条、跟它共享至少 3 个有信息量的二字词的几条。"""
    when = hit.get("valid_from")
    subject = hit.get("subject")
    if not when or not subject:
        return []
    mine = _grams(hit.get("content") or "")
    if len(mine) < 3:
        return []
    rows = con.execute(
        """SELECT ext_id, content, valid_from, subject FROM facts
           WHERE subject = ? AND status IN ('candidate','confirmed') AND valid_to IS NULL
             AND valid_from > datetime(?, '+12 hours')
             AND content NOT LIKE '我：%' AND content NOT LIKE ?
             AND content NOT LIKE ?""", (subject, when, f"{persona.N_CLAUDE}（%", f"{persona.N_USER}（%")).fetchall()
    scored = []
    for r in rows:
        if (r["content"] or "").strip() == (hit.get("content") or "").strip():
            continue
        ov = len(mine & _grams(r["content"] or ""))
        if ov >= 3:
            scored.append((ov, r["valid_from"], dict(r)))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored[:MAX_NEWER]]


def superseded_by(con, hit: dict) -> dict | None:
    """返回让这条旧事不再成立的那条更新记录；没有、判不出、出错都返回 None。"""
    news = later_siblings(con, hit)
    if not news:
        return None
    try:
        text = chat([
            {"role": "system", "content": "你只输出合法 JSON。"},
            {"role": "user", "content": PROMPT.format(
                old_when=(hit.get("valid_from") or "")[:10], old=(hit.get("content") or "")[:200],
                news="\n".join(f"[{i}]（{(n['valid_from'] or '')[:10]}）{n['content'][:160]}"
                               for i, n in enumerate(news)))},
        ], purpose="rerank", max_tokens=80, timeout=TIMEOUT_S)
        m = re.search(r"\{.*\}", text, re.S)
        got = json.loads(m.group(0) if m else text)
        idx = got.get("index")
        if isinstance(idx, int) and 0 <= idx < len(news):
            return news[idx] | {"why": str(got.get("why") or "")[:60]}
    except Exception:                                        # noqa: BLE001
        pass
    return None
