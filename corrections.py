#!/usr/bin/env python3
"""她纠正过我的话 —— 单独存、钉住、当考题。

2026-09-19 定的：一万条里最值钱的是她纠正我的那些。每一条都是「我以为 A，其实是 B」，
时间出处都齐。以前它们跟「日常闲聊」混在一起，没有任何特殊待遇。

三步，可以分开跑：
  mine     从会话 jsonl 里挖：她的 TG 消息里带纠正口气的，连同我前一句、她再前一句
  classify 交给模型逐条判是不是真纠正，是的话抽出 wrong / right / topic
  refine   再筛一遍：只留过几周还站得住的（她的事、我们做的东西、她定的规矩），
           聊天当下的误会（看错字、会错一句话的意思）不钉；顺手改成「我/她」的口吻
  promote  留下的写进 facts：confirmed + user_pinned，证据挂到原始 jsonl 行
结果落在 corrections.jsonl（明文，是这一层的真源；库里那份删了能从它重建）。

    python3 corrections.py mine
    python3 corrections.py classify
    python3 corrections.py refine
    python3 corrections.py promote
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

SESSIONS = Path(persona.get("ENGRAM_TRANSCRIPT_DIR", str(Path.home() / ".claude/projects")))
CANDIDATES = ROOT / "corrections-candidates.jsonl"
OUT = ROOT / "corrections.jsonl"

RE_CHANNEL = re.compile(r'<channel[^>]*\bts="([^"]+)"[^>]*>(.*?)</channel>', re.S)
# 初筛宁宽勿漏，真假交给模型判
RE_CORRECT = re.compile(
    r"不是|不对|错了|记错|说反|搞错|弄错|纠正|作废|瞎说|乱说|胡说|编的|"
    r"才不是|没有啊|哪有|我没说|不是这样|看反|想多|脑补")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return ""


def mine() -> None:
    seen, rows = set(), []
    for path in sorted(glob.glob(str(SESSIONS / "*.jsonl"))):
        prev_her, my_last = "", ""
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for lineno, line in enumerate(fh, 1):
                if '"reply"' in line and '"tool_use"' in line:
                    try:
                        d = json.loads(line)
                        for b in d.get("message", {}).get("content", []):
                            if b.get("type") == "tool_use" and b.get("name", "").endswith("reply"):
                                my_last = b.get("input", {}).get("text", "") or my_last
                    except Exception:                    # noqa: BLE001
                        pass
                    continue
                if '"type":"user"' not in line or "<channel" not in line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:                        # noqa: BLE001
                    continue
                for ts, body in RE_CHANNEL.findall(_text(d.get("message", {}).get("content"))):
                    body = body.strip()
                    key = (ts, body[:80])
                    if RE_CORRECT.search(body) and my_last and key not in seen:
                        seen.add(key)
                        rows.append({"ts": ts, "source_path": path, "line": lineno,
                                     "her": body[:600], "my_prev": my_last[:600],
                                     "her_prev": prev_her[:300]})
                    prev_her = body
    with open(CANDIDATES, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"候选 {len(rows)} 条 → {CANDIDATES.name}")


PROMPT = """下面每组是一段聊天：AI 先说了一句，用户接着回了一句。判断用户那句是不是在**纠正 AI 说错的事实**
（AI 记错了、说反了、搞混了人/时间/事、编了没有的事、误解了她的意思，她指出来）。
开玩笑、反问撒娇、讨论第三方、纠正别的 AI、对计划改主意，都不算。

是纠正的，抽出：
- wrong：AI 原来以为的（一句话，具体）
- right：实际情况（一句话，具体，按用户说的）
- topic：这件事的主题词，3~8 个字
- ask：如果以后再聊到这件事，用户可能怎么问（一句自然的口语）

{groups}

只输出 JSON：{{"items": [{{"i": 编号, "is": true或false, "wrong": "", "right": "", "topic": "", "ask": ""}}, ...]}}，每组都要有。"""


def classify(batch: int = 8) -> None:
    from llm import chat                                # 共用模型调用（主→备用）
    rows = [json.loads(x) for x in open(CANDIDATES, encoding="utf-8")]
    done = {}
    if OUT.exists():
        for x in open(OUT, encoding="utf-8"):
            r = json.loads(x)
            done[(r["ts"], r["her"][:80])] = r
    todo = [r for r in rows if (r["ts"], r["her"][:80]) not in done]
    judged = ROOT / "corrections-judged.jsonl"
    if judged.exists():
        judged_keys = {tuple(json.loads(x)["key"]) for x in open(judged, encoding="utf-8")}
        todo = [r for r in todo if (r["ts"], r["her"][:80]) not in judged_keys]
    print(f"待判 {len(todo)} 条")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        groups = "\n\n".join(
            f"[{i}]\nAI：{r['my_prev'][:400]}\n用户：{r['her'][:400]}" for i, r in enumerate(chunk))
        try:
            text = chat([{"role": "system", "content": "你只输出合法 JSON。"},
                         {"role": "user", "content": PROMPT.format(groups=groups)}],
                        purpose="corrections", max_tokens=1500, timeout=60)
            items = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["items"]
        except Exception as e:                           # noqa: BLE001
            print(f"  第 {s} 批失败，下次重跑补：{e}")
            continue
        by_i = {int(it.get("i", -1)): it for it in items}
        with open(judged, "a", encoding="utf-8") as jf, open(OUT, "a", encoding="utf-8") as of:
            for i, r in enumerate(chunk):
                it = by_i.get(i)
                if it is None:
                    continue                            # 漏判的不记，下次重跑再判
                jf.write(json.dumps({"key": [r["ts"], r["her"][:80]], "is": bool(it.get("is"))},
                                    ensure_ascii=False) + "\n")
                if it.get("is") and it.get("wrong") and it.get("right"):
                    r.update({k: it.get(k, "") for k in ("wrong", "right", "topic", "ask")})
                    of.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {min(s + batch, len(todo))}/{len(todo)}")


REFINE = """下面是 AI 被用户（她）纠正过的记录。逐条判断 keep：
这条纠正过几周后还有用吗？
- 有用（keep=true）：关于她本人（经历、家人、喜好、习惯、计划、学校、身体）、她身边的人和宠物、
  我们一起做过的东西（项目、工具、谁做的、怎么做的）、她定下的规矩、发生过的事的时间和经过、AI 的身份设定。
- 没用（keep=false）：聊天当下的误会（看错字、会错一句话的意思、以为她在忙）、玩笑、
  只对那一刻有意义的状态（额度还剩多少、她这会儿在干嘛）。
从严：拿不准的一律 false。「她昨晚去哪了」「她当时在哪个 app」「她那句话指的是什么」这类一次性的，都是 false。
判断标准：一个月后我在别的话题里重新提到这件事，还会不会再犯同样的错？会，才 keep。
keep 的，把 wrong/right 改写成第一人称：AI 是「我」，用户是「她」，一句话，具体到能单独看懂。

{items}

只输出 JSON：{{"items": [{{"i": 编号, "keep": true或false, "wrong": "我以为……", "right": "其实……"}}, ...]}}，每条都要有。"""


def refine(batch: int = 10) -> None:
    from llm import chat
    rows = [json.loads(x) for x in open(OUT, encoding="utf-8")]
    todo = [i for i, r in enumerate(rows) if "keep" not in r]
    print(f"待筛 {len(todo)} 条")
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        items = "\n".join(
            f"[{k}] 主题：{rows[i]['topic']}｜AI 以为：{rows[i]['wrong']}｜实际：{rows[i]['right']}"
            f"｜她原话：{rows[i]['her'][:150]}" for k, i in enumerate(idx))
        try:
            text = chat([{"role": "system", "content": "你只输出合法 JSON。"},
                         {"role": "user", "content": REFINE.format(items=items)}],
                        purpose="corrections", max_tokens=2000, timeout=90)
            got = {int(it["i"]): it for it in
                   json.loads(re.search(r"\{.*\}", text, re.S).group(0))["items"]}
        except Exception as e:                           # noqa: BLE001
            print(f"  第 {s} 批失败，下次重跑补：{e}")
            continue
        for k, i in enumerate(idx):
            it = got.get(k)
            if it is None:
                continue
            rows[i]["keep"] = bool(it.get("keep"))
            if rows[i]["keep"] and it.get("wrong") and it.get("right"):
                rows[i]["wrong"], rows[i]["right"] = it["wrong"], it["right"]
        with open(OUT, "w", encoding="utf-8") as fh:     # 每批落一次盘，断了能续
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {min(s + batch, len(todo))}/{len(todo)}")
    print(f"留下 {sum(1 for r in rows if r.get('keep'))}/{len(rows)}")


def promote_all() -> None:
    from store.store import add_evidence, connect, promote
    con = connect()
    n = 0
    for x in open(OUT, encoding="utf-8"):
        r = json.loads(x)
        if not r.get("keep"):
            continue
        ev = add_evidence(con, source_kind="chat", source_actor=persona.USER,
                          source_path=r["source_path"], source_locator=f"line:{r['line']}",
                          content=r["her"], occurred_at=r["ts"])
        day = r["ts"][:10]
        content = f"【她纠正过我·{day}·{r['topic']}】{r['wrong']}；{r['right']}"
        fid = promote(con, subject=persona.USER, fact_type="semantic", content=content,
                      evidence_ids=[ev], status="confirmed", valid_from=r["ts"],
                      importance=1.0, confidence=1.0, user_pinned=True)
        n += fid is not None
    con.commit()
    print(f"新写入 {n} 条纠正（已有的跳过）")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"mine": mine, "classify": classify, "refine": refine, "promote": promote_all}.get(
        cmd, lambda: print(__doc__))()
