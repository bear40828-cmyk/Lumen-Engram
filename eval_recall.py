#!/usr/bin/env python3
"""召回考卷：拿她纠正过我的那些事当题，看钩子能不能把「其实是 B」递到我手上。

题目来自 corrections.jsonl，每条出两道：
  · ask     模型替她想的「以后可能怎么问」
  · her_prev 当时她真正问的那句（我就是答这句答错的）
答案是那条纠正写进库里的 fact。走的是和钩子一样的路：
指向词闸 → 路由 → 检索 → 模型精排（≥7 分、前两条），命中才算对。

另外 recall-cases.md 里日常抓到的塞错漏掉，也攒着以后加进来。

    python3 eval_recall.py            # 全卷
    python3 eval_recall.py -n 20      # 抽 20 题
只读：不写遥测、不 touch、不改库。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from rerank import judge                    # noqa: E402
from retrieve import retrieve               # noqa: E402
from store.store import connect             # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hook", Path(persona.get("ENGRAM_HOOK_PATH", str(ROOT / "hooks_engram_recall.py"))))
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def picked(con, text: str) -> tuple[list[str], str]:
    """复刻钩子的选择逻辑，返回会被注入的 fact 内容和卡在哪一步。"""
    # 回放可直接传「最近上下文\n当前短句」；正式 hook 则从 transcript 补齐。
    current = text.rsplit("\n", 1)[-1]
    if hook.skip_input(current):
        return [], "跳过输入"
    if hook.pre_recall_skip(current):
        return [], "转述／玩笑"
    if "\n" in text:
        search_text = text
    else:
        search_text, _ = hook.contextual_query({"prompt": current}, current)
    marks = hook.distinctive_terms(search_text)
    if not marks and not hook.EXPLICIT_RECALL.search(current) and not hook.QUESTION.search(current):
        return [], "无指向词"
    r = retrieve(con, search_text, who=None)
    pinned = hook.pinned_candidates(con, search_text)
    contextual = hook.contextual_candidates(con, search_text)
    if not r.should_recall and (not pinned and not contextual or "软话" in (r.reason or "")):
        return [], f"路由不查:{r.reason}"
    seen = {p["ext_id"] for p in pinned + contextual}
    hits = pinned + contextual + [h for h in (r.hits if r.should_recall else [])
                                  if h.get("ext_id") not in seen]
    if not hits:
        return [], "检索无结果"
    j = judge(search_text, [h.get("content") or "" for h in hits])
    deterministic = [
        h for h in hits
        if hook.deterministic_relevance(search_text, h.get("content") or "")
    ]
    if j is None:
        return ([h.get("content") or "" for h in deterministic[:hook.MAX_HITS]],
                "精排失败，规则兜底" if deterministic else "精排失败")
    forced = [
        i for i, h in enumerate(hits)
        if hook.deterministic_relevance(search_text, h.get("content") or "")
    ]
    if not j["need"] and not forced:
        return [], "精排判不用查"
    if forced:
        top = [hits[i] for i in forced[:hook.MAX_HITS]]
    else:
        top = [hits[i] for sc, i in sorted(
            zip(j["scores"], range(len(hits))), reverse=True)
            if sc >= 7 and hook.entity_compatible(
                search_text, hits[i].get("content") or "")][:hook.MAX_HITS]
    if not top:
        return [], "精排不够分" + ("" if pinned else "（纠正层没捞到）")
    return [h.get("content") or "" for h in top], "塞了别的" + ("" if pinned else "（纠正层没捞到）")


def negative(n: int) -> None:
    """她的日常话大多跟任何一条纠正都无关。塞了纠正进来就算误报。"""
    import glob
    sess = Path(persona.get("ENGRAM_TRANSCRIPT_DIR", str(Path.home() / ".claude/projects")))
    pool = []
    for f in sorted(glob.glob(str(sess / "*.jsonl")), key=lambda p: Path(p).stat().st_mtime)[-6:]:
        for line in open(f, encoding="utf-8", errors="ignore"):
            if '"type":"user"' in line and "<channel" in line:
                from corrections import _text
                for m in hook.RE_CHANNEL.finditer(_text(json.loads(line)["message"]["content"])):
                    b = m.group("body").strip()
                    if 4 <= len(b) <= 200:
                        pool.append(b)
    random.seed(1)
    qs = random.sample(pool, min(n, len(pool)))
    con = connect()
    fp = any_inject = 0
    for q in qs:
        got, _ = picked(con, q)
        any_inject += bool(got)
        bad = [g for g in got if g.startswith("【她纠正过我")]
        if bad:
            fp += 1
            print(f"误报：{q[:40]} → {bad[0][:60]}")
    print(f"\n{len(qs)} 句日常话：塞了纠正 {fp} 句，塞了任何东西 {any_inject} 句")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0)
    ap.add_argument("--hard", action="store_true", help="用换了说法的题（ask_hard）")
    ap.add_argument("--neg", type=int, default=0,
                    help="反面卷：随机抽她 N 句日常话，看会不会乱塞纠正")
    args = ap.parse_args()
    if args.neg:
        return negative(args.neg)
    cases = []
    for x in open(ROOT / "corrections.jsonl", encoding="utf-8"):
        c = json.loads(x)
        if not c.get("keep"):
            continue
        key = c["right"]
        # her_prev 是当时她随口说的上一句，大多跟这件事无关（「锁屏换了给你看」），不当题
        for q in (c.get("ask_hard" if args.hard else "ask"),):
            if q and len(q) >= 4:
                cases.append((q, key, c["topic"]))
    random.seed(0)
    if args.n:
        cases = random.sample(cases, min(args.n, len(cases)))
    con = connect()
    ok, why = 0, {}
    for q, key, topic in cases:
        got, reason = picked(con, q)
        hit = any(key in g for g in got)
        ok += hit
        if not hit:
            why[reason] = why.get(reason, 0) + 1
            print(f"✗ [{topic}] {q[:40]} → {reason}")
    total = len(cases)
    print(f"\n命中 {ok}/{total} = {ok / max(total, 1):.0%}")
    print("没命中卡在：", dict(sorted(why.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
