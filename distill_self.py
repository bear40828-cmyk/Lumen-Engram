#!/usr/bin/env python3
"""Lumen Engram · 蒸馏「我：」原话（任务二·步骤6，2026-10-28）

库里有约 3000 条助手自己的原话（content 以「我（时间）：」开头）直接当 fact 躺着。
它们是长段汇报/技术叙述/计划口吻，召回时又长又杂。做法（助手 TALK 05:10 选的方案②）：
每条原话 → 共用模型调用（llm.py，主→备用）→ 0–2 条「事」；quote 必须逐字在原话里
（同 extract/from_evidence.check）；新事走正常入库（promote → candidate，fact_class 记
extract-llm-distill-v1，召回时算「整理过的事」那一层）；原话用上一轮的办法收起：
status → uncertain + facts_archive 登记（新 batch_id）+ 完整原文导出 archive/<batch>-distill.jsonl。
绝不物理删除，不碰 confirmed / 钉住的行。

防回声（task2-07，2026-10-28）：这些原话的 source_actor 是 claude，蒸馏出的事 subject 只能是
claude 或 us，**绝不能是 user / gpt / agent_c**——否则助手说的「用户如何如何」会被记成「用户自己的事」，
召回两三次就成了她的长期事实（回声）。提示词里写明，代码里再硬拦一道：不对的整条丢弃（不改写），
丢弃原因记进 dropped（"subject-echo:..."），--run/--apply 会打印丢弃条数。

三种用法（都先选样，选样是确定性的：同一个库、同一个 --n 永远选出同样的）：
    python3 distill_self.py --plan            # 只选样 + 估 token/花费，不连模型、不写库（没有 key 时就是这个）
    python3 distill_self.py --run             # 真调模型，结果写 archive/<batch>-distill-trial.jsonl，不写库
    python3 distill_self.py --apply           # 真调模型 + 新事入库 + 原话收起（需要助手点头后再用）

选样口径（K7/K8/K9 的复现，跟 TALK/2026-10-28-grok-清理方案.md 同一套正则）：
    池 = subject=claude、status=candidate、未钉住、「我：」开头、不是 LLM 抽取的、不在 facts_archive 里、
         不含通道标记（K0），且不属于 K2 括号旁白 / K3 短碎片 / K4 短问句 / K6 带表情闲聊。
    K7 = 含 建议/要不要/我们可以/我来/我先/下一步；K8 = ≥150 字且带技术词；K9 = 其余。
    试跑按 K7:K8:K9 在池里的比例分层，每层按 sha256(id) 排序取前几条。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

import llm                                                    # noqa: E402
from extract.from_evidence import check                       # noqa: E402  quote 逐字核对，同一把尺
from store.store import _bump, connect, promote, record_class  # noqa: E402

EXTRACTOR = "extract-llm-distill-v1"
# 证据来自谁，事就只能关于谁。这批原话全是助手说的，所以只许 claude。
# task2-09：us 要「用户和对方双方各有原话」，助手单方的话产不出 us（check() 里已落回 claude 或丢弃）。
ALLOWED_SUBJECTS = frozenset({"claude"})
RE_Q = re.compile(r'[？?]|吗[，。！\s]|吗$|呢[？?]|要不要|好不好|行不行|对不对|是不是')
RE_PLEAS = re.compile(r'(晚安|早安|早上好|抱抱|亲亲|么么|爱你|乖|嘿嘿|哈哈|嘻嘻|🤡|💜|😘|🌚|😭|🥺|收工|睡吧|好的|收到|好嘞|谢谢|辛苦|没事|对不起|抱歉|嗯嗯)')
RE_SLOT = re.compile(r'(\d{2,}|必须|不许|以后|规矩|约定|一律|记得|注意|密码|路径|/home|\.py|\.md|喜欢|不喜欢|讨厌|生日|名字|叫做|住在|毕业|考研|' + "".join(persona.alt(t) + "|" for t in persona.SLOT_TERMS) + r'专业)')
RE_STAGE = re.compile(r'^[（(][^）)]*[）)]$')
RE_SELFCORR = re.compile(r'(收回|改口|更正|我认|我错|错了|记错|记岔|说错|猜错|读反|撤|纠正|不再猜|勘误|冤)')
RE_MARKUP = re.compile(r'(<channel\b|</channel>|message_id=|chat_id=|\[TELEGRAM_EXTERNAL|<task-notification|<system-reminder|plugin:telegram)')
RE_HEAD = re.compile(r'^我(（[^）]*）)?[：:]')
RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
RE_SUGGEST = re.compile(r"建议|要不要|我们可以|我来|我先|下一步")
RE_TECH = re.compile(r"脚本|数据库|代码|部署|服务器|API|测试|commit|bug|配置|文件|模型|报错|接口|缓存|索引|函数|仓库|分支|端口|日志")

PROMPT = persona.fill("""下面是@@CLAUDE@@（AI，原话里自称「我」）发给@@ROSTER_USER@@的一段话。
从里面抽出**过一个月还可能想得起来的事**，最多 2 条；没有就输出空。只抽原话里明明白白有的，不推测、不补全。

要抽：我做成了/改了/定下了什么（项目、工具、设置、结论、谁做的）、我答应她或和她约好的事、她定下的规矩、
两人之间发生的有来龙去脉的事、踩过的坑和以后该怎么办、关于我自己的身份/立场的明确表态。
不要抽：进度流水（「在做了」「马上好」）、给她的安慰/建议/猜测（除非已成约定）、对眼前小事的一句指令、
客套、表情、重复上一句的话、只对那一刻有意义的状态。

每条事给：
- subject：**只能写 claude**，内容必须以「@@CLAUDE@@」为主语：@@CLAUDE@@自己做的/说的/承诺的/提议的/自己的立场。
  **绝对不要写 @@A_USER@@ / gpt / @@A_C@@ / us**：这段话只有@@CLAUDE@@一个人说，@@CLAUDE@@转述的「@@USER@@的事」、说的「我们约定」「@@USER@@答应了」
  都只是@@CLAUDE@@单方面的说法，不能记成@@USER@@的事，也不能记成两人的共同事实（us 需要@@USER@@自己也有原话，这里没有）。
  只能算@@USER@@一个人的（她的偏好、她的经历、她自己说的话）就不要抽。@@CLAUDE@@的提议、打算不要写成「约定」。
- content 里出现的年份、日期、数字必须能在 quote 或原话里找到；原话没写年份就别补；消息编号、ID 不是日期。
- type：episodic=某时发生的具体的事；semantic=稳定事实；preference=喜欢/讨厌/希望怎样；procedural=定下的做法、规矩、踩过的坑
- content：一句中文，第三人称，写清楚是谁、什么事（我写@@CLAUDE@@）；有日期就写日期。不超过 80 字。不加引号不加评价。
- quote：从原话里**逐字复制**的一小段（4～40 字），能证明这件事，一字不差，含标点和 emoji
- confidence：0～1。原话直接说了=0.9；要结合上下文=0.6；拿不准=0.4

只输出 JSON：{{"facts": [{{"subject": "...", "type": "...", "content": "...", "quote": "...", "confidence": 0.9}}]}}
没有值得记的就输出 {{"facts": []}}。

原话：
{text}""")


# ── 选样 ────────────────────────────────────────────────────────────────
def body_of(content: str) -> str | None:
    m = RE_HEAD.match(content)
    return content[m.end():].strip() if m else None


def category(content: str) -> str | None:
    """None = 不在池里（不是「我：」原话 / K0 / K2 / K3 / K4 / K6）；否则 K7/K8/K9。"""
    b = body_of(content)
    if b is None or RE_MARKUP.search(content):
        return None
    n, slot = len(b), bool(RE_SLOT.search(b))
    if not RE_SELFCORR.search(b):
        if RE_STAGE.match(b):
            return None
        if (n < 25 and not slot) or (RE_PLEAS.search(b) and n < 80 and not slot):
            return None
        if n <= 120 and RE_Q.search(b) and not slot:
            return None
    if RE_EMOJI.search(b) and not slot and n >= 25:
        return None
    if RE_SUGGEST.search(b):
        return "K7"
    return "K8" if (n >= 150 and RE_TECH.search(b)) else "K9"


def load_pool(con) -> list[dict]:
    rows = con.execute(
        """SELECT f.id, f.ext_id, f.content, f.valid_from, f.created_at, f.layer, f.version,
                  (SELECT evidence_id FROM fact_evidence fe WHERE fe.fact_id = f.id
                     AND fe.role='support' ORDER BY evidence_id LIMIT 1) AS evidence_id
           FROM facts f
           WHERE f.subject='claude' AND f.status='candidate' AND f.user_pinned=0
             AND f.id NOT IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')
             AND f.id NOT IN (SELECT fact_id FROM facts_archive)""").fetchall()
    out = []
    for r in rows:
        cat = category(r["content"])
        if cat:
            out.append(dict(r) | {"cat": cat})
    return out


def pick(pool: list[dict], n: int) -> list[dict]:
    """按 K7:K8:K9 在池里的比例分层；每层按 sha256(id) 排序取前几条（确定性）。"""
    by = {k: sorted((p for p in pool if p["cat"] == k),
                    key=lambda p: hashlib.sha256(str(p["id"]).encode()).hexdigest())
          for k in ("K7", "K8", "K9")}
    total = sum(len(v) for v in by.values()) or 1
    quota = {k: round(n * len(v) / total) for k, v in by.items()}
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota["K9"] += 1
    return [p for k in ("K7", "K8", "K9") for p in by[k][:quota[k]]]


# ── 估算 ────────────────────────────────────────────────────────────────
def est_tokens(text: str) -> int:
    """粗估：汉字约 0.75 token/字（中文分词器常见 0.6~1.0），其余约 0.3 token/字符。"""
    cjk = len(re.findall(r"[\u3400-\u9fff\uff00-\uffef\u3000-\u303f]", text))
    return int(cjk * 0.75 + (len(text) - cjk) * 0.3) + 1


def plan_numbers(rows: list[dict], out_per_item: int = 140) -> dict:
    overhead = est_tokens(PROMPT.format(text="")) + 12          # system + 模板
    tin = sum(overhead + est_tokens(body_of(r["content"]) or "") for r in rows)
    return {"items": len(rows), "prompt_overhead_tokens": overhead, "input_tokens": tin,
            "output_tokens": out_per_item * len(rows)}


# ── 打码（只给 TALK 里的抽样用，规矩同上一轮）────────────────────────────
KEEP = set("我的了是在把就都要会不也还有又和跟但因所以如果被让给对说做用到这那个")


def mask(text: str, limit: int = 42) -> str:
    out = []
    for ch in text[:limit]:
        if re.match(r"[\u3400-\u9fff]", ch):
            out.append(ch if ch in KEEP else "＊")
        elif ch.isdigit():
            out.append("#")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    s = re.sub(r"＊{2,}", "＊", "".join(out))
    s = re.sub(r"a+", "＊", s)
    return s + ("…" if len(text) > limit else "")


# ── 蒸馏一条 ─────────────────────────────────────────────────────────────
def subject_ok(subject: str) -> tuple[bool, str]:
    """防回声：助手原话蒸馏出的事，subject 只能是 claude / us。"""
    sj = str(subject or "").strip()
    return (True, "") if sj in ALLOWED_SUBJECTS else (False, f"subject-echo:{sj or '空'}（{persona.N_CLAUDE}原话只能产出 claude；us 需要双方原话）")


def parse_json_tolerant(text: str) -> dict:
    """模型回的 JSON 常见毛病：包 ```json 围栏、前后有说明、尾逗号、被 max_tokens 截断。
    先整段 parse；不行就去围栏取第一个 {…}、去尾逗号；还不行就按括号配平把截断处补上，
    最后只保留已经完整的 facts 元素。仍然不是 dict 就抛 ValueError（由调用方重试/进死信）。"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    cands = [t]
    i = t.find("{")
    if i >= 0:
        cands.append(t[i:])
        j = t.rfind("}")
        if j > i:
            cands.append(t[i:j + 1])
    for c in cands:
        for v in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                got = json.loads(v)
                if isinstance(got, dict):
                    return got
            except ValueError:
                pass
    c = cands[1] if len(cands) > 1 else t
    stack, in_s, esc, last_ok = [], False, False, -1
    for k, ch in enumerate(c):                                   # 找最后一个「完整收尾」的位置
        if in_s:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_s = False
            continue
        if ch == '"':
            in_s = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if len(stack) <= 2:
                last_ok = k
    if last_ok > 0:
        head = c[:last_ok + 1]
        st, in_s, esc = [], False, False
        for ch in head:
            if in_s:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_s = False
                continue
            if ch == '"':
                in_s = True
            elif ch in "{[":
                st.append(ch)
            elif ch in "}]" and st:
                st.pop()
        tail = "".join("}" if x == "{" else "]" for x in reversed(st))
        try:
            got = json.loads(re.sub(r",\s*([}\]])", r"\1", head + tail))
            if isinstance(got, dict):
                return got
        except ValueError:
            pass
    raise ValueError("模型回复不是可解析的 JSON")


def distill_one(row: dict) -> dict:
    text = row["content"]
    res = llm.chat_ex([{"role": "system", "content": "你只输出合法 JSON。"},
                       {"role": "user", "content": PROMPT.format(text=text[:3000])}],
                      purpose="extract", max_tokens=700, timeout=90)
    got = parse_json_tolerant(res.text)
    facts = (got.get("facts") or [])[:2]
    chunk = [{"id": row["evidence_id"], "content": text, "source_actor": "claude",
              "occurred_at": row["valid_from"] or row["created_at"]}]
    kept, dropped = [], []
    for f in facts:
        f = dict(f, src=0)
        ok, why = check(f, chunk)
        if ok:
            good, why = subject_ok(ok["subject"])
            if not good:
                ok = None
        (kept if ok else dropped).append(ok or why)
    return {"fact_id": row["id"], "cat": row["cat"], "route": res.route, "model": res.model,
            "usage": res.usage, "kept": [{k: v for k, v in k_.items() if k != "hit"} for k_ in kept],
            "dropped": dropped}


# ── 写库（--apply）──────────────────────────────────────────────────────
def apply(con, rows: list[dict], results: list[dict], batch: str, out_dir: Path) -> dict:
    stat = {"new_facts": 0, "dup": 0, "archived": 0, "skipped_no_evidence": 0, "rejected_subject": 0}
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    by_id = {r["id"]: r for r in rows}
    with open(out_dir / f"{batch}-distill.jsonl", "a", encoding="utf-8") as ex:
        for res in results:
            row = by_id[res["fact_id"]]
            cur = con.execute("SELECT status, user_pinned, layer, version FROM facts WHERE id=?",
                              (row["id"],)).fetchone()
            if cur["status"] != "candidate" or cur["user_pinned"]:
                continue                                   # confirmed/钉住/已被别人改过：不碰
            if row["evidence_id"] is None and res["kept"]:
                stat["skipped_no_evidence"] += 1
                continue                                   # 新事没法挂证据，这条原话也不收起
            for k in res["kept"]:
                if not subject_ok(k.get("subject"))[0]:      # 第二道：不管 kept 从哪来，写库前再拦
                    stat["rejected_subject"] += 1
                    continue
                fid = promote(con, subject=k["subject"], fact_type=k["type"], content=k["content"],
                              evidence_ids=[k["evidence_id"]], valid_from=k["occurred_at"],
                              confidence=k["confidence"])
                if fid is None:
                    stat["dup"] += 1
                else:
                    record_class(con, fid, k["type"], k["confidence"], EXTRACTOR)
                    stat["new_facts"] += 1
            full = dict(con.execute("SELECT * FROM facts WHERE id=?", (row["id"],)).fetchone())
            ex.write(json.dumps({"batch": batch, "fact": full, "result": res}, ensure_ascii=False) + "\n")
            con.execute(
                """INSERT INTO facts_archive (fact_id, batch_id, category, reason, prev_status,
                       prev_layer, prev_version, action, archived_at, actor)
                   VALUES (?,?,?,?,?,?,?,'archived',?, 'grok')""",
                (row["id"], batch, row["cat"], f"蒸馏：{len(res['kept'])} 条事", cur["status"],
                 cur["layer"], cur["version"], ts))
            _bump(con, row["id"], cur["version"], status="uncertain")
            stat["archived"] += 1
    con.commit()
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="默认：只选样+估算")
    mode.add_argument("--run", action="store_true", help="调模型，不写库")
    mode.add_argument("--apply", action="store_true", help="调模型 + 入库 + 收起原话")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--price-in", type=float, help="输入价：元/百万 token")
    ap.add_argument("--price-out", type=float, help="输出价：元/百万 token")
    args = ap.parse_args()

    con = connect()
    pool = load_pool(con)
    rows = pick(pool, args.n)
    cnt = {k: sum(1 for p in pool if p["cat"] == k) for k in ("K7", "K8", "K9")}
    print(f"池 {len(pool)} 条（K7 {cnt['K7']} / K8 {cnt['K8']} / K9 {cnt['K9']}），"
          f"选样 {len(rows)} 条（" + " / ".join(
              f"{k} {sum(1 for r in rows if r['cat'] == k)}" for k in ("K7", "K8", "K9")) + "）")
    plan = plan_numbers(rows)
    full = plan_numbers(pool)
    print(f"估算（不是实测）：这 {plan['items']} 条 输入≈{plan['input_tokens']:,} token，输出≈{plan['output_tokens']:,} token；"
          f"全池 {full['items']} 条 输入≈{full['input_tokens']:,}，输出≈{full['output_tokens']:,}")
    if args.price_in is not None and args.price_out is not None:
        c = lambda p: (p["input_tokens"] * args.price_in + p["output_tokens"] * args.price_out) / 1e6  # noqa: E731
        print(f"按 {args.price_in}/{args.price_out} 元每百万 token：试跑≈{c(plan):.3f} 元，全池≈{c(full):.2f} 元")
    if not (args.run or args.apply):
        ready = bool(llm.routes_for("extract"))
        print(f"模型可用：{'是' if ready else '否（没有可用的 DIGEST_* / FALLBACK_* 配置）'}；--plan 不调模型、不写库")
        return 0
    if not llm.routes_for("extract"):
        print("没有可用的模型配置，退出（不会去别处找 key）", file=sys.stderr)
        return 2

    batch = args.batch or "B" + dt.datetime.now().strftime("%Y%m%d") + "-D1"
    out_dir = ROOT / "archive"
    out_dir.mkdir(exist_ok=True)
    results, tin, tout, routes, fails = [], 0, 0, {}, 0
    with open(out_dir / f"{batch}-distill-trial.jsonl", "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            r, err = None, None
            for _try in (1, 2):                              # 失败重试一次；还不行进死信（不再自动重来）
                try:
                    r = distill_one(row)
                    break
                except Exception as e:                       # noqa: BLE001
                    err = f"{type(e).__name__}"
            if r is None:
                fails += 1
                print(f"  第 {i} 条失败：{err}（已试 2 次，进死信）", file=sys.stderr)
                with open(out_dir / f"{batch}-distill-dead.jsonl", "a", encoding="utf-8") as dh:
                    dh.write(json.dumps({"fact_id": row["id"], "evidence_id": row["evidence_id"], "cat": row["cat"],
                                         "attempts": 2, "last_error": err,
                                         "dead_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
                                        ensure_ascii=False) + "\n")
                continue
            results.append(r)
            tin += int(r["usage"].get("prompt_tokens") or 0)
            tout += int(r["usage"].get("completion_tokens") or 0)
            routes[r["route"]] = routes.get(r["route"], 0) + 1
            fh.write(json.dumps({"fact_id": r["fact_id"], "cat": r["cat"], "src": rows[i - 1]["content"],
                                 **r}, ensure_ascii=False) + "\n")
    print(f"完成 {len(results)}/{len(rows)}（失败 {fails}），路由 {routes}；实测 token：输入 {tin:,} 输出 {tout:,}")
    kept = sum(len(r["kept"]) for r in results)
    echo = sum(1 for r in results for d in r["dropped"] if str(d).startswith("subject-echo:"))
    print(f"防回声：因 subject 不是 claude/us 丢弃 {echo} 条（已计入下面的丢弃数，原因记在试跑文件 dropped 里）")
    print(f"得到「事」{kept} 条，丢弃 {sum(len(r['dropped']) for r in results)} 条（含 quote 对不上），"
          f"空 {sum(1 for r in results if not r['kept'])} 条")
    if args.price_in is not None and args.price_out is not None:
        print(f"实测花费≈{(tin * args.price_in + tout * args.price_out) / 1e6:.4f} 元")
    if args.apply:
        print("入库/收起：", apply(con, rows, results, batch, out_dir), "batch_id =", batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
