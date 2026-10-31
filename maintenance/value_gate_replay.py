"""值得记闸回放（只读）：把确定性部分（预过滤 + 角色规则 + 后检）套到现有 candidate 事实和它们的证据上，不调模型。
用法：python3 value_gate_replay.py <库副本.db> <输出目录> [上次挡的清单.json] [上次summary.json] —— 只读打开，不写库。输出：
  value-gate-replay.md      打码版（600）
  value-gate-replay-raw.md  原文版（600，只有密钥打码）
  value-gate-replay.json    计数（带「之前」的对照）
旧事实没有 category / quote：category 用 value_gate.infer_category 按关键词推（助手自己立的规矩/承诺算 commitment）；
quote 子串检查对旧事实不适用，跳过。
和真管线对齐：
  · 依据检查看「同一会话前后各 8 条」窗口（真管线是整个片段），她的偏好/生活事实还要有一部分出自她自己的话；
  · 应答/太短的证据留作上下文，只是不能当引文来源：链接证据是「弄好了」这类时，改在邻近同一说话人的有内容的几条里找来源，找不到才挡。
库副本取法（和上次一样）：sqlite3 的 backup API 从正式库只读（mode=ro）拷到 /tmp，权限 600，用完删。"""
import bisect, collections, json, os, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # 仓库根（副本、正式库都能跑）
import sqlite3
import persona
import value_gate as VG
import redact
import actor_fix

DB, OUT = sys.argv[1], Path(sys.argv[2])
PREV_BLOCKED = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
PREV = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
K = 8
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); con.row_factory = sqlite3.Row

def session_key(path, loc):
    loc = loc or ""
    return (path or "", loc.split("#")[0] if loc.startswith("conv:") else "")

# 说话人标注修正层（overlay）：只读库里算出要改判的证据（不写库）；VG_NO_OVERLAY=1 时关掉（对照用）
FIX = {} if os.environ.get("VG_NO_OVERLAY") else actor_fix.compute_all(con)

# 会话：同一文件（导出文件里再按对话 id）按证据 id 排好的 (actor, text)
sess = collections.defaultdict(list)          # key -> [(eid, actor, text)]
def line_no(loc):
    m = re.fullmatch(r"L(\d+)(?:[-:#].*)?", loc or "")
    return int(m.group(1)) if m else None
_ln = {}
for r in con.execute("select id, source_path p, source_locator l, source_actor a, content c from evidence order by id"):
    _ln[r["id"]] = line_no(r["l"])
    if r["id"] in FIX:                                   # 改判过的：按段拆开（同一个 id 的几段挨着放）
        for a2, t2 in actor_fix.segments(FIX[r["id"]], r["a"], r["c"] or "") or [(None, "")]:
            sess[session_key(r["p"], r["l"])].append((r["id"], a2 or "", t2))
    else:
        sess[session_key(r["p"], r["l"])].append((r["id"], r["a"] or "", r["c"] or ""))
pos = {}                                       # eid -> (key, index)
lines_of = {}
for key, arr in sess.items():
    if all(_ln[eid] is not None for eid, _, _ in arr):
        arr.sort(key=lambda x: (_ln[x[0]], x[0]))        # 会话记录文件：按行号排（证据 id 不一定按行号入库）
    lines_of[key] = [(a, t) for _, a, t in arr]
    for i, (eid, _, _) in enumerate(arr):
        pos.setdefault(eid, (key, i))

ev_by_fact = collections.defaultdict(list)
for r in con.execute("""select fe.fact_id, e.id eid, e.source_actor a, e.content c, e.occurred_at t, e.source_path p, e.source_locator l from fact_evidence fe join evidence e on e.id=fe.evidence_id"""):
    e = dict(r)
    fx = FIX.get(e["eid"])
    e["a0"] = e["a"]
    if fx:                                               # 改判：说话人/正文按第一段（leading＝包装里的话；embedded＝前半原说话人）
        segs = actor_fix.segments(fx, e["a"], e["c"] or "")
        e["a"], e["c"] = (segs[0] if segs else (None, ""))
        e["fx"] = fx
    ev_by_fact[r["fact_id"]].append(e)
rows = con.execute("""select f.id, f.subject, f.content, f.fact_type, f.created_at from facts f where f.status='candidate' and f.user_pinned=0 order by f.id""").fetchall()

res = []
for f in rows:
    evs = ev_by_fact.get(f["id"], [])
    actors = set()
    for e in evs:
        actors |= actor_fix.actors_of(e.get("fx"), e["a0"]) if e.get("fx") else ({e["a"]} if e["a"] else set())
    why = None
    use = evs                                  # 实际拿来查的证据：[(actor, text)]
    # 1) 预过滤：心跳/工具输出/JSON/Error 整条不要；应答/太短留作上下文，只是不能当引文来源
    pre = [VG.prefilter(e["c"] or "", e["a"]) for e in evs]
    if evs and all(pre):
        hard = [x for x in pre if x in VG.HARD_PRE]
        if hard:
            why = hard[0]
        else:
            srcs = []
            for e in evs:
                key, i = pos[e["eid"]]
                j = VG.quote_source(f["content"], lines_of[key], i, e["a"], K)
                if j is not None:
                    srcs.append({"a": e["a"], "c": lines_of[key][j][1], "t": e["t"], "eid": e["eid"], "p": e["p"], "l": e["l"], "_j": j})
            if srcs:
                use = srcs
            else:
                why = pre[0]
    cat = VG.infer_category(f["content"], actors, f["subject"], f["fact_type"])
    if not why:
        confirmed = True
        if f["subject"] == "us" and len(evs) >= 2:
            lt = max((e["t"] or "" for e in evs if e["a"] == persona.USER), default="")
            ct = max((e["t"] or "" for e in evs if e["a"] != persona.USER), default="")
            confirmed = lt >= ct
            if confirmed:      # (a) 她最后那句得是同意（共用同意判定：明确同意或委托式）
                last = max((e for e in evs if e["a"] == persona.USER), key=lambda e: e["t"] or "")
                confirmed = VG.user_consents(last["c"] or "", f["content"])
        us_reject = None
        if f["subject"] == "us" and len(evs) >= 2 and not confirmed:
            # (b) 她先提要求、助手随后明确答应（第四轮放宽）：她最早那条是请求、助手之后那条明确答应、事实出自她的话
            her = sorted((e for e in evs if e["a"] == persona.USER), key=lambda e: e["t"] or "")
            cl = sorted((e for e in evs if e["a"] == "claude" and (e["t"] or "") > (her[0]["t"] or "")), key=lambda e: e["t"] or "") if her else []
            if her and cl:
                us_reject = VG.us_request_then_accept(her[0]["c"] or "", cl[-1]["c"] or "", f["content"])
                confirmed = us_reject is None
        why = VG.role_check(f["subject"], cat, actors, confirmed_by_user_after=confirmed, us_reject=us_reject) if evs else None
    if not why and evs:
        text = "\n".join(e["c"] or "" for e in use)
        ctx, utext = [text], [text if actors == {persona.USER} else "\n".join(e["c"] or "" for e in use if e["a"] == persona.USER)]
        for e in evs:
            key, i = pos[e["eid"]]
            c2, u2 = VG.window(lines_of[key], i, K)
            ctx.append(c2); utext.append(u2)
        fact = {"subject": f["subject"], "content": f["content"]}
        if cat: fact["category"] = cat
        why = VG.postcheck(fact, text, actor=(next(iter(actors)) if len(actors) == 1 else None),
                           context_text="\n".join(ctx), user_text="\n".join(utext))
    res.append({"id": f["id"], "subject": f["subject"], "content": f["content"], "actors": sorted(actors), "reason": why,
                "cat": cat, "created_at": f["created_at"], "evsnip": (evs[0]["c"] if evs else ""),
                "src": (use[0]["c"] if use and use is not evs else None)})

blocked = [r for r in res if r["reason"]]
kept = [r for r in res if not r["reason"]]
by_reason = collections.Counter(r["reason"] for r in blocked)
by_subject = collections.Counter(r["subject"] for r in blocked)
subj_total = collections.Counter(r["subject"] for r in res)
by_subj_reason = collections.defaultdict(collections.Counter)
for r in blocked: by_subj_reason[r["subject"]][r["reason"]] += 1
mism = [r for r in res if r["subject"] != "us" and r["subject"] in persona.ACTORS and r["actors"] and r["subject"] not in r["actors"]]
mism_caught = [r for r in mism if r["reason"]]
mism_role = [r for r in mism if r["reason"] and r["reason"].startswith("role:")]
summary = {"candidates": len(res), "blocked": len(blocked), "kept": len(kept),
           "by_reason": dict(by_reason.most_common()), "by_subject": dict(by_subject), "subject_total": dict(subj_total),
           "mismatch_rows": len(mism), "mismatch_caught_any": len(mism_caught), "mismatch_caught_by_role": len(mism_role),
           "by_subject_reason": {k: dict(v.most_common()) for k, v in by_subj_reason.items()}}
# 之前（上一版闸）对照
prev_b = {int(k): v for k, v in PREV_BLOCKED.items()}
recovered = [r for r in res if str(r["id"]) in PREV_BLOCKED and not r["reason"]]
newly = [r for r in res if str(r["id"]) not in PREV_BLOCKED and r["reason"]] if PREV_BLOCKED else []
if PREV:
    summary["before"] = {k: PREV.get(k) for k in ("candidates", "blocked", "kept", "by_reason", "by_subject", "subject_total",
                                                  "mismatch_rows", "mismatch_caught_any", "mismatch_caught_by_role", "suspect_false_positives")}
    summary["recovered"] = len(recovered); summary["newly_blocked"] = len(newly)
    summary["recovered_by_prev_reason"] = dict(collections.Counter(PREV_BLOCKED[str(r["id"])]["reason"] for r in recovered).most_common())
    summary["newly_blocked_by_reason"] = dict(collections.Counter(r["reason"] for r in newly).most_common())

# 疑似误杀：她自己的话、事实是她的、被「过程/状态/指标/建议…」词表挡掉；或助手规矩类被当 other；她的话被判无依据
fp = []
for r in blocked:
    rs = r["reason"]
    if r["actors"] == [persona.USER] and r["subject"] == persona.USER and rs in VG.LISTS:
        fp.append(("她自己的话被词表挡", r))
    elif r["actors"] == [persona.USER] and r["subject"] == persona.USER and rs == "post:verbatim-question":
        fp.append(("她的话被「照抄的问句」挡", r))
    elif r["subject"] == "claude" and rs == "role:claude-only-category" and re.search(r"以后|规矩|每次|不再|一定|必须|约定", r["content"]):
        fp.append((f"{persona.N_CLAUDE}的规矩类被当 other", r))
    elif rs.startswith("post:ungrounded") and r["actors"] == [persona.USER]:
        fp.append(("她的话被判无依据", r))
summary["suspect_false_positives"] = dict(collections.Counter(k for k, _ in fp))

def gist(s, n=60): 
    s = redact.redact((s or "").replace("\n", " "))[0]
    return s if len(s) <= n else s[:n] + "…"
NAMES = re.compile(persona.mask_re())
def mask(s, n=20):
    s = redact.redact((s or "").replace("\n", " "))[0]
    s = re.sub(r"〔[^〕]*〕", "〔密〕", s)
    s = NAMES.sub("〔名〕", s); s = re.sub(r"\d", "#", s)
    return s[:n]

rnd = random.Random(20261028)
def pick_blocked(k):
    groups = collections.defaultdict(list)
    for r in blocked: groups[r["reason"]].append(r)
    for g in groups.values(): rnd.shuffle(g)
    order = [rs for rs, _ in by_reason.most_common()]
    out, i = [], 0
    while len(out) < k and any(groups[o] for o in order):
        o = order[i % len(order)]
        if groups[o]: out.append(groups[o].pop())
        i += 1
    return out
sb = pick_blocked(15)
sk = rnd.sample(kept, 5)
sr = rnd.sample(recovered, min(15, len(recovered)))
sn = rnd.sample(newly, min(15, len(newly)))
summary["samples_blocked_ids"] = [r["id"] for r in sb]; summary["samples_kept_ids"] = [r["id"] for r in sk]
summary["samples_recovered_ids"] = [r["id"] for r in sr]; summary["samples_newly_blocked_ids"] = [r["id"] for r in sn]

def render(raw: bool):
    g = (lambda s, n=60: gist(s, n)) if raw else (lambda s, n=20: mask(s, n))
    def item(r, extra=""):
        return (f"- [{r['id']}] {r['subject']} · 说话人 {','.join(r['actors']) or '无证据'}{extra} · {g(r['content'])}" +
                (f"\n    证据：{gist(r['evsnip'], 70)}" if raw else ""))
    L = ["# 值得记闸回放（只读，不调模型）" + ("（原文版，只有密钥打码）" if raw else "（打码版）"), "",
         f"候选 {summary['candidates']}，会挡 {summary['blocked']}，会留 {summary['kept']}。", ""]
    if PREV:
        L += [f"上一版：会挡 {PREV['blocked']}，会留 {PREV['kept']}；这一版比上一版多放出 {len(recovered)} 条、新多挡 {len(newly)} 条。", ""]
    L += ["## 按原因（现在 / 之前）", ""]
    pr = (PREV or {}).get("by_reason", {})
    L += [f"- {k}：{v}" + (f"（之前 {pr.get(k, 0)}）" if PREV else "") for k, v in by_reason.most_common()]
    L += [f"- （已不再出现的原因）{k}：0（之前 {v}）" for k, v in pr.items() if k not in by_reason]
    ps = (PREV or {}).get("by_subject", {}); pt = (PREV or {}).get("subject_total", {})
    L += ["", "## 按主体（挡/总，括号是之前）", ""] + [f"- {k}：{by_subject.get(k,0)}/{v}" + (f"（之前 {ps.get(k,0)}/{pt.get(k,v)}）" if PREV else "") for k, v in subj_total.most_common()]
    L += ["", f"## 「证据说话人不含 subject」：共 {len(mism)}，会挡 {len(mism_caught)}（其中角色规则 {len(mism_role)}）", "",
          "## 抽样：会挡的 15 条", ""]
    for r in sb: L.append(item(r, f" · {r['reason']}"))
    L += ["", "## 抽样：会留的 5 条", ""]
    for r in sk: L.append(item(r))
    if PREV:
        L += ["", f"## 抽样：这一版放出来的（之前挡、现在留）{len(recovered)} 条里的 15 条", ""]
        for r in sr: L.append(item(r, f" · 之前 {PREV_BLOCKED[str(r['id'])]['reason']}"))
        L += ["", f"## 抽样：这一版新多挡的（之前留、现在挡）{len(newly)} 条里的 15 条", ""]
        for r in sn: L.append(item(r, f" · {r['reason']}"))
    L += ["", "## 疑似误杀", ""]
    for k, c in collections.Counter(k for k, _ in fp).most_common():
        L.append(f"- {k}：{c} 条" + (f"（之前 {PREV.get('suspect_false_positives', {}).get(k, 0)}）" if PREV else ""))
    for k, r in fp[:40]:
        L.append(f"  - [{r['id']}] {k} · {r['reason']} · {g(r['content'])}")
    L += ["", "## 全部会挡的清单（id · 主体 · 说话人 · 原因 · 摘要）", ""]
    for r in blocked:
        L.append(f"- {r['id']} · {r['subject']} · {','.join(r['actors']) or '-'} · {r['reason']} · {g(r['content'], 60 if raw else 20)}")
    return "\n".join(L) + "\n"

OUT.mkdir(parents=True, exist_ok=True)
for name, raw in (("value-gate-replay.md", False), ("value-gate-replay-raw.md", True)):
    pth = OUT / name
    pth.write_text(render(raw), encoding="utf-8"); os.chmod(pth, 0o600)
(OUT / "value-gate-replay.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"); os.chmod(OUT / "value-gate-replay.json", 0o600)
if os.environ.get("VG_DUMP"):                       # 调试用：整表（含原文）写到 /tmp，600
    pth = Path(os.environ["VG_DUMP"]); pth.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8"); os.chmod(pth, 0o600)
print(json.dumps({k: v for k, v in summary.items() if not k.startswith("samples")}, ensure_ascii=False, indent=1))
