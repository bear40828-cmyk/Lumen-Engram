"""只读扫描（不写库）：库副本里已有事实过一遍实体归属闸（entity_gate），列出 strip/review/reject 的。
用法：python3 maintenance/entity_scan.py <库副本.db> <词表> <out.json>"""
import sys, json, re, collections, sqlite3
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import entity_gate as G
g = G.parse(open(sys.argv[2], encoding="utf-8").read())
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True); con.row_factory = sqlite3.Row
# 会话窗口：同一 source_path 按 id 排序 ±8
ev = {}
by_path = collections.defaultdict(list)
for r in con.execute("select id,source_path p,source_actor a,content c from evidence order by id"):
    ev[r["id"]] = r; by_path[r["p"]].append(r["id"])
pos = {i: (p, k) for p, ids in by_path.items() for k, i in enumerate(ids)}
def ctx(eid, K=8):
    p, k = pos[eid]; ids = by_path[p]
    return "\n".join(ev[i]["c"] for i in ids[max(0, k-K):k+K+1])
out = []
for f in con.execute("select id,subject,status,user_pinned,fact_type,valid_from,content from facts where status in ('candidate','confirmed','uncertain')"):
    if not (G.mentions(g, f["content"]) or G._group_of_subject(g, f["subject"])):
        continue
    eids = [r[0] for r in con.execute("select evidence_id from fact_evidence where fact_id=?", (f["id"],))]
    eids = [e for e in eids if e in ev]
    if not eids:
        continue
    srcs = [ev[e]["c"] for e in eids]
    c = "\n".join(ctx(e) for e in eids)
    v = G.judge(f["content"], srcs, c, f["subject"], g, actors={ev[e]["a"] for e in eids})
    if v.action != "ok":
        out.append({"id": f["id"], "subject": f["subject"], "status": f["status"], "pinned": f["user_pinned"],
                    "action": v.action, "reason": v.reason, "content": f["content"], "new": v.content,
                    "src": [s[:220] for s in srcs][:2], "actors": sorted({ev[e]["a"] for e in eids})})
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print(collections.Counter((o["status"], o["action"]) for o in out))
