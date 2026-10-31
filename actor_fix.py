"""说话人标注修正层（overlay）——历史 evidence 里标错的说话人，不改原行，在 evidence_actor_fix 里改判。

为什么有这层：evidence 表只增不改，但老入库路径（每日摘要 from_digest：说话人取摘要条目里的名字，正文取原 jsonl 行）留下了两类错标：
  1. leading-block：整条正文就是 <channel user="某_bot" user_id=…> 转发包装，库里却标成 user（她的话）。
     实际是 bot（GPT）发的 → 按 speakers 名单改判成 gpt。认不出的 bot → effective_actor=NULL（不当任何人的话）。
  2. embedded-block：标成 claude 的证据里，后半夹着 `user<channel … user="她的 id">她的话</channel>`（反方向：她的话被算在助手名下）。
     前半仍是助手；块里的话归 block_actor——但只有同一会话里另有一条该说话人的证据含同样的话才算数（corroborated）；
     对不上的不当任何人的话（助手侧的 channel 块常是模型自己续写的假「下一轮」，见 from_channel 的硬规矩一；
     实查：夹着她名字的 30 行里只有 1 行能在库里找到她本人的同句原话）。
读取方（echo.evidence_actors / echo.provenance / consolidate / distill_slice 渲染 / 回放）用 lookup()/segments()；没有记录用原标注。
本模块只读 evidence；写只写 evidence_actor_fix（build）。没有任何 UPDATE/DELETE evidence。
"""
from __future__ import annotations

import json
import re
import sqlite3

import speakers

_CH = re.compile(r'<channel\s+(?P<attrs>[^>]*)>(?P<body>.*?)(?:</channel>|\Z)', re.S)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_TAIL_JUNK = re.compile(r'\s*</?(?:parameter|channel)>\s*$')
_NORM = re.compile(r'[\s\W_]+')


def _norm(s: str) -> str:
    return _NORM.sub("", s or "")


def _body(raw: str) -> str:
    b = (raw or "").strip()
    while True:
        n = _TAIL_JUNK.sub("", b)
        if n == b:
            break
        b = n
    return b.strip()


def _blocks(content: str):
    out = []
    for m in _CH.finditer(content or ""):
        a = dict(_ATTR.findall(m.group("attrs")))
        out.append({"start": m.start(), "end": m.end(), "user": a.get("user", ""), "user_id": a.get("user_id", ""),
                    "actor": speakers.channel_actor(a.get("user", ""), a.get("user_id", "")),
                    "body": _body(m.group("body"))})
    return out


def analyze(actor: str | None, content: str, corroborate=None) -> dict | None:
    """一条证据（原说话人 + 正文）要不要改判。不用改返回 None。
    corroborate(block_actor, body) -> bool：embedded-block 才用，查库里有没有佐证。"""
    if not content or "<channel" not in content:
        return None
    bl = _blocks(content)
    if not bl:
        return None
    first = bl[0]
    if content.lstrip().startswith("<channel") and first["start"] == len(content) - len(content.lstrip()):
        if first["actor"] == actor:
            return None
        who = first["actor"]
        return {"kind": "leading-block", "original_actor": actor, "effective_actor": who, "block_actor": who,
                "block_user": first["user"], "corroborated": None,
                "reason": (f"正文是 <channel user={first['user'] or first['user_id']}> 转发包装，按名单是 {who}，库里标成 {actor}"
                           if who else f"正文是 <channel user={first['user'] or first['user_id']}> 转发包装，user 不在名单里，库里标成 {actor}：不当任何人的话")}
    diff = [b for b in bl if b["actor"] != actor]
    if not diff:
        return None
    b = diff[0]
    cor = None
    if b["actor"] and corroborate is not None and len(_norm(b["body"])) >= 4:
        cor = 1 if corroborate(b["actor"], b["body"]) else 0
    elif b["actor"] is None:
        cor = 0
    return {"kind": "embedded-block", "original_actor": actor, "effective_actor": None, "block_actor": b["actor"],
            "block_user": b["user"] or b["user_id"], "corroborated": cor,
            "reason": (f"{actor} 的证据里夹着 <channel user={b['user'] or b['user_id']}> 块（按名单是 {b['actor']}）；"
                       + ("同会话另有原话佐证，块里的话归" + str(b["actor"]) if cor else "库里查不到佐证——助手侧的 channel 块常是模型自己续写出来的假「下一轮」（已有多起记录），不当任何人的话"))}


def corroborator(con, source_path: str, self_id: int | None = None):
    def _f(block_actor: str, body: str) -> bool:
        nb = _norm(body)
        if len(nb) < 4:
            return False
        for r in con.execute("SELECT id, content FROM evidence WHERE source_path=? AND source_actor=?",
                             (source_path, block_actor)):
            if r[0] != self_id and nb in _norm(r[1]):
                return True
        return False
    return _f


def compute_all(con) -> dict[int, dict]:
    """扫一遍库（只读）算出全部需要改判的证据。{evidence_id: fix}"""
    out = {}
    rows = con.execute("SELECT id, source_actor, content, source_path FROM evidence WHERE content LIKE '%<channel %'").fetchall()
    for r in rows:
        fx = analyze(r[1], r[2], corroborator(con, r[3], r[0]))
        if fx:
            out[r[0]] = fx
    return out


def has_table(con) -> bool:
    try:
        return bool(con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='evidence_actor_fix'").fetchone())
    except sqlite3.Error:
        return False


def build(con, *, now: str | None = None) -> dict:
    """把 compute_all 的结果写进 evidence_actor_fix（只插入新行；已有的不动）。返回计数。"""
    from store.store import now as _now
    ts = now or _now()
    fixes = compute_all(con)
    n_new = 0
    for eid, fx in fixes.items():
        cur = con.execute(
            "INSERT OR IGNORE INTO evidence_actor_fix (evidence_id, kind, original_actor, effective_actor, block_actor,"
            " block_user, corroborated, reason, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (eid, fx["kind"], fx["original_actor"], fx["effective_actor"], fx["block_actor"], fx["block_user"],
             fx["corroborated"], fx["reason"], ts))
        n_new += cur.rowcount
    con.commit()
    return {"candidates": len(fixes), "inserted": n_new}


def lookup(con, ids) -> dict[int, dict]:
    """库里已建好的改判（只查表；表不存在/出错＝没有）。召回热路径也用这个，所以不扫库。"""
    ids = [i for i in dict.fromkeys(ids or ()) if i is not None]
    if not ids or not has_table(con):
        return {}
    out = {}
    try:
        for k in range(0, len(ids), 500):
            part = ids[k:k + 500]
            q = ",".join("?" * len(part))
            for r in con.execute(f"SELECT evidence_id, kind, original_actor, effective_actor, block_actor, block_user,"
                                 f" corroborated, reason FROM evidence_actor_fix WHERE evidence_id IN ({q})", part):
                out[r[0]] = {"kind": r[1], "original_actor": r[2], "effective_actor": r[3], "block_actor": r[4],
                             "block_user": r[5], "corroborated": r[6], "reason": r[7]}
    except sqlite3.Error:
        return {}
    return out


def fixes_for_rows(con, rows) -> dict[int, dict]:
    """蒸馏渲染用：表里有就用表；表里没有（还没 build）就对含 <channel 的那几行现场算。rows 要有 id/source_actor/content/source_path。"""
    todo = [r for r in rows if "<channel" in (r["content"] or "")]
    if not todo:
        return {}
    got = lookup(con, [r["id"] for r in todo])
    for r in todo:
        if r["id"] in got:
            continue
        fx = analyze(r["source_actor"], r["content"], corroborator(con, r["source_path"], r["id"]))
        if fx:
            got[r["id"]] = fx
    return got


def actors_of(fix: dict | None, original: str | None) -> set:
    """这条证据实际算谁的话（防回声用）。"""
    if not fix:
        return {original} if original else set()
    if fix["kind"] == "leading-block":
        return {fix["effective_actor"]} if fix["effective_actor"] else set()
    s = {original} if original else set()
    if fix.get("corroborated") and fix.get("block_actor"):
        s.add(fix["block_actor"])
    return s


def segments(fix: dict | None, original: str | None, content: str) -> list[tuple[str | None, str]]:
    """按改判把正文拆成 [(说话人, 文本)]。leading-block：只剩包装里的话；embedded-block：前半原说话人 + 佐证过的块；
    没佐证的块整段丢掉（不当任何人的话）。没有改判：[(原说话人, 原文)]。"""
    if not fix:
        return [(original, content)]
    bl = _blocks(content)
    if fix["kind"] == "leading-block":
        if not bl:
            return [(fix["effective_actor"], content)]
        return [(fix["effective_actor"], bl[0]["body"])] if fix["effective_actor"] else []
    segs = []
    pos = 0
    for b in bl:
        pre = content[pos:b["start"]]
        pre = re.sub(r"\s*user\s*$", "", pre).strip()
        if pre:
            segs.append((original, pre))
        if b["actor"] != original:
            if b["actor"] and fix.get("corroborated") and b["body"]:
                segs.append((b["actor"], b["body"]))
        elif b["body"]:
            segs.append((original, b["body"]))
        pos = b["end"]
    tail = content[pos:].strip()
    if tail:
        segs.append((original, tail))
    return segs or [(original, content)]


def as_json(fix: dict) -> str:
    return json.dumps(fix, ensure_ascii=False)
