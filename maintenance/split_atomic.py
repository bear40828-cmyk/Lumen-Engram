#!/usr/bin/env python3
"""Lumen Engram · 旧事实拆原子（任务四·步骤5，2026-10-29，开关 ENGRAM_ATOMIC_SPLIT，默认关）

问题：早期规则管线和每日摘要灌库留下的事实，常常一条里塞了好几件事（「今天改完了路由；她喜欢粉色；以后不许用 Markdown」）。
粒度太粗，后面的取代、冲突、归纳、精确召回都不准。新的蒸馏已经按「一条一件事」抽（quote 逐字核对），
这个工具只管**已经在库里的旧事实**。

做法（保持简单）：
  1. 找「看着像复合句」的活事实：candidate（不碰 confirmed / 钉住 / 归纳 / 已处理过的），长度 ≥ 50 且分句 ≥ 2。
  2. 让模型拆成原子事实，每条带 quote（必须是原事实正文里的逐字子串）。
  3. 代码后检：≥2 条才算拆（只有 1 条＝本来就是原子，记 atomic）；每条 quote 逐字在原文里；
     原子里的数字必须出现在原文；原子的二字词 ≥60% 来自原文（防凭空加信息）；无密钥；4–120 字；最多 6 条。
  4. 通过的写成**新的派生事实**（同 subject、同证据、同 valid_from），derived_from 连回原事实；**原事实一行不改**。
  5. atomic_split_log 记每条原事实的结果，处理过的不再处理。

**不要对正式库跑。** 必须显式给 --db；默认只看（不调模型、不写库）；--apply 才调模型并写库。
    .venv/bin/python maintenance/split_atomic.py --db 副本.db                   # 干看：多少条像复合句
    .venv/bin/python maintenance/split_atomic.py --db 副本.db --apply --limit 20 --show
按名单拆（2026-10-29，人看过的提案原样写，不调模型，整批可撤回）：
    ... --db X --proposals 提案.json --ids 1,2,3 [--batch 批号]          # 干看
    ... --db X --proposals 提案.json --ids 1,2,3 --apply                 # 写库（原事实一行不改，新事实记进 atomic_split_batch）
    ... --db X --rollback 批号 --apply                                   # 整批撤回（新事实标 invalid，不删）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

MIN_LEN = 50
MIN_CLAUSES = 2
MAX_ATOMS = 6
ATOM_MIN, ATOM_MAX = 4, 120
OVERLAP_MIN = 0.6
LABEL = "〔归纳〕"
TYPES = {"episodic", "semantic", "preference", "procedural"}
SECRET = re.compile(r"1[3-9]\d{9}|密码|password|passwd|token|密钥|api[_-]?key", re.I)
CLAUSE_SPLIT = re.compile(r"[；;。！？!?\n]+")


def clauses(text: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT.split(text or "") if len(c.strip()) >= 4]


def looks_compound(content: str) -> bool:
    return len(content or "") >= MIN_LEN and len(clauses(content)) >= MIN_CLAUSES


# —— 候选筛选（2026-10-29 真模型实测后加）——
# 实测：3033 条「像复合句」的候选里 2863 条是别人的整行聊天原文，只有 11 条是她自己的真事实。
# 拆别人的话、拆整段聊天记录、拆别的 AI 的工作汇报，都不是这个工具要做的事。
# 借鉴：MemoryScope 的 info_filter（先判这段话「有没有值得记的、关于这个人的信息」，没有就不进下一步）、
# mem0 的「只从用户消息抽、别把助理的话当用户的事实」。这里用便宜的规则做这一层，不额外花模型。
RE_TURN = re.compile(r"(?:^|\n|\s)[^\s：:（(]{1,12}\s*[（(]\s*\d{4}-\d{2}-\d{2}[^）)]*[）)]\s*[：:]")
WORK_WORDS = ("已完成", "已提交", "提交了", "commit", "测试通过", "全部通过", "已修复", "修复了", "部署", "回归", "重构",
              "pytest", "脚本", "报错", "分支", "合并", "接口", "字段", "迁移", "写入库", "跑通",
              "重启", "备份", "thread", "config", "源码", "测试", "编译", "服务")
MAX_RAW_LEN = 600


def is_transcript(content: str) -> bool:
    """整段聊天记录：出现 ≥2 个「某某（时间）：」说话人前缀，或长得不像一条事实。"""
    return len(RE_TURN.findall(content or "")) >= 2 or len(content or "") > MAX_RAW_LEN


def is_work_report(content: str) -> bool:
    """工作汇报（别的 AI 做了什么、改了哪个文件）：命中 ≥3 个工程词，或带文件名/路径。"""
    c = content or ""
    hits = sum(1 for w in WORK_WORDS if w.lower() in c.lower())
    return hits >= 3 or bool(re.search(r"\b\w+\.(py|sh|sql|yaml|yml|md|json|ts|tsx|js|toml|rs|go|cfg|ini)\b|/srv/|git ", c))


def spoken_by_other(content: str) -> bool:
    """开头是「某某（时间）：」且说话的不是她本人：拆的是别人说的话，不是她的事。"""
    m = re.match(r"^\s*([^\s：:（(]{1,12})\s*(?:[（(][^）)]*[）)])?\s*[：:]", content or "")
    if not m:
        return False
    who = m.group(1)
    mine = {persona.N_USER, persona.USER_FULL, *persona.USER_ALIASES}
    return who not in mine


def eligible(r) -> bool:
    c = r["content"]
    return (looks_compound(c) and not is_transcript(c) and not is_work_report(c) and not spoken_by_other(c))


def _bg(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def check_atoms(original: str, atoms) -> tuple[list[dict] | None, str]:
    """模型给的拆分过不过代码后检。返回 (清洗后的原子列表 或 None, 结果/原因)。结果 'atomic' 表示本来就是原子。"""
    if not isinstance(atoms, list) or len(atoms) < 2:
        return None, "atomic"
    if len(atoms) > MAX_ATOMS:
        return None, f"too-many-atoms:{len(atoms)}"
    nums = set(re.findall(r"\d+", original))
    ogram = _bg(original)
    out, seen = [], set()
    for a in atoms:
        if not isinstance(a, dict):
            return None, "bad-shape"
        content = str(a.get("content") or "").strip()
        quote = str(a.get("quote") or "").strip()
        if not (ATOM_MIN <= len(content) <= ATOM_MAX):
            return None, "atom-length"
        if SECRET.search(content):
            return None, "secret"
        if len(quote) < 4 or quote not in original:
            return None, "quote-not-verbatim"
        if not set(re.findall(r"\d+", content)) <= nums:
            return None, "invented-number"
        g = _bg(content)
        if not g or len(g & ogram) / len(g) < OVERLAP_MIN:
            return None, "not-from-original"
        if content in seen or content == original:
            return None, "duplicate-atom"
        seen.add(content)
        typ = str(a.get("type") or "").strip()
        out.append({"content": content, "quote": quote, "type": typ if typ in TYPES else None})
    return out, "split"


def build_prompt(content: str) -> str:
    # 借鉴：mem0 的「一条记忆一个事实、保留具体细节、没有可记的就返回空」+ few-shot；
    # Graphiti 的「每条事实脱离上下文也看得懂（把代词换成名字）、别概括掉数字和专名」；
    # A-Mem 的「原子笔记」。few-shot 全是虚构例子。
    return ("下面是记忆库里的一条旧记录，可能一条里塞了好几件互不相干的事。\n"
            "任务：只有它确实包含 2 件以上**互相独立**的事时，才拆成原子事实（一件事一条）；否则输出空列表。\n"
            "规则：\n"
            "1. 每条原子 content 要自成一句、脱离原文也看得懂：把「她/他/这个/那边」换成原文里出现的名字或具体对象；"
            "但名字、数字、日期、专名必须原样保留，不许概括成「很多」「某个」。\n"
            "2. 只用原记录里有的信息，不许推测、不许补因果、不许加原文没有的数字和日期。\n"
            "3. 同一件事的前因后果、同一个动作的几个细节，不要拆开（拆开就看不懂了）。\n"
            "4. 不要为了凑数拆。原子里不能出现「这件事」「上述」「如上」这种要靠上下文才懂的指代。\n"
            "5. quote：原记录里的一段原文，必须逐字复制（是原记录的子串）。type：episodic / semantic / preference / procedural 之一。\n"
            "6. 最多 6 条。\n\n"
            "例1（该拆）原记录：小林周三要去牙科复诊；小林更喜欢用纸质笔记本；下个月打算把自行车卖掉\n"
            "输出：{\"atoms\":[{\"content\":\"小林周三要去牙科复诊\",\"quote\":\"小林周三要去牙科复诊\",\"type\":\"episodic\"},"
            "{\"content\":\"小林更喜欢用纸质笔记本\",\"quote\":\"小林更喜欢用纸质笔记本\",\"type\":\"preference\"},"
            "{\"content\":\"小林下个月打算把自行车卖掉\",\"quote\":\"下个月打算把自行车卖掉\",\"type\":\"episodic\"}]}\n"
            "例2（不拆：一件事的两个细节）原记录：小林周三下午三点去牙科复诊，医生说要补一颗牙\n"
            "输出：{\"atoms\":[]}\n"
            "例3（不拆：没有可记的事实）原记录：好的，我知道了，晚点再说吧，先这样\n"
            "输出：{\"atoms\":[]}\n\n"
            "只输出 JSON：{\"atoms\":[{\"content\":\"…\",\"quote\":\"…\",\"type\":\"…\"}]}\n\n"
            f"原记录：{content}")


def ask(prompt: str) -> str:
    """调模型（测试里 mock 这一个函数）。"""
    from llm import chat
    return chat([{"role": "system", "content": "你只输出合法 JSON。"}, {"role": "user", "content": prompt}],
                purpose="extract", max_tokens=600, timeout=60)


def parse(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        return d.get("atoms") if isinstance(d, dict) else None
    except (ValueError, TypeError):
        return None


def candidates(con, limit: int = 0, subject: str | None = None) -> list:
    """只拆**她自己**的事实（subject 默认 persona.USER），不拆别人的话、聊天整段、工作汇报。"""
    subject = subject or persona.USER
    rows = con.execute(
        """SELECT f.id, f.ext_id, f.subject, f.fact_type, f.content, f.valid_from, f.importance, f.confidence
           FROM facts f
           WHERE f.status = 'candidate' AND f.user_pinned = 0 AND f.valid_to IS NULL
             AND f.subject = ?
             AND f.content NOT LIKE ?
             AND f.id NOT IN (SELECT fact_id FROM atomic_split_log)
             AND f.id NOT IN (SELECT child_fact_id FROM fact_relations WHERE relation='derived_from')
             AND f.id NOT IN (SELECT parent_fact_id FROM fact_relations WHERE relation='derived_from')
           ORDER BY f.id""", (subject, LABEL + "%")).fetchall()
    rows = [r for r in rows if eligible(r)]
    return rows[:limit] if limit else rows


def _log(con, fid, outcome, n=0, reason=None):
    from store.store import now
    con.execute("INSERT OR IGNORE INTO atomic_split_log (fact_id, outcome, n_atoms, reason, created_at)"
                " VALUES (?,?,?,?,?)", (fid, outcome, n, reason, now()))


def run(con, *, apply: bool = False, limit: int = 20, show: bool = False) -> dict:
    from store.store import promote, relate
    todo = candidates(con)
    st = {"compound": len(todo), "calls": 0, "split": 0, "atomic": 0, "rejected": 0, "atoms_written": 0}
    if not apply:
        return st
    for r in todo[:limit] if limit else todo:
        st["calls"] += 1
        try:
            atoms = parse(ask(build_prompt(r["content"])))
        except Exception:                                   # noqa: BLE001  模型失败＝跳过，明天再来
            continue
        good, why = check_atoms(r["content"], atoms)
        if good is None:
            outcome = "atomic" if why == "atomic" else "rejected"
            _log(con, r["id"], outcome, 0, None if outcome == "atomic" else why)
            con.commit()
            st[outcome] += 1
            continue
        ev = [x[0] for x in con.execute(
            "SELECT evidence_id FROM fact_evidence WHERE fact_id=? AND role='support'", (r["id"],))]
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        try:
            n = 0
            for a in good:
                fid = promote(con, subject=r["subject"], fact_type=a["type"] or r["fact_type"], content=a["content"],
                              evidence_ids=ev, valid_from=r["valid_from"], importance=r["importance"],
                              confidence=r["confidence"])
                if fid is None:
                    continue                                # 同 subject 同正文已存在：不重复
                relate(con, fid, r["id"], "derived_from")
                n += 1
                if show:
                    print(f"  → {a['content']}")
            _log(con, r["id"], "split", n)
            con.commit()
            st["split"] += 1
            st["atoms_written"] += n
        except BaseException:
            con.rollback()
            raise
    return st


# ───────── 按名单拆（2026-10-29）：看过的提案原样写入，不调模型；整批可撤回 ─────────
def load_proposals(path) -> list[dict]:
    """提案文件：[{review_no, fact_id, ext_id, orig, atoms:[{content, quote, type}]}]（拆分结果是人看过的那一份）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("提案文件应是列表")
    return data


def apply_proposals(con, props: list[dict], ids: list[int], *, apply: bool = False, batch_id: str | None = None) -> dict:
    """只写 ids（review_no）点名的提案，不调模型。每条落库前重验：
    原事实还活着、是 candidate、没钉住/确认、属于她、正文一字未变、没处理过；
    每个原子再过一遍 check_atoms（引文逐字、数字不凭空、二字词出自原文、无密钥、长度）。
    任何一条不过就整条跳过（记 skipped 和原因，不写任何东西）。原事实一行不改；新事实 derived_from 连回、记进 batch 表。"""
    from store.store import now, promote, relate
    st: dict = {"batch_id": batch_id, "selected": 0, "written_parents": 0, "new_facts": 0, "exists": 0,
                "skipped": {}, "new_fact_ext_ids": []}
    want = set(ids)
    chosen = [p for p in props if p.get("review_no") in want]
    st["selected"] = len(chosen)
    for no in sorted(want - {p.get("review_no") for p in chosen}):
        st["skipped"][no] = "提案文件里没有这个编号"
    for p in chosen:
        no = p["review_no"]
        row = con.execute("SELECT * FROM facts WHERE id=? AND ext_id=?", (p.get("fact_id"), p.get("ext_id"))).fetchone()
        why = None
        if row is None:
            why = "找不到原事实"
        elif row["content"] != p.get("orig"):
            why = "原事实正文已变"
        elif row["status"] != "candidate" or row["user_pinned"] or row["valid_to"]:
            why = "原事实不是活着的 candidate（确认/钉住/作废的不碰）"
        elif row["subject"] != persona.USER:
            why = "原事实不是她的"
        elif con.execute("SELECT 1 FROM atomic_split_log WHERE fact_id=?", (row["id"],)).fetchone() or con.execute(
                "SELECT 1 FROM fact_relations WHERE relation='derived_from' AND (parent_fact_id=? OR child_fact_id=?)",
                (row["id"], row["id"])).fetchone():
            why = "这条已经拆过/本身是派生事实"
        else:
            good, res = check_atoms(row["content"], p.get("atoms"))
            if good is None:
                why = "后检不过：" + res
        if why:
            st["skipped"][no] = why
            continue
        if not apply:
            st["written_parents"] += 1
            st["new_facts"] += len(good)
            continue
        ev = [x[0] for x in con.execute(
            "SELECT evidence_id FROM fact_evidence WHERE fact_id=? AND role='support'", (row["id"],))]
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        try:
            made = []
            for a in good:
                fid = promote(con, subject=row["subject"], fact_type=a["type"] or row["fact_type"], content=a["content"],
                              evidence_ids=ev, valid_from=row["valid_from"], importance=row["importance"],
                              confidence=row["confidence"])
                if fid is None:
                    st["exists"] += 1                   # 同正文已有活着的一条，或防回声没放行：不重复、不强写
                    continue
                relate(con, fid, row["id"], "derived_from")
                con.execute("INSERT INTO atomic_split_batch (batch_id, child_fact_id, parent_fact_id, created_at)"
                            " VALUES (?,?,?,?)", (batch_id, fid, row["id"], now()))
                made.append(fid)
            if not made:
                con.rollback()
                st["skipped"][no] = "所有原子都已存在或被防回声拦下，什么都没写"
                continue
            con.execute("INSERT OR IGNORE INTO atomic_split_log (fact_id, outcome, n_atoms, reason, created_at)"
                        " VALUES (?,?,?,?,?)", (row["id"], "split", len(made), f"batch:{batch_id}", now()))
            con.commit()
            st["written_parents"] += 1
            st["new_facts"] += len(made)
            st["new_fact_ext_ids"] += [con.execute("SELECT ext_id FROM facts WHERE id=?", (m,)).fetchone()[0] for m in made]
        except BaseException:
            con.rollback()
            raise
    return st


def rollback_batch(con, batch_id: str, *, apply: bool = False) -> dict:
    """整批撤回：把这批新写的事实标 invalid（不删）。只动还是 candidate、没钉住、没被她确认的；其余跳过并报告。原事实不碰。"""
    from store.store import _bump, now
    kids = con.execute(
        "SELECT f.id, f.ext_id, f.status, f.user_pinned, f.valid_to FROM atomic_split_batch b JOIN facts f ON f.id=b.child_fact_id"
        " WHERE b.batch_id=?", (batch_id,)).fetchall()
    st = {"batch_id": batch_id, "in_batch": len(kids), "invalidated": 0, "skipped": []}
    for k in kids:
        if k["valid_to"] or k["status"] == "invalid":
            continue
        if k["status"] != "candidate" or k["user_pinned"]:
            st["skipped"].append(k["ext_id"])
            continue
        if apply:
            _bump(con, k["id"], None, status="invalid", valid_to=now())
        st["invalidated"] += 1
    if apply:
        con.commit()
    return st


def main(argv=None) -> int:
    import flags
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="必须显式给；不要指向正式库")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--proposals", help="提案 JSON（人看过的拆分结果）；配 --ids 就只写点名的，不调模型")
    ap.add_argument("--ids", help="提案里的 review_no，逗号分隔，如 1,2,3")
    ap.add_argument("--batch", help="批号（缺省自动生成）")
    ap.add_argument("--rollback", metavar="BATCH", help="整批撤回这个批号写的新事实（要 --apply 才真改）")
    args = ap.parse_args(argv)
    if not flags.on("ATOMIC_SPLIT"):
        print("拆原子已关闭（ENGRAM_ATOMIC_SPLIT=1 才运行；见 flags.py），未做任何事。")
        return 0
    from store.store import connect
    con = connect(args.db)
    if args.rollback:
        st = rollback_batch(con, args.rollback, apply=args.apply)
        print(("撤回：" if args.apply else "干看撤回：") + json.dumps(st, ensure_ascii=False))
        return 0
    if args.proposals:
        if not args.ids:
            ap.error("--proposals 要配 --ids")
        import time
        batch = args.batch or time.strftime("split-%Y%m%d-%H%M%S")
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        st = apply_proposals(con, load_proposals(args.proposals), ids, apply=args.apply, batch_id=batch)
        print(("按名单写库：" if args.apply else "按名单干看（没写库）：") + json.dumps(st, ensure_ascii=False))
        return 0
    st = run(con, apply=args.apply, limit=args.limit, show=args.show)
    print(("写库：" if args.apply else "干看（没调模型、没写库）：") + json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
