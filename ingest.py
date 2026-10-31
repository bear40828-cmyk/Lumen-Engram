#!/usr/bin/env python3
"""Lumen Engram · 一条命令把摘要灌进库：抽取 → 核对 → 门控 → 晋升

    python3 ingest.py                      # 把 memory/digests 下所有摘要过一遍
    python3 ingest.py 2026-08-16           # 只灌某一天
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import persona  # noqa: E402

from common import read_line, sha256                    # noqa: E402
from extract.from_digest import CN, SPEAKER_ACTOR, load_lines, locate, parse_digest  # noqa: E402
from classify import classify                           # noqa: E402
from gate.gate import judge                             # noqa: E402
from keys import match as match_key                     # noqa: E402
import unified                                           # noqa: E402
from store.store import (add_evidence, connect, promote, record_class, record_key,  # noqa: E402
                         record_tier, stats)

DIGESTS = Path(persona.get("ENGRAM_DIGEST_DIR", str(Path.home() / ".engram/digests")))


def ingest_one(con, digest: Path) -> dict:
    jsonl_path, bullets = parse_digest(digest)
    if not jsonl_path or not Path(jsonl_path).exists():
        return {"skipped_file": True}

    day = digest.stem
    cache: dict[tuple[int, int], list] = {}
    tally = {"promoted": 0, "gate_no": 0, "no_source": 0, "dup": 0}

    for b in bullets:
        key = (b.line_lo, b.line_hi)
        cache.setdefault(key, load_lines(jsonl_path, *key))
        hit = locate(b, cache[key])
        if not hit:
            tally["no_source"] += 1
            continue

        line_no, real_text = hit
        occurred = (dt.datetime.strptime(f"{day} {b.hhmm}", "%Y-%m-%d %H:%M")
                    .replace(tzinfo=CN).astimezone(dt.timezone.utc).isoformat())

        # 核对：拿 locator 回原文重算，对上才给 verified
        got = read_line(jsonl_path, line_no)
        status = "verified" if got and sha256(got[1]) == sha256(real_text) else "uncertain"

        ev_id = add_evidence(con, source_kind="chat", source_actor=SPEAKER_ACTOR[b.speaker],
                             source_path=jsonl_path, source_locator=f"L{line_no}",
                             content=real_text, occurred_at=occurred,
                             verification_status=status)

        # 证据分级：这条 evidence 是本人原话，origin_root 用「文件+行号」聚簇——
        # 同一句原话不管被摘要抽一次还是直抽再抽一次，都是同一票。
        record_tier(con, ev_id, "primary", f"{jsonl_path}:L{line_no}")

        # 防回声审计（task2-08）：subject = 证据说话人本人（SPEAKER_ACTOR），证据与事实同一个人，不可能记到别人名下；
        # unified.decide 对 LLM 越权 subject 退回规则，store.promote 再按库里证据兜底。
        subject = SPEAKER_ACTOR[b.speaker]
        content = f"{b.speaker}（{day} {b.hhmm}）：{b.text}"
        verdict = unified.decide(con, subject=subject, content=content,
                                 evidence_rows=[{"verification_status": status}])
        if not verdict.promote:
            tally["gate_no"] += 1
            continue

        fact_id = promote(con, subject=verdict.subject, fact_type=verdict.fact_type, content=content,
                          evidence_ids=[ev_id], valid_from=occurred,
                          importance=verdict.importance, confidence=verdict.confidence)
        if fact_id is not None:
            record_class(con, fact_id, verdict.fact_type, verdict.type_conf, verdict.type_why)
            if verdict.fact_key:
                record_key(con, fact_id, verdict.fact_key, verdict.value, verdict.polarity,
                           verdict.eligible)
        tally["dup" if fact_id is None else "promoted"] += 1

    con.commit()
    return tally


def main() -> int:
    import flags
    if flags.on("RAW_ONLY"):
        # 任务三：每日摘要不再产出事实（摘要只当参考；原话已由 extract/collect.py 进 evidence，由 distill_slice 整理）
        print("已冻结（RAW_ONLY）：每日摘要不再灌库")
        return 0
    con = connect()
    want = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(DIGESTS.glob("*.md"))
    if want:
        files = [f for f in files if f.stem == want]
    if not files:
        print("没有匹配的摘要文件", file=sys.stderr)
        return 1

    total = {"promoted": 0, "gate_no": 0, "no_source": 0, "dup": 0}
    for f in files:
        t = ingest_one(con, f)
        if t.get("skipped_file"):
            print(f"{f.stem}  出处 jsonl 不在了，跳过")
            continue
        for k in total:
            total[k] += t.get(k, 0)
        print(f"{f.stem}  晋升 {t['promoted']:>3}   门控拒 {t['gate_no']:>3}   "
              f"无原话 {t['no_source']:>3}   重复 {t['dup']:>3}")

    print(f"\n合计：晋升 {total['promoted']}，门控拒 {total['gate_no']}，"
          f"无原话 {total['no_source']}，重复 {total['dup']}")
    s = stats(con)
    print(f"库：证据 {s['evidence']}（已核对 {s['evidence_verified']}）"
          f"｜事实 {s['facts_by_status']}｜当下有效 {s['live']}｜钉住 {s['pinned']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
