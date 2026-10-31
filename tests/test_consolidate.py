"""巩固的回归检查：只读库，验证候选组满足自己声明的门槛。

    python3 tests/test_consolidate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import consolidate as C          # noqa: E402
from keys import by_name         # noqa: E402
from store.store import connect  # noqa: E402


def main() -> int:
    con = connect()
    cands = C.candidates(con)
    bad = []
    for c in cands:
        spec = by_name(c["key"])
        if spec is not None and not spec.consolidatable:
            bad.append(("不该参与巩固的槽位", c["key"]))
        if spec is not None and spec.owner and c["subject"] != spec.owner:
            bad.append(("主语没按 owner 算", c["key"], c["subject"], spec.owner))
        if c["votes"] < C.MIN_SUPPORT:
            bad.append(("票数不够", c["key"], c["votes"]))
        if c["owner_votes"] < C.MIN_OWNER_VOTES:
            bad.append(("没有当事人本人的原话", c["key"]))
        if len(c["days"]) < C.MIN_DAYS:
            bad.append(("天数不够", c["key"], c["days"]))
        roots = [r["origin_root"] for r in c["rows"]]
        if len(roots) != len(set(roots)):
            bad.append(("同源没去重", c["key"]))

    for b in bad:
        print("FAIL", b)
    print(f"候选 {len(cands)} 组，问题 {len(bad)} 处")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
