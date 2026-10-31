"""收尾 A5：distill_self 的 JSON 容错解析 + 失败重试一次 / 死信（模型全 mock）"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import distill_self as D  # noqa: E402

GOOD = '{"facts":[{"subject":"claude","type":"episodic","content":"甲","quote":"甲甲"},{"subject":"claude","type":"episodic","content":"乙","quote":"乙乙"}]}'


class Tolerant(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(len(D.parse_json_tolerant(GOOD)["facts"]), 2)

    def test_fence_and_prose(self):
        self.assertEqual(len(D.parse_json_tolerant("好的：\n```json\n" + GOOD + "\n```\n以上")["facts"]), 2)

    def test_trailing_comma(self):
        t = '{"facts":[{"subject":"claude","content":"甲",},]}'
        self.assertEqual(len(D.parse_json_tolerant(t)["facts"]), 1)

    def test_truncated_keeps_complete_items(self):
        t = GOOD[:GOOD.index('{"subject":"claude","type":"episodic","content":"乙"') + 30]
        got = D.parse_json_tolerant(t)
        self.assertEqual([f["content"] for f in got["facts"]], ["甲"])

    def test_empty_facts_ok(self):
        self.assertEqual(D.parse_json_tolerant('{"facts":[]}'), {"facts": []})

    def test_garbage_raises(self):
        for bad in ("", "没有事实", "[1,2]", "{"):
            with self.assertRaises(ValueError):
                D.parse_json_tolerant(bad)


if __name__ == "__main__":
    unittest.main()
