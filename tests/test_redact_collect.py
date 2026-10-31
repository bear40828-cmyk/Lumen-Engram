"""collect：带密钥的行存打码版；重扫要认得「已有」，不能重复入库。"""
import json
import random
import string
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import redact  # noqa: E402
from extract import collect  # noqa: E402
from store.store import connect  # noqa: E402


class CollectRescan(unittest.TestCase):
    def test_secret_line_stored_redacted_and_rescan_is_noop(self):
        r = random.Random(3)
        secret = "sk-" + "".join(r.choice(string.ascii_letters + string.digits) for _ in range(35)) + "Aa1"
        with tempfile.TemporaryDirectory() as d:
            old, redact.LOG_FILE = redact.LOG_FILE, Path(d) / "redact.log"
            try:
                con = connect(Path(d) / "t.db")
                f = Path(d) / "s.jsonl"
                text = f"这是新的 key {secret} 你先记着，别告诉别人哦"
                f.write_text(json.dumps({"type": "user", "message": {"role": "user", "content": text},
                                         "timestamp": "2026-10-28T01:00:00Z"}, ensure_ascii=False) + "\n", encoding="utf-8")
                t1 = collect.collect_file(con, f, full=True)
                con.commit()
                t2 = collect.collect_file(con, f, full=True)
                con.commit()
                rows = con.execute("select content, content_sha256 from evidence").fetchall()
                log = (Path(d) / "redact.log").read_text(encoding="utf-8")
            finally:
                redact.LOG_FILE = old
        self.assertEqual((t1["new"], t2["new"], t2["have"]), (1, 0, 1), (t1, t2))
        self.assertEqual(len(rows), 1)
        self.assertNotIn(secret, rows[0]["content"])
        self.assertIn("〔已打码:sk:", rows[0]["content"])
        self.assertNotIn(secret[3:14], log)


if __name__ == "__main__":
    unittest.main()
