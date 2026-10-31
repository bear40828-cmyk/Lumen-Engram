"""密钥打码接在哪儿：入库（证据/事实）、送模型（llm/embedding）、出口（钩子/MCP/遥测）。密钥都是随机假值。"""
import importlib.util
import io
import json
import random
import string
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
import redact  # noqa: E402
from common import sha256  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

_r = random.Random(11)
B62 = string.ascii_letters + string.digits


def key():
    while True:
        s = "".join(_r.choice(B62) for _ in range(35))
        if any(c.isdigit() for c in s) and any(c.isupper() for c in s) and any(c.islower() for c in s):
            return "sk-" + s


SECRET = key()
TEXT = f"这是我的 key：{SECRET}，你先记着"


class Ingest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.d.name) / "t.db")
        self.old_log, redact.LOG_FILE = redact.LOG_FILE, Path(self.d.name) / "redact.log"

    def tearDown(self):
        redact.LOG_FILE = self.old_log
        self.con.close()
        self.d.cleanup()

    def test_evidence_redacted_and_hash_on_redacted(self):
        ev = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L1",
                          source_path="/x.jsonl", content=TEXT)
        row = self.con.execute("select content, content_sha256 from evidence where id=?", (ev,)).fetchone()
        self.assertNotIn(SECRET, row["content"])
        self.assertIn("〔已打码:sk:", row["content"])
        self.assertEqual(row["content_sha256"], sha256(row["content"]))
        log = redact.LOG_FILE.read_text(encoding="utf-8")
        self.assertIn("evidence sk=1", log)
        self.assertNotIn(SECRET[3:14], log)

    def test_idempotent_rescan_same_id(self):
        a = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L1", content=TEXT)
        b = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L1", content=TEXT)
        self.assertEqual(a, b)

    def test_old_raw_row_is_found_not_duplicated(self):
        """老证据是原样存的（不可改）。重扫同一行不能再塞一份打码版。"""
        self.con.execute(
            "INSERT INTO evidence (ext_id, captured_at, source_kind, source_actor, source_locator, content, content_sha256)"
            " VALUES ('ev_old', '2026-01-01', 'chat', 'tester', 'L9', ?, ?)", (TEXT, sha256(TEXT)))
        old = self.con.execute("select id from evidence where ext_id='ev_old'").fetchone()[0]
        got = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L9", content=TEXT)
        self.assertEqual(got, old)
        self.assertEqual(self.con.execute("select count(*) from evidence").fetchone()[0], 1)

    def test_fact_content_redacted(self):
        ev = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L2", content="普通的话 " * 5)
        fid = promote(self.con, subject="tester", fact_type="episodic", content=f"她给了我 {SECRET}",
                      evidence_ids=[ev], importance=0.5, confidence=0.7)
        c = self.con.execute("select content from facts where id=?", (fid,)).fetchone()[0]
        self.assertNotIn(SECRET, c)
        self.assertIn("〔已打码:sk:", c)

    def test_clean_text_unchanged(self):
        t = "今天喝了奶茶 " * 5
        ev = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="L3", content=t)
        self.assertEqual(self.con.execute("select content from evidence where id=?", (ev,)).fetchone()[0], t)


if __name__ == "__main__":
    unittest.main()
