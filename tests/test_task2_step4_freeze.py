"""任务二·步骤4：冻结 consolidate / supersede / invalidate / timeline 的入口（默认关，开关打开原样可用）"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flags  # noqa: E402
from store.store import connect, invalidate, promote, supersede  # noqa: E402

OFF = {"ENGRAM_ENV_FILE": "/nonexistent"}


class Frozen(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.fid = promote(self.con, subject="tester", fact_type="semantic",
                           content="一条测试用的事实", valid_from="2026-09-01",
                           importance=0.5, confidence=0.7)
        self.ext = self.con.execute("SELECT ext_id FROM facts WHERE id=?", (self.fid,)).fetchone()[0]

    def status(self):
        return self.con.execute("SELECT status, valid_to FROM facts WHERE id=?", (self.fid,)).fetchone()[:]

    def test_supersede_frozen_does_not_touch_db(self):
        before = self.status()
        n = self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        with mock.patch.dict(os.environ, OFF, clear=True):
            with self.assertRaises(flags.FrozenError):
                supersede(self.con, self.ext, content="新内容")
        self.assertEqual(self.status(), before)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], n)

    def test_invalidate_frozen_does_not_touch_db(self):
        before = self.status()
        with mock.patch.dict(os.environ, OFF, clear=True):
            with self.assertRaises(flags.FrozenError):
                invalidate(self.con, self.ext, reason="x")
        self.assertEqual(self.status(), before)

    def test_supersede_and_invalidate_work_when_enabled(self):
        with mock.patch.dict(os.environ, {**OFF, "ENGRAM_INVALIDATE_CHAIN": "1"}, clear=True):
            with self.assertRaises(flags.FrozenError):      # 2026-10-29：不带 hard=True 一律不让封
                supersede(self.con, self.ext, content="新内容", subject="tester", fact_type="semantic")
            self.assertEqual(self.status()[0], "candidate")
            new = supersede(self.con, self.ext, content="新内容", subject="tester", fact_type="semantic", hard=True)
            self.assertEqual(self.status()[0], "superseded")
            self.assertIsNotNone(new)
            n2 = self.con.execute("SELECT ext_id FROM facts WHERE id=?", (new,)).fetchone()[0]
            with self.assertRaises(flags.FrozenError):
                invalidate(self.con, n2, reason="测试")
            invalidate(self.con, n2, reason="测试", hard=True)
            self.assertEqual(self.con.execute("SELECT status FROM facts WHERE id=?", (new,)).fetchone()[0], "invalid")

    def test_consolidate_main_frozen_never_connects(self):
        import consolidate
        buf = io.StringIO()
        with mock.patch.dict(os.environ, OFF, clear=True), \
                mock.patch.object(consolidate, "connect", side_effect=AssertionError("冻结时不该连库")), \
                mock.patch.object(sys, "argv", ["consolidate.py", "--apply"]), redirect_stdout(buf):
            self.assertEqual(consolidate.main(), 0)
        self.assertIn("已冻结", buf.getvalue())

    def test_consolidate_run_still_callable(self):
        import consolidate
        self.assertEqual(consolidate.run(self.con, apply=False), 0)

    def test_timeline_frozen(self):
        from extract import timeline
        with mock.patch.dict(os.environ, OFF, clear=True):
            self.assertEqual(timeline.candidates(self.con, "tester", "一条测试用的事实", None), [])
            self.assertEqual(timeline.judge("n", "2026-01-01", [{"content": "o", "valid_from": ""}]),
                             (None, "时间线判断已冻结"))

    def test_timeline_enabled_finds_candidates_path(self):
        from extract import timeline
        with mock.patch.dict(os.environ, {**OFF, "ENGRAM_TIMELINE": "1"}, clear=True):
            self.assertEqual(timeline.judge("n", "2026-01-01", []), (None, "没有沾边的旧事"))
            self.assertEqual(timeline.judge("n", "2026-01-01", [{"content": "o", "valid_from": ""}]),
                             (None, "没配模型"))

    def test_newer_unchanged_and_importable(self):
        import newer
        self.assertTrue(callable(newer.superseded_by))


if __name__ == "__main__":
    unittest.main()
