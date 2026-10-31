"""任务二·步骤2：metamem 的 FOK/TOT 摘出召回路径（默认关），软话豁免仍在 router"""
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flags  # noqa: E402
import retrieve  # noqa: E402
from router import route  # noqa: E402


class FlagDefaults(unittest.TestCase):
    def test_all_defaults_off(self):
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True):
            for k in flags.DEFAULTS:
                self.assertFalse(flags.on(k), k)

    def test_env_turns_on(self):
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_METAMEM": "1"}, clear=True):
            self.assertTrue(flags.on("METAMEM"))

    def test_unknown_flag(self):
        with self.assertRaises(KeyError):
            flags.on("NOPE")


class SoftTalkStaysInRouter(unittest.TestCase):
    def test_soft_talk_not_recalled(self):
        for t in ("想你了", "亲爱的👀"):
            d = route(t)
            self.assertFalse(d.should_recall, t)
            self.assertTrue(d.reason.startswith("软话"), d.reason)

    def test_explicit_recall_overrides_soft_talk(self):
        self.assertTrue(route("你还记得我说过喜欢你吗").should_recall)


class NoMetamemInRecallPath(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(":memory:")
        self.con.row_factory = sqlite3.Row

    def test_metamem_not_imported_by_default(self):
        sys.modules.pop("metamem", None)
        import importlib
        importlib.reload(retrieve)
        # 步骤3 之后默认是混合召回；这条测的是旧分层路径里 metamem 默认不再参与，所以打开回滚开关
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_LAYERED_RETRIEVE": "1"}, clear=True):
            with mock.patch.object(retrieve, "search", return_value=[
                    {"ext_id": "fact_a", "subject": "x", "content": "星光湖 很好看", "score": 1.0}]), \
                    mock.patch.object(retrieve.working, "relevance", return_value=(0.0, "")), \
                    mock.patch.object(retrieve, "rarity", return_value=5):
                r = retrieve.retrieve(self.con, "上次那个星光湖是哪年的事")
        self.assertTrue(r.should_recall)
        self.assertNotIn("metamem", sys.modules)
        # 不判够不够 → 每层都读，且没有 enough
        self.assertTrue(all(not s.enough for s in r.steps))
        self.assertEqual(r.escalations, 0)

    def test_soft_talk_returns_no_hits_without_search(self):
        with mock.patch.object(retrieve, "search", side_effect=AssertionError("不该查")), \
                mock.patch.object(retrieve.working, "relevance", return_value=(0.0, "")):
            r = retrieve.retrieve(self.con, "想你了")
        self.assertFalse(r.should_recall)


if __name__ == "__main__":
    unittest.main()
