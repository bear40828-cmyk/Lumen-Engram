"""任务七：把默认关的功能在副本上跑真（metamem 混合召回接线 + 误报修正、timeline 接进 distill_slice 且不碰 confirmed/钉住）"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metamem  # noqa: E402
from store.store import connect, promote, confirm, record_class, record_key  # noqa: E402

OFF = {"ENGRAM_ENV_FILE": "/nonexistent"}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def fact(self, content, when="2026-09-01", subject="tester"):
        fid = promote(self.con, subject=subject, fact_type="episodic", content=content,
                      valid_from=when, importance=0.5, confidence=0.7)
        if self._has_class():
            record_class(self.con, fid, "episodic", 0.9, "extract-llm-v1")
        return fid

    def _has_class(self):
        return bool(self.con.execute("SELECT 1 FROM sqlite_master WHERE name='fact_class'").fetchone())

    def ext(self, fid):
        return self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]


class MetamemKeys(Base):
    def keyed(self, fid, key, value, pol="pos"):
        record_key(self.con, fid, key, value, pol, True)

    def hits(self, *fids):
        return [{"ext_id": self.ext(f)} for f in fids]

    def test_single_key_two_values_is_conflict(self):
        a, b = self.fact("用户老家在甲市"), self.fact("用户老家在乙县")
        self.keyed(a, "person.hometown", "甲市"); self.keyed(b, "person.hometown", "乙县")
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="single")):
            ev = metamem.check_evidence(self.con, self.hits(a, b), lexical=False)
        self.assertTrue(any("person.hometown 有两个值" in c for c in ev.conflicts), ev.conflicts)

    def test_multi_key_two_values_is_not_conflict(self):
        a, b = self.fact("用户考301"), self.fact("用户考302")
        self.keyed(a, "education.exam_subject", "301"); self.keyed(b, "education.exam_subject", "302", "neg")
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="multi")):
            ev = metamem.check_evidence(self.con, self.hits(a, b), lexical=False)
        self.assertEqual(ev.conflicts, ())

    def test_polarity_conflict_needs_same_value(self):
        a, b = self.fact("爱吃苹果"), self.fact("用户说爱吃苹果其实不爱")
        self.keyed(a, "preference.food", "苹果"); self.keyed(b, "preference.food", "苹果", "neg")
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="multi")):
            ev = metamem.check_evidence(self.con, self.hits(a, b), lexical=False)
        self.assertTrue(any("一条肯定一条否定" in c for c in ev.conflicts))

    def test_lexical_rule_can_be_turned_off(self):
        a, b = self.fact("记忆库还没做完"), self.fact("记忆库已经做完了")
        on = metamem.check_evidence(self.con, self.hits(a, b), terms=["记忆库"])
        off = metamem.check_evidence(self.con, self.hits(a, b), terms=["记忆库"], lexical=False)
        self.assertTrue(on.conflicts)
        self.assertEqual(off.conflicts, ())


class TimelineWiring(Base):
    def setUp(self):
        super().setUp()
        if not self._has_class():
            self.skipTest("schema 里没有 fact_class")
        self.old = self.fact("用户说chess游戏每天都在做", "2026-09-01")
        from store.store import add_evidence
        ev = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="conv:t#msg:77",
                          content="我不做chess游戏了", occurred_at="2026-09-23")
        self.new_ok = {"subject": "tester", "content": "用户说chess游戏已经不做了",
                       "occurred_at": "2026-09-23", "evidence_id": ev}

    def run_target(self, env, idx):
        import extract.distill_slice as DS
        with mock.patch.dict(os.environ, {**OFF, **env}, clear=True), \
             mock.patch("extract.timeline.judge", return_value=(idx, "t")):
            return DS._timeline_target(self.con, dict(self.new_ok))

    def test_off_by_default(self):
        self.assertIsNone(self.run_target({}, 0))
        self.assertIsNone(self.run_target({"ENGRAM_TIMELINE": "1"}, 0))          # 只开一个也不判
        self.assertIsNone(self.run_target({"ENGRAM_INVALIDATE_CHAIN": "1"}, 0))

    def test_both_on_returns_old(self):
        t = self.run_target({"ENGRAM_TIMELINE": "1", "ENGRAM_INVALIDATE_CHAIN": "1"}, 0)
        self.assertIsNotNone(t)
        self.assertEqual(t[0]["id"], self.old)      # (旧事, 槽描述)

    def test_confirmed_and_pinned_never_candidates(self):
        from extract import timeline
        confirm(self.con, self.ext(self.old))
        env = {**OFF, "ENGRAM_TIMELINE": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(timeline.candidates(self.con, "tester", "用户说chess游戏已经不做了", None), [])

    def test_dup_new_fact_does_not_seal_old(self):
        self.fact("用户说chess游戏已经不做了", "2026-09-22")
        t = self.run_target({"ENGRAM_TIMELINE": "1", "ENGRAM_INVALIDATE_CHAIN": "1"}, 0)
        self.assertIsNone(t)

    def test_supersede_keeps_old_text(self):
        from store.store import supersede
        before = self.con.execute("SELECT content FROM facts WHERE id=?", (self.old,)).fetchone()[0]
        with mock.patch.dict(os.environ, {**OFF, "ENGRAM_INVALIDATE_CHAIN": "1"}, clear=True):
            nid = supersede(self.con, self.ext(self.old), content=self.new_ok["content"],
                            subject="tester", fact_type="episodic", valid_from="2026-09-23", hard=True)
        row = self.con.execute("SELECT status, valid_to, content FROM facts WHERE id=?", (self.old,)).fetchone()
        self.assertEqual((row[0], row[1], row[2]), ("superseded", "2026-09-23", before))
        self.assertEqual(self.con.execute("SELECT supersedes_id FROM facts WHERE id=?", (nid,)).fetchone()[0], self.old)


if __name__ == "__main__":
    unittest.main()
