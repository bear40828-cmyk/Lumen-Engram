"""任务四·步骤2（2026-10-29）：同槽「可能过时」软标记（ENGRAM_INVALIDATE_CHAIN）——只标不封不删。虚构夹具。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import persona  # noqa: E402
import stale  # noqa: E402
from store.store import add_evidence, confirm, connect, pin, promote, record_key  # noqa: E402

U, C = persona.USER, persona.CLAUDE
ON = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_INVALIDATE_CHAIN": "1"}
OFF = {"ENGRAM_ENV_FILE": "/nonexistent"}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.n = 0

    def fact(self, content, when, actor=U, subject=U, status="candidate"):
        self.n += 1
        eid = add_evidence(self.con, source_kind="chat", source_actor=actor,
                           source_locator=f"conv:t#msg:{self.n}", content=content, occurred_at=when)
        fid = promote(self.con, subject=subject, fact_type="semantic", content=content,
                      evidence_ids=[eid], status=status, valid_from=when, importance=0.5, confidence=0.7)
        self.assertIsNotNone(fid)
        return self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]

    def row(self, ext):
        return self.con.execute("SELECT status, valid_to, content, version FROM facts WHERE ext_id=?", (ext,)).fetchone()[:]

    def marks(self):
        return self.con.execute("SELECT COUNT(*) FROM fact_stale_marks").fetchone()[0]


class Gate(Base):
    def setUp(self):
        super().setUp()
        self.old = self.fact("我用的是蓝色手机", "2026-09-01")
        self.new = self.fact("我换了绿色手机", "2026-09-20")

    def mark(self, old=None, new=None, slot="她用什么手机", env=ON):
        with mock.patch.dict(os.environ, env, clear=True):
            return stale.mark(self.con, old or self.old, new or self.new, slot=slot)

    def test_marks_without_touching_facts(self):
        before_old, before_new = self.row(self.old), self.row(self.new)
        self.assertIsNotNone(self.mark())
        self.assertEqual(self.marks(), 1)
        self.assertEqual(self.row(self.old), before_old)        # 状态、有效期、版本号一个都没动
        self.assertEqual(self.row(self.new), before_new)
        self.assertIsNone(self.row(self.old)[1])

    def test_off_by_default_writes_nothing(self):
        self.assertIsNone(self.mark(env=OFF))
        self.assertEqual(self.marks(), 0)

    def test_newer_must_be_users_own_words(self):
        # 证据只有助理在说（助理转述）。入库防回声本来就会拦；confirmed 能绕过入库那道，这里测的是标记这一道也拦
        n2 = self.fact("她换了绿色手机", "2026-09-21", actor=C, status="confirmed")
        ok, why = stale.check(self.con, self.old, n2, "手机")
        self.assertFalse(ok)
        self.assertIn("原话", why)
        self.assertIsNone(self.mark(new=n2))
        self.assertEqual(self.marks(), 0)

    def test_claude_subject_never_marked(self):
        a = self.fact("助理说他用蓝色主题", "2026-09-01", actor=C, subject=C)
        b = self.fact("助理说他改用绿色主题", "2026-09-20", actor=C, subject=C)
        self.assertIsNone(self.mark(a, b, "主题"))

    def test_different_subject_rejected(self):
        b = self.fact("助理换了绿色手机", "2026-09-20", actor=C, subject=C)
        self.assertIsNone(self.mark(new=b))

    def test_confirmed_and_pinned_old_never_marked(self):
        confirm(self.con, self.old)
        self.assertIsNone(self.mark())
        o2 = self.fact("我住在甲市", "2026-08-01")
        n2 = self.fact("我搬到乙县了", "2026-09-25")
        pin(self.con, o2)
        self.assertIsNone(self.mark(o2, n2, "住哪"))
        self.assertEqual(self.marks(), 0)

    def test_newer_must_be_later(self):
        older = self.fact("我换了红色手机", "2026-08-01")
        self.assertIsNone(self.mark(self.old, older))

    def test_needs_slot_or_key(self):
        self.assertIsNone(self.mark(slot=""))
        self.assertIsNone(self.mark(slot=None))

    def test_fact_key_same_slot_different_value_ok_without_slot_text(self):
        for ext, v in ((self.old, "蓝"), (self.new, "绿")):
            record_key(self.con, self.con.execute("SELECT id FROM facts WHERE ext_id=?", (ext,)).fetchone()[0],
                       "device.phone_color", v, "pos", True)
        self.assertIsNotNone(self.mark(slot=""))

    def test_fact_key_different_keys_rejected(self):
        for ext, k in ((self.old, "device.phone_color"), (self.new, "device.phone_brand")):
            record_key(self.con, self.con.execute("SELECT id FROM facts WHERE ext_id=?", (ext,)).fetchone()[0],
                       k, "x", "pos", True)
        self.assertIsNone(self.mark())

    def test_idempotent(self):
        self.assertIsNotNone(self.mark())
        self.assertIsNone(self.mark())
        self.assertEqual(self.marks(), 1)

    def test_marks_cannot_be_deleted_or_edited(self):
        self.mark()
        with self.assertRaises(Exception):
            self.con.execute("DELETE FROM fact_stale_marks")
        with self.assertRaises(Exception):
            self.con.execute("UPDATE fact_stale_marks SET slot='x'")


class Reading(Base):
    def setUp(self):
        super().setUp()
        self.old = self.fact("我用的是蓝色手机", "2026-09-01")
        self.new = self.fact("我换了绿色手机", "2026-09-20")
        with mock.patch.dict(os.environ, ON, clear=True):
            stale.mark(self.con, self.old, self.new, slot="用什么手机")
            self.con.commit()

    def test_marks_for_only_when_flag_on(self):
        with mock.patch.dict(os.environ, ON, clear=True):
            self.assertEqual(stale.marks_for(self.con, [self.old])[self.old]["ext_id"], self.new)
        with mock.patch.dict(os.environ, OFF, clear=True):
            self.assertEqual(stale.marks_for(self.con, [self.old]), {})

    def test_latest_new_wins(self):
        n3 = self.fact("我换了黑色手机", "2026-10-02")
        with mock.patch.dict(os.environ, ON, clear=True):
            stale.mark(self.con, self.old, n3, slot="用什么手机")
            self.assertEqual(stale.marks_for(self.con, [self.old])[self.old]["ext_id"], n3)

    def test_later_confirm_voids_mark(self):
        confirm(self.con, self.old)
        with mock.patch.dict(os.environ, ON, clear=True):
            self.assertEqual(stale.marks_for(self.con, [self.old]), {})

    def test_hybrid_ranks_marked_lower_but_keeps_it(self):
        import hybrid
        merged = {e: {"ext_id": e, "rrf": 0.03, "via": ["fts"], "status": "candidate", "layer": "active"}
                  for e in (self.old, self.new)}
        with mock.patch.dict(os.environ, ON, clear=True):
            m = stale.marks_for(self.con, [self.old, self.new])
        out = hybrid.finalize(merged, stale=m)
        self.assertEqual([h["ext_id"] for h in out], [self.new, self.old])
        old_hit = out[1]
        self.assertEqual(old_hit["stale_by"], self.new)
        self.assertAlmostEqual(old_hit["hybrid_score"], round(0.03 * stale.STALE_WEIGHT, 5))

    def test_full_retrieve_marks_hit(self):
        import retrieve
        with mock.patch.dict(os.environ, ON, clear=True):
            r = retrieve.retrieve(self.con, "我用什么手机")
        by = {h["ext_id"]: h for h in r.hits}
        self.assertIn(self.old, by)
        self.assertEqual(by[self.old]["stale_by"], self.new)
        self.assertIsNone(by[self.new]["stale_by"])


class Wiring(Base):
    def test_distill_slice_marks_instead_of_superseding(self):
        import extract.distill_slice as DS
        old = self.fact("我用的是蓝色手机", "2026-09-01")
        eid = add_evidence(self.con, source_kind="chat", source_actor=U, source_locator="conv:t#msg:901",
                           content="我换了绿色手机", occurred_at="2026-09-20")
        ok = {"subject": U, "content": "我换了绿色手机", "occurred_at": "2026-09-20", "evidence_id": eid}
        with mock.patch.dict(os.environ, {**ON, "ENGRAM_TIMELINE": "1"}, clear=True), \
             mock.patch("extract.timeline.candidates", return_value=[{"ext_id": old, "content": "x"}]), \
             mock.patch("extract.timeline.judge", return_value=(0, "用什么手机｜换了")):
            tgt = DS._timeline_target(self.con, dict(ok))
        self.assertEqual(tgt, ({"ext_id": old, "content": "x"}, "用什么手机"))

    def test_no_pipeline_code_calls_hard_supersede(self):
        root = Path(__file__).resolve().parent.parent
        for f in ("extract/distill_slice.py", "extract/from_evidence.py"):
            txt = (root / f).read_text(encoding="utf-8")
            self.assertNotIn("supersede(", txt, f)


class WriteChunkWiring(Base):
    """distill_slice._write_chunk 端到端：判新旧在写事务之外，写进去的只有「可能过时」标记。"""

    def test_marks_old_and_never_seals_it_and_judges_outside_txn(self):
        import datetime as dt
        import extract.distill_slice as DS
        from store.store import record_class
        old = self.fact("我用的是蓝色手机", "2026-09-01")
        record_class(self.con, self.con.execute("SELECT id FROM facts WHERE ext_id=?", (old,)).fetchone()[0],
                     "semantic", 0.8, "extract-llm-v1")
        eid = add_evidence(self.con, source_kind="chat", source_actor=U, source_locator="conv:t#msg:900",
                           content="我换了绿色手机，之前那个蓝色的不用了", occurred_at="2026-09-20")
        chunk = [{"id": eid, "content": "我换了绿色手机，之前那个蓝色的不用了", "source_actor": U,
                  "occurred_at": "2026-09-20"}]
        facts = [{"subject": U, "type": "semantic", "content": "她换了绿色手机", "quote": "我换了绿色手机",
                  "src": 0, "confidence": 0.8}]
        seen = {}

        def fake_judge(new, when, olds):
            seen["in_txn"] = self.con.in_transaction          # 判的时候不许在写事务里
            return 0, "她用什么手机｜换了"
        before_old = self.row(old)
        old_row = dict(self.con.execute("SELECT * FROM facts WHERE ext_id=?", (old,)).fetchone())
        with mock.patch.dict(os.environ, {**ON, "ENGRAM_TIMELINE": "1"}, clear=True), \
             mock.patch("extract.timeline.candidates", return_value=[old_row]), \
             mock.patch("extract.timeline.judge", fake_judge):
            res = DS._write_chunk(self.con, "sess1", chunk, facts, "run_t1", "manual", {}, {},
                                  dt.datetime.now(dt.timezone.utc), 1, advance=False)
        self.assertEqual(res["kept"], 1)
        self.assertFalse(seen["in_txn"])
        self.assertEqual(self.marks(), 1)
        self.assertEqual(self.row(old), before_old)              # 旧的一个字没动、没封
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE status='superseded'").fetchone()[0], 0)

    def test_non_user_new_fact_never_reaches_the_model(self):
        import extract.distill_slice as DS
        ok = {"subject": C, "content": "助理换了绿色主题", "occurred_at": "2026-09-20", "evidence_id": 1}
        with mock.patch.dict(os.environ, {**ON, "ENGRAM_TIMELINE": "1"}, clear=True), \
             mock.patch("extract.timeline.judge", side_effect=AssertionError("must not call")):
            self.assertIsNone(DS._timeline_target(self.con, ok))


class BackfillAndQuoting(Base):
    def test_candidates_survive_fts_special_characters(self):
        from extract import timeline
        from store.store import record_class
        old = self.fact("我用 AND-OR 这个记账 app", "2026-09-01")
        record_class(self.con, self.con.execute("SELECT id FROM facts WHERE ext_id=?", (old,)).fetchone()[0],
                     "semantic", 0.8, "extract-llm-v1")
        with mock.patch.dict(os.environ, {**OFF, "ENGRAM_TIMELINE": "1"}, clear=True), \
             mock.patch.object(timeline, "keywords", return_value=["AND-OR", "记账"]):
            got = timeline.candidates(self.con, U, "我不用 AND-OR 了", "2026-09-20")
        self.assertEqual([g["ext_id"] for g in got], [old])

    def test_backfill_only_marks(self):
        import extract.backfill_timeline as BF
        from store.store import record_class
        old = self.fact("我用的是蓝色手机", "2026-09-01")
        new = self.fact("我换了绿色手机", "2026-09-20")
        for e in (old, new):
            record_class(self.con, self.con.execute("SELECT id FROM facts WHERE ext_id=?", (e,)).fetchone()[0],
                         "semantic", 0.8, "extract-llm-v1")
        before = (self.row(old), self.row(new))
        with mock.patch.dict(os.environ, {**ON, "ENGRAM_TIMELINE": "1"}, clear=True), \
             mock.patch("extract.timeline.candidates", side_effect=lambda con, s, c, b: (
                 [dict(r) for r in con.execute("SELECT * FROM facts WHERE ext_id=?", (old,))] if "绿色" in c else [])), \
             mock.patch("extract.timeline.judge", return_value=(0, "用什么手机｜换了")):
            st = BF.run(self.con, dry_run=False)
        self.assertEqual(st["marked"], 1)
        self.assertEqual((self.row(old), self.row(new)), before)
        self.assertEqual(self.marks(), 1)


if __name__ == "__main__":
    unittest.main()
