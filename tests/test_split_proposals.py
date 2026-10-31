"""2026-10-29：按名单拆原子（split_atomic --proposals/--ids）与整批撤回。虚构夹具；不调模型。"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))
import persona  # noqa: E402
import split_atomic as SA  # noqa: E402
from store.store import add_evidence, confirm, connect, pin, promote  # noqa: E402

U = persona.USER
ORIG = "小林喜欢粉色；小林周三要去牙科复诊；小林下个月打算把自行车卖掉"
ATOMS = [{"content": "小林喜欢粉色", "quote": "小林喜欢粉色", "type": "preference"},
         {"content": "小林周三要去牙科复诊", "quote": "小林周三要去牙科复诊", "type": "episodic"},
         {"content": "小林下个月打算把自行车卖掉", "quote": "下个月打算把自行车卖掉", "type": "episodic"}]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.n = 0

    def fact(self, content, status="candidate"):
        self.n += 1
        eid = add_evidence(self.con, source_kind="chat", source_actor=U, source_locator=f"conv:t#msg:{self.n}",
                           content=content, occurred_at="2026-09-01")
        fid = promote(self.con, subject=U, fact_type="episodic", content=content, evidence_ids=[eid], status=status,
                      valid_from="2026-09-01", importance=0.5, confidence=0.7)
        self.con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,'support')", (fid, eid))
        self.con.commit()
        return fid

    def prop(self, fid, no, orig=ORIG, atoms=ATOMS):
        r = self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()
        return {"review_no": no, "fact_id": fid, "ext_id": r[0], "orig": orig, "atoms": atoms}

    def snap(self, fid):
        return tuple(self.con.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone())

    def apply(self, props, ids, **kw):
        with mock.patch.object(SA, "ask", side_effect=AssertionError("按名单写入不许调模型")):
            return SA.apply_proposals(self.con, props, ids, batch_id="b1", **kw)


class ApplyProposals(Base):
    def test_dry_run_writes_nothing(self):
        f = self.fact(ORIG)
        n0 = self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        st = self.apply([self.prop(f, 1)], [1], apply=False)
        self.assertEqual((st["written_parents"], st["new_facts"]), (1, 3))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], n0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_batch").fetchone()[0], 0)

    def test_apply_writes_exact_text_links_batch_and_leaves_original(self):
        f = self.fact(ORIG)
        before = self.snap(f)
        st = self.apply([self.prop(f, 1)], [1], apply=True)
        self.assertEqual((st["written_parents"], st["new_facts"], st["skipped"]), (1, 3, {}))
        self.assertEqual(self.snap(f), before)                                   # 原事实一字未动（含 version/updated_at）
        kids = self.con.execute(
            "SELECT f.content, f.subject, f.status, f.valid_from, b.batch_id FROM atomic_split_batch b"
            " JOIN facts f ON f.id=b.child_fact_id ORDER BY f.id").fetchall()
        self.assertEqual([k[0] for k in kids], [a["content"] for a in ATOMS])
        self.assertTrue(all(k[1] == U and k[2] == "candidate" and k[3] == "2026-09-01" and k[4] == "b1" for k in kids))
        # 沿用 relate() 的既有约定：parent=新原子，child=原事实
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM fact_relations WHERE relation='derived_from' AND child_fact_id=?",
                                          (f,)).fetchone()[0], 3)
        self.assertEqual(self.con.execute("SELECT outcome, n_atoms, reason FROM atomic_split_log").fetchone()[:], ("split", 3, "batch:b1"))

    def test_new_facts_inherit_speaker_evidence_and_subject(self):
        import echo
        f = self.fact(ORIG)
        self.apply([self.prop(f, 1)], [1], apply=True)
        for (cid,) in self.con.execute("SELECT child_fact_id FROM atomic_split_batch").fetchall():
            ev = [r[0] for r in self.con.execute("SELECT evidence_id FROM fact_evidence WHERE fact_id=? AND role='support'", (cid,))]
            self.assertEqual(echo.evidence_actors(self.con, ev), {U})
            self.assertEqual(self.con.execute("SELECT subject FROM facts WHERE id=?", (cid,)).fetchone()[0], U)

    def test_only_selected_ids(self):
        f1, f2 = self.fact(ORIG), self.fact(ORIG.replace("小林", "小周"))
        o2 = ORIG.replace("小林", "小周")
        a2 = [dict(a, content=a["content"].replace("小林", "小周"), quote=a["quote"].replace("小林", "小周")) for a in ATOMS]
        st = self.apply([self.prop(f1, 1), self.prop(f2, 2, o2, a2)], [2], apply=True)
        self.assertEqual((st["selected"], st["written_parents"]), (1, 1))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_log WHERE fact_id=?", (f1,)).fetchone()[0], 0)

    def test_unknown_review_no_is_reported(self):
        f = self.fact(ORIG)
        st = self.apply([self.prop(f, 1)], [1, 9], apply=True)
        self.assertIn(9, st["skipped"])

    def test_confirmed_pinned_changed_and_done_are_skipped(self):
        fc, fp, fch, fd = (self.fact(ORIG + s) for s in ("一", "二", "三", "四"))
        confirm(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fc,)).fetchone()[0])
        pin(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fp,)).fetchone()[0])
        self.con.execute("INSERT INTO atomic_split_log (fact_id, outcome, n_atoms, created_at) VALUES (?, 'split', 2, 'x')", (fd,))
        self.con.commit()
        props = [self.prop(fc, 1, ORIG + "一"), self.prop(fp, 2, ORIG + "二"),
                 self.prop(fch, 3, ORIG + "改过的"), self.prop(fd, 4, ORIG + "四")]
        st = self.apply(props, [1, 2, 3, 4], apply=True)
        self.assertEqual(st["written_parents"], 0)
        self.assertEqual(sorted(st["skipped"]), [1, 2, 3, 4])
        self.assertIn("确认/钉住", st["skipped"][1])
        self.assertIn("正文已变", st["skipped"][3])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_batch").fetchone()[0], 0)

    def test_bad_atoms_fail_recheck(self):
        f = self.fact(ORIG)
        bad_num = ATOMS[:2] + [dict(ATOMS[2], content="小林下个月打算把自行车卖掉，价格 300 元")]
        bad_quote = ATOMS[:2] + [dict(ATOMS[2], quote="根本不在原文里的引文")]
        for atoms, key in ((bad_num, "invented-number"), (bad_quote, "quote-not-verbatim")):
            st = self.apply([self.prop(f, 1, atoms=atoms)], [1], apply=True)
            self.assertEqual(st["written_parents"], 0)
            self.assertIn(key, st["skipped"][1])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)

    def test_not_her_fact_is_skipped(self):
        f = self.fact(ORIG)
        with mock.patch.object(persona, "USER", "someone-else"):
            st = self.apply([self.prop(f, 1)], [1], apply=True)
        self.assertEqual(st["written_parents"], 0)

    def test_nothing_written_when_promote_refuses(self):
        f = self.fact(ORIG)
        with mock.patch("store.store.promote", return_value=None):
            st = self.apply([self.prop(f, 1)], [1], apply=True)
        self.assertEqual(st["written_parents"], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_log").fetchone()[0], 0)

    def test_rerun_does_not_duplicate(self):
        f = self.fact(ORIG)
        self.apply([self.prop(f, 1)], [1], apply=True)
        st = self.apply([self.prop(f, 1)], [1], apply=True)
        self.assertEqual(st["written_parents"], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_batch").fetchone()[0], 3)


class Rollback(Base):
    def test_rollback_invalidates_only_the_batch_children(self):
        f = self.fact(ORIG)
        orig_before = self.snap(f)
        other = self.fact("小周今天去了图书馆还借了三本书")
        other_before = self.snap(other)
        self.apply([self.prop(f, 1)], [1], apply=True)
        kids = [r[0] for r in self.con.execute("SELECT child_fact_id FROM atomic_split_batch ORDER BY child_fact_id")]
        confirm(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (kids[0],)).fetchone()[0])
        self.con.commit()
        dry = SA.rollback_batch(self.con, "b1", apply=False)
        self.assertEqual((dry["in_batch"], dry["invalidated"], len(dry["skipped"])), (3, 2, 1))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE status='invalid'").fetchone()[0], 0)
        st = SA.rollback_batch(self.con, "b1", apply=True)
        self.assertEqual(st["invalidated"], 2)
        status = {r[0]: r[1] for r in self.con.execute("SELECT id, status FROM facts WHERE id IN (%s)" % ",".join("?" * 3), kids)}
        self.assertEqual(sorted(status.values()), ["confirmed", "invalid", "invalid"])
        self.assertEqual(self.snap(f), orig_before)
        self.assertEqual(self.snap(other), other_before)
        self.assertEqual(SA.rollback_batch(self.con, "b1", apply=True)["invalidated"], 0)     # 幂等

    def test_unknown_batch_is_empty(self):
        self.assertEqual(SA.rollback_batch(self.con, "nope", apply=True)["in_batch"], 0)


class Cli(Base):
    def run_main(self, argv, env):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent", **env}, clear=True), redirect_stdout(buf):
            rc = SA.main(argv)
        return rc, buf.getvalue()

    def test_flag_off_does_nothing_and_flag_on_runs(self):
        f = self.fact(ORIG)
        pf = Path(self.dir.name) / "p.json"
        pf.write_text(json.dumps([self.prop(f, 1)]), encoding="utf-8")
        db = str(Path(self.dir.name) / "t.db")
        rc, out = self.run_main(["--db", db, "--proposals", str(pf), "--ids", "1", "--apply"], {})
        self.assertIn("已关闭", out)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_batch").fetchone()[0], 0)
        rc, out = self.run_main(["--db", db, "--proposals", str(pf), "--ids", "1", "--batch", "bx", "--apply"], {"ENGRAM_ATOMIC_SPLIT": "1"})
        self.assertIn('"new_facts": 3', out)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_batch WHERE batch_id='bx'").fetchone()[0], 3)
        rc, out = self.run_main(["--db", db, "--rollback", "bx", "--apply"], {"ENGRAM_ATOMIC_SPLIT": "1"})
        self.assertIn('"invalidated": 3', out)


if __name__ == "__main__":
    unittest.main()
