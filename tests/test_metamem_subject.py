"""任务四·步骤1（2026-10-29）：metamem 主体归属——问「我的生日」不能拿助理的生日答。全部虚构夹具。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metamem  # noqa: E402
import persona  # noqa: E402
from store.store import add_evidence, connect, promote, record_key  # noqa: E402

U, C = persona.USER, persona.CLAUDE


def hit(subject, content, ext="x", **kw):
    return {"ext_id": ext, "subject": subject, "content": content, "score": 0.9, **kw}


class AskedSubject(unittest.TestCase):
    def test_i_and_she_mean_user(self):
        for q in ("我哪天生日", "她哪天生日", "你知道我哪天生日吗", "告诉我我哪年去的", "我生日你记得吗",
                  f"{persona.N_USER}哪天生日"):
            self.assertEqual(metamem.asked_subject(q), U, q)

    def test_you_means_claude(self):
        for q in ("你哪天生日", "告诉我你哪年上线的", f"{persona.N_CLAUDE}哪天生日"):
            self.assertEqual(metamem.asked_subject(q), C, q)

    def test_ambiguous_or_none(self):
        self.assertIsNone(metamem.asked_subject("我和你哪天认识"))
        self.assertIsNone(metamem.asked_subject("哪天生日"))


class JudgeSubject(unittest.TestCase):
    def test_claude_birthday_does_not_answer_user_question(self):
        """原缺陷：问我的生日，命中是助理的生日，判成「够了」。"""
        hits = [hit(C, f"{persona.N_CLAUDE}的生日是3月3日", "a")]
        j = metamem.judge("我哪天生日", hits)
        self.assertFalse(j.enough)
        self.assertEqual(j.fok, 0.0)

    def test_user_birthday_answers_user_question(self):
        hits = [hit(C, f"{persona.N_CLAUDE}的生日是3月3日", "a"),
                hit(U, f"{persona.N_USER}的生日是5月20日", "b")]
        j = metamem.judge("我哪天生日", hits)
        self.assertTrue(j.enough, j.reason)

    def test_question_about_claude_needs_claude_fact(self):
        hits = [hit(U, f"{persona.N_USER}的生日是5月20日", "b")]
        self.assertFalse(metamem.judge("你哪天生日", hits).enough)
        hits.append(hit(C, f"{persona.N_CLAUDE}的生日是3月3日", "a"))
        self.assertTrue(metamem.judge("你哪天生日", hits).enough)

    def test_text_lead_names_other_party_even_if_subject_wrong(self):
        hits = [hit(U, f"{persona.N_CLAUDE}的生日是3月3日", "a")]
        self.assertFalse(metamem.judge("我哪天生日", hits).enough)

    def test_ambiguous_question_not_filtered(self):
        hits = [hit(C, "我们是3月3日认识的", "a")]
        self.assertTrue(metamem.judge("我和你哪天认识", hits).enough)

    def test_us_facts_count_for_both(self):
        hits = [hit("us", "我们第一次聊天是3月3日", "a")]
        self.assertTrue(metamem.judge("我哪天第一次聊天", hits).enough)

    def test_no_subject_field_falls_back_to_text(self):
        self.assertFalse(metamem.judge("我哪天生日", [{"content": f"{persona.N_CLAUDE}的生日是3月3日"}]).enough)
        self.assertTrue(metamem.judge("我哪天生日", [{"content": "生日是5月20日"}]).enough)


class PreferSubject(unittest.TestCase):
    def test_reorders_without_dropping(self):
        hs = [hit(C, f"{persona.N_CLAUDE}生日3月3日", "a"), hit(U, "生日5月20日", "b")]
        out = metamem.prefer_subject("我哪天生日", hs)
        self.assertEqual([h["ext_id"] for h in out], ["b", "a"])
        self.assertEqual(len(out), 2)

    def test_unknown_owner_keeps_order(self):
        hs = [hit(C, "x", "a"), hit(U, "y", "b")]
        self.assertEqual([h["ext_id"] for h in metamem.prefer_subject("哪天生日", hs)], ["a", "b"])


class DbBacked(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def _fact(self, subject, content, actor=None, status="candidate"):
        eids = []
        if actor:
            eids = [add_evidence(self.con, source_kind="chat", source_actor=actor,
                                 source_locator=f"conv:t#msg:{abs(hash(content)) % 10**8}", content=content)]
        fid = promote(self.con, subject=subject, fact_type="semantic", content=content,
                      evidence_ids=eids, status=status, valid_from="2026-09-01",
                      importance=0.5, confidence=0.7)
        if eids and fid:
            self.con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,'support')",
                             (fid, eids[0]))
            self.con.commit()
        return self.con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]

    def test_same_key_different_subjects_is_not_conflict(self):
        a = self._fact(U, "生日是5月20日")
        b = self._fact(C, "生日是3月3日")
        for ext, v in ((a, "5月20日"), (b, "3月3日")):
            fid = self.con.execute("SELECT id FROM facts WHERE ext_id=?", (ext,)).fetchone()[0]
            record_key(self.con, fid, "person.birthday", v, "pos", True)
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="single")):
            ev = metamem.check_evidence(self.con, [{"ext_id": a}, {"ext_id": b}], lexical=False)
        self.assertEqual(ev.conflicts, ())

    def test_same_key_same_subject_still_conflict(self):
        a = self._fact(U, "老家在甲市")
        b = self._fact(U, "老家在乙县")
        for ext, v in ((a, "甲市"), (b, "乙县")):
            fid = self.con.execute("SELECT id FROM facts WHERE ext_id=?", (ext,)).fetchone()[0]
            record_key(self.con, fid, "person.hometown", v, "pos", True)
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="single")):
            ev = metamem.check_evidence(self.con, [{"ext_id": a}, {"ext_id": b}], lexical=False)
        self.assertTrue(ev.conflicts)

    def test_user_subject_but_only_claude_spoke_is_not_users_fact(self):
        """说话人/主体错位：记在她名下、证据却只有助理在说 → 不算她的事。"""
        ext = self._fact(U, "生日是3月3日", actor=C, status="confirmed")   # confirmed 才能绕过入库防回声
        h = {"ext_id": ext, "subject": U, "content": "生日是3月3日", "score": 0.9}
        j, _ = metamem.judge_with_evidence(self.con, "我哪天生日", [h])
        self.assertFalse(j.enough)

    def test_user_subject_spoken_by_user_is_ok(self):
        ext = self._fact(U, "生日是5月20日", actor=U)
        h = {"ext_id": ext, "subject": U, "content": "生日是5月20日", "score": 0.9}
        j, _ = metamem.judge_with_evidence(self.con, "我哪天生日", [h])
        self.assertTrue(j.enough, j.reason)

    def test_hybrid_recall_puts_users_fact_first_when_metamem_on(self):
        import retrieve
        self._fact(C, f"{persona.N_CLAUDE}的生日是3月3日，生日蛋糕是抹茶味", actor=C)
        self._fact(U, f"{persona.N_USER}的生日是5月20日，生日想吃芒果蛋糕", actor=U)
        with mock.patch.dict("os.environ", {"ENGRAM_METAMEM": "1"}):
            r = retrieve.retrieve(self.con, "我哪天生日")
        self.assertTrue(r.hits)
        self.assertEqual(r.hits[0]["subject"], U)
        with mock.patch.dict("os.environ", {"ENGRAM_METAMEM": "0"}):
            r2 = retrieve.retrieve(self.con, "我哪天生日")
        self.assertEqual(len(r2.hits), len(r.hits))      # 只换顺序，不丢条目


if __name__ == "__main__":
    unittest.main()
