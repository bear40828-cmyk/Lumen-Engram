"""任务四·步骤5（2026-10-29）：按上下文分发到人（CONTEXT_ROUTE）+ 旧事实拆原子工具（ATOMIC_SPLIT，不对正式库跑）。虚构夹具，模型 mock。"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))
import persona  # noqa: E402
import router  # noqa: E402
import split_atomic as SA  # noqa: E402
from store.store import add_evidence, confirm, connect, pin, promote  # noqa: E402

U, C = persona.USER, persona.CLAUDE
OFF = {"ENGRAM_ENV_FILE": "/nonexistent"}


class InferWho(unittest.TestCase):
    def test_single_party(self):
        self.assertEqual(router.infer_who("我哪天生日"), U)
        self.assertEqual(router.infer_who("你喜欢什么颜色"), C)
        self.assertEqual(router.infer_who(f"{persona.N_GPT}之前说过什么"), persona.GPT)
        self.assertEqual(router.infer_who(f"{persona.N_C}最近怎么样"), persona.AGENT_C)

    def test_ambiguous_is_none(self):
        for q in ("我和你哪天认识", f"我跟{persona.N_GPT}聊过什么", "今天吃什么", ""):
            self.assertIsNone(router.infer_who(q), q)

    def test_framing_pronoun_stripped(self):
        self.assertEqual(router.infer_who("你记得我哪天生日吗"), U)


class ContextRouteRanking(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        for subj, txt in ((C, f"{persona.N_CLAUDE}周末喜欢整理书架，也喜欢泡茶"),
                          (U, f"{persona.N_USER}周末喜欢整理书架，也喜欢泡茶")):
            promote(self.con, subject=subj, fact_type="preference", content=txt, valid_from="2026-09-01",
                    importance=0.5, confidence=0.7)

    def subjects(self, env, q="我周末喜欢整理书架吗"):
        import retrieve
        with mock.patch.dict(os.environ, {**OFF, **env}, clear=True):
            r = retrieve.retrieve(self.con, q)
        return [h["subject"] for h in r.hits], r

    def test_flag_off_is_unchanged_and_no_hint(self):
        subs, r = self.subjects({})
        self.assertIsNone(r.who_hint)
        self.assertEqual(sorted(subs), sorted([C, U]))

    def test_flag_on_prefers_users_fact_without_dropping(self):
        subs, r = self.subjects({"ENGRAM_CONTEXT_ROUTE": "1"})
        self.assertEqual(r.who_hint, U)
        self.assertEqual(subs[0], U)
        self.assertEqual(sorted(subs), sorted([C, U]))
        subs2, r2 = self.subjects({"ENGRAM_CONTEXT_ROUTE": "1"}, "你周末喜欢整理书架吗")
        self.assertEqual((r2.who_hint, subs2[0]), (C, C))

    def test_explicit_who_wins(self):
        import retrieve
        with mock.patch.dict(os.environ, {**OFF, "ENGRAM_CONTEXT_ROUTE": "1"}, clear=True):
            r = retrieve.retrieve(self.con, "我周末喜欢整理书架吗", who=C)
        self.assertIsNone(r.who_hint)
        self.assertEqual({h["subject"] for h in r.hits}, {C})

    def test_bonus_is_small_and_only_reorders(self):
        import hybrid
        merged = {"a": {"ext_id": "a", "subject": C, "rrf": 0.03, "via": ["fts"]},
                  "b": {"ext_id": "b", "subject": U, "rrf": 0.03 - 0.005, "via": ["fts"]}}
        out = hybrid.finalize({k: dict(v) for k, v in merged.items()}, who_hint=U)
        self.assertEqual([h["ext_id"] for h in out], ["b", "a"])
        far = {"a": {"ext_id": "a", "subject": C, "rrf": 0.05, "via": ["fts"]},
               "b": {"ext_id": "b", "subject": U, "rrf": 0.02, "via": ["fts"]}}
        self.assertEqual([h["ext_id"] for h in hybrid.finalize(far, who_hint=U)], ["a", "b"])   # 差得远的翻不过去


ORIG = "今天把路由改完了，而且她说她喜欢粉色；以后不许在回复里用 Markdown 表格；下周三下午要去办理 3 号窗口的手续"


def atoms_json(*items):
    return json.dumps({"atoms": [{"content": c, "quote": q, "type": t} for c, q, t in items]}, ensure_ascii=False)


GOOD = atoms_json(("今天把路由改完了", "今天把路由改完了", "episodic"),
                  ("她喜欢粉色", "她喜欢粉色", "preference"),
                  ("以后不许在回复里用 Markdown 表格", "以后不许在回复里用 Markdown 表格", "procedural"),
                  ("下周三下午要去办理 3 号窗口的手续", "下周三下午要去办理 3 号窗口的手续", "episodic"))


class AtomicChecks(unittest.TestCase):
    def test_looks_compound(self):
        self.assertTrue(SA.looks_compound(ORIG))
        self.assertFalse(SA.looks_compound("她喜欢粉色"))
        self.assertFalse(SA.looks_compound("她周末喜欢整理书架然后泡一壶茶坐在窗边看书看到很晚才睡觉" * 2))   # 长但只有一个分句

    def test_good_split(self):
        good, why = SA.check_atoms(ORIG, SA.parse(GOOD))
        self.assertEqual((len(good), why), (4, "split"))

    def test_single_atom_means_already_atomic(self):
        self.assertEqual(SA.check_atoms(ORIG, [{"content": "x" * 10, "quote": "今天把路由改完了"}]), (None, "atomic"))
        self.assertEqual(SA.check_atoms(ORIG, SA.parse('{"atoms": []}')), (None, "atomic"))

    def test_rejections(self):
        def two(a, b):
            return [a, b]
        ok1 = {"content": "今天把路由改完了", "quote": "今天把路由改完了"}
        ok2 = {"content": "她喜欢粉色", "quote": "她喜欢粉色"}
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, {**ok2, "quote": "她其实不喜欢粉色"}))[1], "quote-not-verbatim")
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, {**ok2, "content": "她喜欢粉色，一共 7 件衣服"}))[1], "invented-number")
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, {**ok2, "content": "她最爱去海边晒太阳游泳"}))[1], "not-from-original")
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, {**ok2, "content": "她喜欢粉色，密码是1234"}))[1], "secret")
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, ok1))[1], "duplicate-atom")
        self.assertEqual(SA.check_atoms(ORIG, [ok1] * 7)[1].split(":")[0], "too-many-atoms")
        self.assertEqual(SA.check_atoms(ORIG, two(ok1, {**ok2, "content": "粉"}))[1], "atom-length")


class AtomicRun(unittest.TestCase):
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

    def snap(self):
        return [tuple(r) for r in self.con.execute(
            "SELECT id, content, status, valid_to, user_pinned, version, updated_at FROM facts ORDER BY id")]

    def run_sa(self, answer, **kw):
        calls = []
        with mock.patch.object(SA, "ask", lambda p: calls.append(p) or answer):
            st = SA.run(self.con, apply=True, **kw)
        return st, calls

    def test_dry_run_no_model_no_write(self):
        self.fact(ORIG)
        before = self.snap()
        with mock.patch.object(SA, "ask", side_effect=AssertionError("dry run must not call the model")):
            st = SA.run(self.con, apply=False)
        self.assertEqual(st["compound"], 1)
        self.assertEqual(self.snap(), before)

    def test_split_writes_new_derived_facts_and_leaves_original(self):
        orig = self.fact(ORIG)
        before = self.snap()
        st, calls = self.run_sa(GOOD)
        self.assertEqual((st["split"], st["atoms_written"], len(calls)), (1, 4, 1))
        self.assertEqual(self.snap()[:len(before)], before)                      # 原事实一行没动
        kids = self.con.execute("SELECT parent_fact_id FROM fact_relations WHERE child_fact_id=? AND relation='derived_from'",
                                (orig,)).fetchall()
        self.assertEqual(len(kids), 4)
        rows = self.con.execute("SELECT subject, status, valid_from, fact_type FROM facts WHERE id IN (%s)"
                                % ",".join(str(k[0]) for k in kids)).fetchall()
        self.assertTrue(all(r[0] == U and r[1] == "candidate" and r[2] == "2026-09-01" for r in rows))
        self.assertEqual(sorted(r[3] for r in rows), ["episodic", "episodic", "preference", "procedural"])
        ev = self.con.execute("SELECT COUNT(*) FROM fact_evidence WHERE fact_id IN (%s)" % ",".join(str(k[0]) for k in kids)).fetchone()[0]
        self.assertEqual(ev, 4)                                                   # 原子事实挂着同一条原话证据

    def test_second_run_skips_processed_and_derived(self):
        self.fact(ORIG)
        self.run_sa(GOOD)
        st, calls = self.run_sa(GOOD)
        self.assertEqual((calls, st["compound"]), ([], 0))

    def test_confirmed_and_pinned_never_touched_or_selected(self):
        a = self.fact(ORIG, status="confirmed")
        b = self.fact(ORIG + "（另一条）")
        pin(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (b,)).fetchone()[0])
        self.con.commit()
        before = self.snap()
        st, calls = self.run_sa(GOOD)
        self.assertEqual((calls, st["split"]), ([], 0))
        self.assertEqual(self.snap(), before)

    def test_bad_split_is_logged_and_writes_nothing(self):
        self.fact(ORIG)
        bad = atoms_json(("今天把路由改完了", "今天把路由改完了", "episodic"), ("她喜欢粉色", "她根本没说过的话", "preference"))
        st, _ = self.run_sa(bad)
        self.assertEqual((st["rejected"], st["atoms_written"]), (1, 0))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)
        self.assertEqual(self.con.execute("SELECT outcome, reason FROM atomic_split_log").fetchone()[:], ("rejected", "quote-not-verbatim"))

    def test_model_failure_not_logged_retry_later(self):
        self.fact(ORIG)
        with mock.patch.object(SA, "ask", side_effect=RuntimeError("x")):
            st = SA.run(self.con, apply=True)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM atomic_split_log").fetchone()[0], 0)

    def test_limit(self):
        self.fact(ORIG)
        self.fact(ORIG.replace("粉色", "蓝色").replace("路由", "菜单"))
        st, calls = self.run_sa(GOOD, limit=1)
        self.assertEqual(len(calls), 1)

    def test_main_requires_db_and_flag(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, OFF, clear=True), mock.patch("store.store.connect",
                                                                     side_effect=AssertionError("no connect")), redirect_stdout(buf):
            self.assertEqual(SA.main(["--db", "/nonexistent.db"]), 0)
        self.assertIn("已关闭", buf.getvalue())
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            SA.main([])                       # 没给 --db 直接报错：不会默认连到正式库

    def test_flags_default_off(self):
        import flags
        self.assertFalse(flags.DEFAULTS["ATOMIC_SPLIT"])
        self.assertFalse(flags.DEFAULTS["CONTEXT_ROUTE"])


if __name__ == "__main__":
    unittest.main()
