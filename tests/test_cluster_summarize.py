"""任务四·步骤3（2026-10-29）：夜间向量聚类归纳——模型全部 mock，向量是手造的小维度，夹具全虚构。"""
import hashlib
import io
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))
import persona  # noqa: E402
import cluster_summarize as CS  # noqa: E402
from store.store import add_evidence, confirm, connect, promote, record_key  # noqa: E402

try:
    import numpy as np
except ImportError:                                          # pragma: no cover
    np = None

U, C = persona.USER, persona.CLAUDE
ON = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_CLUSTER_SUMMARY": "1"}
ROOT = Path(__file__).resolve().parent.parent


def unit(angle_deg, dim=6, axis=0):
    v = np.zeros(dim, np.float32)
    a = math.radians(angle_deg)
    v[axis], v[axis + 1] = math.cos(a), math.sin(a)
    return v


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.con.execute("CREATE TABLE IF NOT EXISTS fact_vec (fact_id INTEGER PRIMARY KEY, model TEXT NOT NULL, vec BLOB NOT NULL)")
        self.n = 0

    def fact(self, content, day, vec, actor=U, subject=U, status="candidate", evidence_from=None):
        """造一条带证据、带向量的事实。evidence_from：复用另一条事实的证据（同一事件）。"""
        if evidence_from is None:
            self.n += 1
            eid = add_evidence(self.con, source_kind="chat", source_actor=actor,
                               source_locator=f"conv:t#msg:{self.n}", content=content, occurred_at=day)
            self.con.execute("INSERT OR IGNORE INTO evidence_tier (evidence_id, tier, origin_root, built_at)"
                             " VALUES (?,?,?,datetime('now'))", (eid, "primary", f"root{self.n}"))
        else:
            eid = self.con.execute("SELECT evidence_id FROM fact_evidence WHERE fact_id=?", (evidence_from,)).fetchone()[0]
        fid = promote(self.con, subject=subject, fact_type="episodic", content=content, evidence_ids=[eid],
                      status=status, valid_from=day, importance=0.5, confidence=0.7)
        self.assertIsNotNone(fid, content)
        self.con.execute("INSERT OR IGNORE INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,'support')", (fid, eid))
        self.con.execute("INSERT INTO fact_vec (fact_id, model, vec) VALUES (?,?,?)",
                         (fid, "t", np.asarray(vec, np.float32).tobytes()))
        self.con.commit()
        return fid

    def group(self, texts, days=("2026-09-01", "2026-09-05", "2026-09-09", "2026-09-12"), axis=0):
        return [self.fact(t, days[i % len(days)], unit(i * 3, axis=axis)) for i, t in enumerate(texts)]

    def snapshot(self):
        return [tuple(r) for r in self.con.execute(
            "SELECT id, ext_id, subject, content, status, valid_to, user_pinned, version, updated_at"
            " FROM facts ORDER BY id")]

    def summaries(self):
        return self.con.execute("SELECT id, content, subject, status FROM facts WHERE content LIKE ?",
                                (CS.LABEL + "%",)).fetchall()


def reply(summary, ids, conflict=False):
    return json.dumps({"summary": summary, "ids": ids, "conflict": conflict}, ensure_ascii=False)


@unittest.skipIf(np is None, "需要 numpy（venv）")
class Clustering(unittest.TestCase):
    def test_average_linkage_is_not_single_linkage(self):
        # 三个点：1-2、2-3 相似度 0.8，1-3 只有 0.28。单链会连成一簇，average linkage 不会
        a, b = math.degrees(math.acos(0.82)), math.degrees(math.acos(0.78))
        V = np.stack([unit(0), unit(a), unit(a + b)])
        self.assertEqual(len(CS.components(V, 0.75)), 1)
        got = sorted(sorted(c) for c in CS.cluster(V, 0.75))
        self.assertEqual(got, [[0, 1], [2]])

    def test_tight_group_merges_and_far_apart_stays_single(self):
        V = np.stack([unit(0), unit(4), unit(8), unit(90)])
        got = sorted(sorted(c) for c in CS.cluster(V, 0.75))
        self.assertEqual(got, [[0, 1, 2], [3]])

    def test_matches_scipy_average_linkage_when_available(self):
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
        except ImportError:
            self.skipTest("没有 scipy（只在对拍时用，生产不依赖）")
        from collections import defaultdict
        for seed, scale, thr in ((2, 0.9, 0.4), (4, 0.7, 0.65), (7, 0.55, 0.75)):
            rng = np.random.default_rng(seed)
            pts = []
            for c in rng.normal(size=(60, 24)):
                for _ in range(int(rng.integers(1, 15))):
                    pts.append(c + rng.normal(scale=scale, size=24))
            P = np.array(pts, dtype=np.float32)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            lab = fcluster(linkage(P.astype(np.float64), method="average", metric="cosine"), 1 - thr, criterion="distance")
            d = defaultdict(list)
            for i, l in enumerate(lab):
                d[l].append(i)
            self.assertEqual(sorted(sorted(c) for c in CS.cluster(P, thr)), sorted(sorted(v) for v in d.values()), (seed, thr))

    def test_threshold_boundary(self):
        V = np.stack([unit(0), unit(math.degrees(math.acos(0.74)))])
        self.assertEqual(len(CS.cluster(V, 0.75)), 2)
        V = np.stack([unit(0), unit(math.degrees(math.acos(0.76)))])
        self.assertEqual(len(CS.cluster(V, 0.75)), 1)


@unittest.skipIf(np is None, "需要 numpy（venv）")
class Pipeline(Base):
    TEXTS = ["我每周六都会去公园遛狗", "我周六早上又去公园遛狗了", "我今天周六去公园遛狗", "我带狗去公园是周六的习惯"]

    def run_cs(self, answer, **kw):
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return answer
        with mock.patch.object(CS, "ask", fake):
            st = CS.run(self.con, apply=True, **kw)
        return st, calls

    def test_happy_path_writes_new_derived_fact_and_leaves_originals(self):
        ids = self.group(self.TEXTS)
        confirm(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (ids[0],)).fetchone()[0])
        self.con.commit()
        before = self.snapshot()
        st, calls = self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        self.assertEqual((st["summary"], st["calls"]), (1, 1))
        s = self.summaries()
        self.assertEqual(len(s), 1)
        self.assertTrue(s[0]["content"].startswith("〔归纳〕"))
        self.assertEqual((s[0]["subject"], s[0]["status"]), (U, "candidate"))
        # 原事实一行不改（包括 confirmed 那条的 status/version/updated_at）
        self.assertEqual(self.snapshot()[:len(before)], before)
        rel = self.con.execute("SELECT child_fact_id FROM fact_relations WHERE parent_fact_id=? AND relation='derived_from'",
                               (s[0]["id"],)).fetchall()
        self.assertEqual(sorted(r[0] for r in rel), sorted(ids))
        self.assertEqual(self.con.execute("SELECT outcome FROM cluster_summary_log").fetchone()[0], "summary")

    def test_second_run_is_idempotent_and_skips_llm(self):
        ids = self.group(self.TEXTS)
        self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        st, calls = self.run_cs(reply("别的", ids))
        self.assertEqual(calls, [])
        self.assertEqual(len(self.summaries()), 1)
        self.assertEqual(st["skipped_done"], 1)

    def test_summary_is_not_clustered_again(self):
        ids = self.group(self.TEXTS)
        self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        sid = self.summaries()[0]["id"]
        self.con.execute("INSERT INTO fact_vec (fact_id, model, vec) VALUES (?,?,?)",
                         (sid, "t", unit(1).tobytes()))
        got, _ = CS.load_vectors(self.con)
        self.assertNotIn(sid, got)

    def test_dry_run_calls_nothing_writes_nothing(self):
        self.group(self.TEXTS)
        before = self.snapshot()
        with mock.patch.object(CS, "ask", side_effect=AssertionError("dry run must not call the model")):
            st = CS.run(self.con, apply=False)
        self.assertEqual(st["eligible"], 1)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM cluster_summary_log").fetchone()[0], 0)

    def test_only_users_own_words_reach_the_prompt(self):
        ids = self.group(self.TEXTS)
        self.fact("助理说她周六大概会去公园", "2026-09-02", unit(1), actor=C, subject=C)
        self.fact("我周六遛狗时把密码 abc123 告诉了朋友", "2026-09-03", unit(2))
        st, calls = self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("助理说", calls[0])
        self.assertNotIn("密码", calls[0])
        self.assertEqual(st["summary"], 1)

    def test_not_eligible_clusters_never_call_model(self):
        self.fact("我周六去公园遛狗", "2026-09-01", unit(0))
        self.fact("我周六去公园遛狗了", "2026-09-01", unit(3))       # 同一天
        self.fact("我周六去公园遛狗呢", "2026-09-01", unit(6))
        st, calls = self.run_cs(reply("x", []))
        self.assertEqual(calls, [])
        self.assertEqual(st["eligible"], 0)
        # 本人原话不足：2 条她的 + 1 条助理的
        self.fact("我爱吃芒果", "2026-09-01", unit(0, axis=2))
        self.fact("助理说她爱吃芒果", "2026-09-04", unit(3, axis=2), actor=C, subject=C)
        self.fact("助理又说她爱吃芒果", "2026-09-05", unit(5, axis=2), actor=C, subject=C)
        st, calls = self.run_cs(reply("x", []))
        self.assertEqual(calls, [])

    # ── 代码后检 ──
    def test_incidental_negation_is_hint_not_hard_reject(self):
        """词面上一条带「不」的，不再直接拒簇：交给模型（提示里带一句），模型认为没矛盾就照常写。"""
        ids = self.group(["我在做记账本小程序", "我记账本小程序做完了", "我继续做记账本小程序", "我不做记账本小程序了"])
        st, calls = self.run_cs(reply("她一直在做记账本小程序", ids))
        self.assertEqual(len(calls), 1)
        self.assertIn("提示", calls[0])
        self.assertIn("否定", calls[0])
        self.assertEqual(st["rejected"], 0)

    def test_model_says_conflict_after_hint_is_rejected(self):
        ids = self.group(["我在做记账本小程序", "我记账本小程序做完了", "我继续做记账本小程序", "我不做记账本小程序了"])
        st, calls = self.run_cs(reply("NONE", ids, conflict=True))
        self.assertEqual((len(calls), st["rejected"], self.summaries()), (1, 1, []))

    def test_no_hint_when_no_negation(self):
        ids = self.group(self.TEXTS)
        _, calls = self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        self.assertNotIn("提示（词面", calls[0])

    def test_lexical_hint_pure_function(self):
        mem = lambda i, t: {"id": i, "content": t, "day": "2026-09-0%d" % i, "events": {i}, "own": True}  # noqa: E731
        with mock.patch.object(CS, "is_own", return_value=True):
            self.assertIsNotNone(CS.lexical_hint([mem(1, "我在做记账本小程序"), mem(2, "我不做记账本小程序了")]))
            self.assertIsNone(CS.lexical_hint([mem(1, "我在做记账本小程序"), mem(2, "今天不知道吃什么好")]))

    def test_first_person_summary_rejected(self):
        ids = self.group(self.TEXTS)
        st, _ = self.run_cs(reply("我每周六都去公园遛狗", ids))
        self.assertEqual(len(self.summaries()), 0)

    def test_fact_key_contradiction(self):
        ids = self.group(["我老家在甲市", "我家乡是甲市来的", "我老家甲市天气很热"])
        for fid, v in zip(ids, ("甲市", "甲市", "乙县")):
            record_key(self.con, fid, "person.hometown", v, "pos", True)
        self.con.commit()
        with mock.patch("keys.by_name", return_value=mock.Mock(cardinality="single")):
            _, calls = self.run_cs(reply("她老家在甲市", ids))
        self.assertEqual(calls, [])
        self.assertEqual(self.summaries(), [])

    def test_model_flagged_conflict_rejected(self):
        ids = self.group(self.TEXTS)
        st, _ = self.run_cs(reply("她有周六去公园遛狗的习惯", ids, conflict=True))
        self.assertEqual((st["rejected"], len(self.summaries())), (1, 0))

    def test_none_answer_recorded_not_retried(self):
        ids = self.group(self.TEXTS)
        st, _ = self.run_cs(reply("NONE", ids))
        self.assertEqual((st["none"], len(self.summaries())), (1, 0))
        _, calls = self.run_cs(reply("NONE", ids))
        self.assertEqual(calls, [])

    def test_frequency_claim_on_single_event_rejected(self):
        # 两条记录出自同一句原话（同一个事件），不同日期：「多次」撑不起
        a = self.fact("我周六去了公园遛狗", "2026-09-01", unit(0))
        b = self.fact("我周六去公园遛狗了", "2026-09-05", unit(3), evidence_from=a)
        c = self.fact("我周六遛狗去了公园", "2026-09-09", unit(6), evidence_from=a)
        st, _ = self.run_cs(reply("她多次周六去公园遛狗", [a, b, c]))
        self.assertEqual((st["rejected"], len(self.summaries())), (1, 0))
        self.assertIn("frequency", self.con.execute("SELECT reason FROM cluster_summary_log").fetchone()[0])

    def test_duo_claim_two_events_two_days_passes(self):
        a = self.fact("我周六去了公园遛狗", "2026-09-01", unit(0))
        b = self.fact("我周六去公园遛狗了", "2026-09-05", unit(3))
        c = self.fact("我周六遛狗去了公园", "2026-09-05", unit(6), evidence_from=b)
        st, _ = self.run_cs(reply("她多次周六去公园遛狗", [a, b, c]))
        self.assertEqual(st["summary"], 1)

    def test_frequency_claim_with_enough_events_passes(self):
        ids = self.group(self.TEXTS)
        st, _ = self.run_cs(reply("她经常周六去公园遛狗", ids))
        self.assertEqual(st["summary"], 1)

    def test_strong_frequency_word_needs_three_events(self):
        a = self.fact("我周六去了公园遛狗", "2026-09-01", unit(0))
        b = self.fact("我周六去公园遛狗了", "2026-09-05", unit(3))
        c = self.fact("我周六遛狗去了公园", "2026-09-05", unit(6), evidence_from=b)
        st, _ = self.run_cs(reply("她总是周六去公园遛狗", [a, b, c]))
        self.assertEqual(st["rejected"], 1)
        self.assertEqual(self.summaries(), [])

    def test_cited_ids_must_be_own_words(self):
        ids = self.group(self.TEXTS)
        bad = self.fact("助理说她周六去公园", "2026-09-02", unit(1), actor=C, subject=C)
        st, _ = self.run_cs(reply("她有周六去公园遛狗的习惯", [ids[0], bad]))
        self.assertEqual((st["rejected"], len(self.summaries())), (1, 0))

    def test_invented_number_rejected(self):
        ids = self.group(self.TEXTS)
        st, _ = self.run_cs(reply("她每周六去公园遛狗两小时，共 20 次", ids))
        self.assertEqual((st["rejected"], len(self.summaries())), (1, 0))

    def test_max_calls_cap_and_resume(self):
        a = self.group(self.TEXTS)
        b = self.group(["我爱喝无糖豆浆", "我又买了无糖豆浆", "我早餐喝无糖豆浆", "我家冰箱里总有无糖豆浆"], axis=2)
        st, calls = self.run_cs(reply("某个习惯一句话", a), max_calls=1)
        self.assertEqual(st["calls"], 1)
        st2, calls2 = self.run_cs(reply("某个习惯一句话", b), max_calls=5)
        self.assertEqual(st2["calls"], 1)               # 明晚接着做剩下的

    def test_model_failure_skips_cluster_and_retries_next_night(self):
        ids = self.group(self.TEXTS)
        with mock.patch.object(CS, "ask", side_effect=RuntimeError("boom")):
            st = CS.run(self.con, apply=True)
        self.assertEqual(st["summary"], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM cluster_summary_log").fetchone()[0], 0)
        st, _ = self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        self.assertEqual(st["summary"], 1)

    def test_grown_cluster_covered_by_old_summary_not_duplicated(self):
        ids = self.group(self.TEXTS)
        self.run_cs(reply("她有周六去公园遛狗的习惯", ids))
        self.fact("我周六又去公园遛狗了", "2026-09-20", unit(12))
        st, calls = self.run_cs(reply("别的归纳", ids))
        self.assertEqual(calls, [])
        self.assertEqual((st["covered"], len(self.summaries())), (1, 1))


class RecallOrder(unittest.TestCase):
    def test_summary_ranks_behind_original_words_but_stays(self):
        import hybrid
        merged = {"fact_a": {"ext_id": "fact_a", "rrf": 0.03, "via": ["fts"], "content": "〔归纳〕她有周六遛狗的习惯"},
                  "fact_b": {"ext_id": "fact_b", "rrf": 0.03, "via": ["fts"], "content": "我周六去公园遛狗"}}
        out = hybrid.finalize(merged)
        self.assertEqual([h["ext_id"] for h in out], ["fact_b", "fact_a"])
        self.assertAlmostEqual(out[1]["hybrid_score"], round(0.03 * hybrid.SUMMARY_WEIGHT, 5))
        self.assertEqual(hybrid.SUMMARY_LABEL, CS.LABEL)


class Main(unittest.TestCase):
    def test_flag_off_does_nothing_and_never_connects(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True), \
             mock.patch("store.store.connect", side_effect=AssertionError("must not connect")), redirect_stdout(buf):
            self.assertEqual(CS.main([]), 0)
        self.assertIn("已关闭", buf.getvalue())

    def test_flag_is_registered_default_off(self):
        import flags
        self.assertFalse(flags.DEFAULTS["CLUSTER_SUMMARY"])

    def test_nightly_script_exists_and_installs_no_crontab(self):
        p = ROOT / "maintenance" / "nightly-cluster.sh"
        self.assertTrue(p.exists() and os.access(p, os.X_OK))
        txt = p.read_text(encoding="utf-8")
        self.assertNotIn("crontab -", txt.replace("crontab 由助手装", ""))
        self.assertIn(".ingest.lock", txt)
        self.assertIn("--apply", txt)

    def test_checks_are_pure_python_without_numpy(self):
        self.assertIsNone(CS.check_frequency("她喜欢喝茶", []))
        self.assertIsNotNone(CS.check_frequency("她经常喝茶", [{"events": {"a"}, "day": "2026-01-01"}]))


if __name__ == "__main__":
    unittest.main()
