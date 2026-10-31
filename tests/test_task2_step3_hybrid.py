"""任务二·步骤3：混合召回（FTS + 向量 → RRF），旧分层路径在开关后"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hybrid  # noqa: E402
import retrieve  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

ENV = {"ENGRAM_ENV_FILE": "/nonexistent"}


def H(ext, rank_score=1.0, **kw):
    return dict({"ext_id": ext, "subject": "tester", "content": ext, "status": "candidate",
                 "user_pinned": 0, "layer": "active", "score": rank_score}, **kw)


class Pure(unittest.TestCase):
    def test_rrf_both_lists_beats_single(self):
        m = hybrid.rrf_merge({"fts": [H("a"), H("b")], "vector": [H("c"), H("b")]})
        out = hybrid.finalize(m)
        self.assertEqual(out[0]["ext_id"], "b")                  # 两路都有
        self.assertEqual(out[0]["via"], "fts+vector")

    def test_pinned_and_confirmed_priority(self):
        m = hybrid.rrf_merge({"fts": [H("x1"), H("x2"), H("x3"), H("p", user_pinned=1),
                                      H("c", status="confirmed")]})
        ids = [h["ext_id"] for h in hybrid.finalize(m)]
        self.assertLess(ids.index("p"), ids.index("x3"))
        self.assertLess(ids.index("c"), ids.index("x3"))

    def test_cold_downweighted(self):
        m = hybrid.rrf_merge({"fts": [H("cold", layer="cold"), H("hot")]})
        ids = [h["ext_id"] for h in hybrid.finalize(m)]
        self.assertEqual(ids, ["hot", "cold"])

    def test_cap(self):
        m = hybrid.rrf_merge({"fts": [H(f"f{i}") for i in range(30)]})
        self.assertEqual(len(hybrid.finalize(m, cap=12)), 12)


class OnRealStore(unittest.TestCase):
    """真实库结构：状态过滤、disown 过滤、向量失败退回 FTS、开关回滚。"""
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def fact(self, content, **kw):
        a = dict(subject="tester", fact_type="episodic", content=content,
                 valid_from="2026-09-01", importance=0.6, confidence=0.8)
        a.update(kw)
        return promote(self.con, **a)

    def run_r(self, text="上次那个星光湖是哪年的事", vec=(), env=ENV):
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(hybrid, "_vsearch", return_value=list(vec)):
            return retrieve.retrieve(self.con, text)

    def test_fts_only_when_vector_fails(self):
        self.fact("我们去看了星光湖，脚印会亮")
        r = self.run_r()
        self.assertEqual(r.mode, "hybrid")
        self.assertTrue(any("星光湖" in h["content"] for h in r.hits))

    def test_invalid_status_not_returned(self):
        self.fact("星光湖那天下雨了")
        self.con.execute("UPDATE facts SET status='uncertain'")
        self.con.commit()
        r = self.run_r()
        self.assertEqual(r.hits, [])

    def test_vector_hits_merged_and_disown_filtered(self):
        self.fact("星光湖很好看")
        v = [H("fact_vecA", content="换个说法的旧事", via="vector", score=0.8)]
        r = self.run_r(vec=v)
        self.assertIn("fact_vecA", [h["ext_id"] for h in r.hits])
        with mock.patch.object(hybrid, "_disowned", return_value={"fact_vecA"}):
            r = self.run_r(vec=v)
        self.assertNotIn("fact_vecA", [h["ext_id"] for h in r.hits])

    def test_layered_flag_rolls_back(self):
        self.fact("我们去看了星光湖，脚印会亮")
        r = self.run_r(env={**ENV, "ENGRAM_LAYERED_RETRIEVE": "1"})
        self.assertEqual(r.mode, "layered")
        self.assertTrue(r.should_recall)

    def test_soft_talk_still_skips(self):
        self.fact("想你了的那天我们吵架")
        r = self.run_r(text="想你了")
        self.assertFalse(r.should_recall)
        self.assertEqual(r.hits, [])

    def test_telemetry_mode_hybrid(self):
        self.fact("我们去看了星光湖，脚印会亮")
        r = self.run_r()
        with mock.patch.dict(os.environ, ENV, clear=True):
            rid = retrieve.log_recall(self.con, "上次那个星光湖是哪年的事", r)
        mode = self.con.execute("SELECT retrieval_mode FROM recall_runs WHERE id=?", (rid,)).fetchone()[0]
        self.assertEqual(mode, "hybrid")


if __name__ == "__main__":
    unittest.main()
