"""LongMemEval 评测脚本冒烟测试：虚构的两道题，--mock 不调任何模型。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tests  # noqa: F401  虚构人设

REPO = Path(__file__).resolve().parents[1]


def item(qid, qtype, q, a, sessions):
    return {"question_id": qid, "question_type": qtype, "question": q, "answer": a,
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_dates": [f"2023/05/{10 + i:02d} (Wed) 09:00" for i in range(len(sessions))],
            "haystack_session_ids": [f"{qid}_s{i}" for i in range(len(sessions))],
            "haystack_sessions": [[{"role": r, "content": c} for r, c in s] for s in sessions]}


DATA = [
    item("q1", "knowledge-update", "How many pottery classes have I taken so far?", "six",
         [[("user", "I have taken four pottery classes so far."), ("assistant", "Nice.")],
          [("user", "Update: I have now taken six pottery classes, the glaze one was fun."), ("assistant", "Great.")]]),
    item("q2", "single-session-user", "What breed is my neighbour's dog?", "beagle",
         [[("user", "My neighbour's dog is a beagle named Pepper."), ("assistant", "Cute.")],
          [("user", "I bought new running shoes yesterday."), ("assistant", "Enjoy.")]]),
]


class CredsTests(unittest.TestCase):
    """_creds()：.env.jev 不存在时，环境变量里的 key 也要能读到（此前会直接 return None，导致 --vectors 静默跑成纯关键词）。"""

    def test_env_vars_used_when_dotenv_missing(self):
        import os
        from unittest import mock

        import vectors as V
        with mock.patch.dict(os.environ, {"CF_API_TOKEN": "tok-x", "CF_ACCOUNT_ID": "acct-x"}):
            with mock.patch.object(V, "ENV", Path("/nonexistent/.env.jev")):
                self.assertEqual(V._creds(), ("tok-x", "acct-x"))

    def test_none_when_nothing_anywhere(self):
        import os
        from unittest import mock

        import vectors as V
        env = {k: v for k, v in os.environ.items() if k not in ("CF_API_TOKEN", "CF_ACCOUNT_ID")}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(V, "ENV", Path("/nonexistent/.env.jev")):
                self.assertIsNone(V._creds())


class RecencyTests(unittest.TestCase):
    """新旧顶替三件套：精排窗口、分组标旧、并列时新的排前面。"""

    OLD_FITBIT = ("Cool, thanks for the suggestions! I'll definitely check them out. By the way, "
                  "I've been using my Fitbit Charge 3 for 6 months now, and it's been helping me stay on track.")
    NEW_FITBIT = ("I've already been working on establishing a consistent bedtime routine and creating a "
                  "sleep-conducive environment, but I think I need to focus on limiting stimulating activities "
                  "before bed. I've been watching TV shows before bed, and I think that might be affecting my "
                  "sleep. By the way, speaking of tracking my health, I just realized I've been using my "
                  "Fitbit Charge 3 for 9 months now - it's crazy how time flies!")

    def test_best_window_finds_buried_mention(self):
        import xrerank as X
        self.assertIn("9 months", X.best_window(self.NEW_FITBIT, "How long have I been using my Fitbit Charge 3?"))
        self.assertNotIn("9 months", self.NEW_FITBIT[:300])  # 旧做法截前 300 字会把新值截掉
        self.assertEqual(X.best_window("short doc", "q"), "short doc")

    def test_superseded_map_groups_old_new(self):
        import recall_flow as RF
        pool = [
            {"ext_id": "old", "content": self.OLD_FITBIT, "valid_from": "2023-06-18"},
            {"ext_id": "new", "content": self.NEW_FITBIT, "valid_from": "2023-09-02"},
            {"ext_id": "junk", "content": "I'm creating a role-playing game with some friends.", "valid_from": "2023-06-19"},
        ]
        self.assertEqual(RF.superseded_map(pool, "How long have I been using my Fitbit Charge 3?"),
                         {"old": "new"})

    def test_order_by_xscore_newer_wins_near_tie(self):
        import recall_flow as RF
        pool = [
            {"ext_id": "old", "content": "old", "valid_from": "2023-06-18"},
            {"ext_id": "new", "content": "new", "valid_from": "2023-09-02"},
        ]
        self.assertEqual(RF.order_by_xscore(pool, [0.9913, 0.9901]), [1, 0])  # 并列时新的在前
        self.assertEqual(RF.order_by_xscore(pool, [0.99, 0.50]), [0, 1])     # 分差大时分说了算


class StripThinkingTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "eval/longmemeval"))
        import run as R
        self.R = R

    def test_paired_think_block_removed(self):
        self.assertEqual(self.R.strip_thinking("<think>let me think...</think>six"), "six")
        self.assertEqual(self.R.strip_thinking("intro <think>hmm</think> middle <think>hmm2</think> tail"),
                         "intro  middle  tail")

    def test_lone_close_think_drops_preceding(self):
        # 只有孤立的 </think>：前面是没打 <think> 的思考，整段去掉
        self.assertEqual(self.R.strip_thinking("some reasoning here</think>the shoes are in the closet"),
                         "the shoes are in the closet")

    def test_lone_open_think_drops_following(self):
        self.assertEqual(self.R.strip_thinking("beagle<think>unclosed thinking"), "beagle")

    def test_no_markers_unchanged(self):
        self.assertEqual(self.R.strip_thinking("  six  "), "six")
        self.assertEqual(self.R.strip_thinking(""), "")
        self.assertEqual(self.R.strip_thinking("I don't know."), "I don't know.")

    def test_answer_strips_think_before_judging(self):
        # 答题模型输出混着思考标记时，answer() 返回的 pred 必须是干净的最终答案
        self.R.chat = lambda prompt: "<think>old value was 100, new value is 250</think>250"
        got = self.R.answer(DATA[0], [("2023-05-10", "six pottery classes")], mock=False)
        self.assertEqual(got, "250")
        self.assertNotIn("<think>", got)


class EvalSmokeTests(unittest.TestCase):
    def test_all_arms_mock(self):
        with tempfile.TemporaryDirectory() as d:
            data, out = Path(d) / "mini.json", Path(d) / "res.json"
            data.write_text(json.dumps(DATA), encoding="utf-8")
            p = subprocess.run([sys.executable, str(REPO / "eval/longmemeval/run.py"), "--data", str(data),
                                "-n", "2", "--seed", "1", "--arms", "legacy,hybrid,1,bm25", "--mock", "--out", str(out)],
                               capture_output=True, text=True, timeout=300, cwd=d)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            res = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(res["meta"]["seed"], 1)
            self.assertEqual(res["meta"]["answer_judge_model"], "mock")
            self.assertTrue(res["meta"]["git_commit"])
            self.assertEqual(sorted(res["meta"]["question_ids"]), ["q1", "q2"])
            self.assertIn("vector_model", res["meta"])
            self.assertIsNone(res["meta"]["vector_model"])  # 没开 --vectors 时记 None
            for arm in ("legacy", "hybrid", "1", "bm25"):
                self.assertEqual(res["summary"][arm]["total"], 2)
                self.assertTrue(all(r[arm]["n_hits"] >= 1 for r in res["results"]), arm)
                self.assertTrue(all(len(r[arm]["hits"]) == r[arm]["n_hits"] for r in res["results"]), arm)

    def test_type_filter_and_seed_are_deterministic(self):
        sys.path.insert(0, str(REPO / "eval/longmemeval"))
        import run as R
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "mini.json"
            data.write_text(json.dumps(DATA), encoding="utf-8")
            got = R.load_items(str(data), ["knowledge-update"], 5, 3)
            self.assertEqual([x["question_id"] for x in got], ["q1"])
            a = [x["question_id"] for x in R.load_items(str(data), None, 1, 42)]
            b = [x["question_id"] for x in R.load_items(str(data), None, 1, 42)]
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
