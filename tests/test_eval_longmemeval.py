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
            for arm in ("legacy", "hybrid", "1", "bm25"):
                self.assertEqual(res["summary"][arm]["total"], 2)
                self.assertTrue(all(r[arm]["n_hits"] >= 1 for r in res["results"]), arm)

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
