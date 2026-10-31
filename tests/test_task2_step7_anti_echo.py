"""task2-07：蒸馏防回声——助手原话蒸馏出的事，subject 只能是 claude / us（模型调用全 mock）"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import distill_self as D  # noqa: E402
import llm  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

LONG = "我把检索脚本改成了混合召回，还把备用模型的配置写进了配置文件，下一步要去跑一遍测试，看看报错日志里有没有新的问题，" \
       "然后再决定要不要把旧的分层路径删掉，现在的结论是先留着开关，等她点头。她说她这周都在准备考试。"


def fact(subject, quote="先留着开关，等她点头", content="助理甲把旧分层路径留在开关后面，等她点头再决定"):
    return {"subject": subject, "type": "procedural", "content": content, "quote": quote, "confidence": 0.9}


class AntiEcho(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        ev = add_evidence(self.con, source_kind="chat", source_actor="claude", source_path="/tmp/x.jsonl",
                          source_locator="L1", content=LONG, verification_status="verified")
        self.ev = ev
        promote(self.con, subject="claude", fact_type="episodic", content="我（2026-10-01 10:00）：" + LONG,
                evidence_ids=[ev], valid_from="2026-10-01T02:00:00+00:00", importance=0.5, confidence=0.7)
        self.con.commit()
        self.row = D.load_pool(self.con)[0]

    def chat(self, facts):
        return mock.patch.object(llm, "chat_ex", return_value=llm.LLMResult(
            json.dumps({"facts": facts}, ensure_ascii=False), "primary", "m",
            {"prompt_tokens": 10, "completion_tokens": 5}, ["primary:ok"]))

    def test_prompt_says_only_claude(self):
        # task2-09：助手单方的话产不出 us（需要用户自己也有原话），提示词随之收紧
        self.assertIn("只能写 claude", D.PROMPT)
        self.assertIn("绝对不要写 tester", D.PROMPT)
        self.assertIn("us", D.PROMPT)

    def test_subject_ok(self):
        self.assertTrue(D.subject_ok("claude")[0])
        for s in ("us", "tester", "gpt", "agentc", "", None, "project:x"):
            ok, why = D.subject_ok(s)
            self.assertFalse(ok)
            self.assertTrue(why.startswith("subject-echo:"))

    def test_distill_one_rejects_other_subjects_and_logs_reason(self):
        with self.chat([fact("tester"), fact("gpt"), fact("agentc"), fact("claude"), fact("us")]):
            r = D.distill_one(self.row)
        self.assertEqual(sorted(k["subject"] for k in r["kept"]), [])   # 只取前 2 条：user、gpt 都被拦
        self.assertEqual(len(r["dropped"]), 2)
        self.assertTrue(all(str(d).startswith("subject-echo") for d in r["dropped"]))   # check() 与 subject_ok 两处，原因前缀同为 subject-echo
        # task2-09：单方证据的 us——内容以「助手」为主语 → 落回 claude；否则丢弃
        with self.chat([fact("claude"), fact("us")]):
            r = D.distill_one(self.row)
        self.assertEqual(sorted(k["subject"] for k in r["kept"]), ["claude", "claude"])
        self.assertEqual(r["dropped"], [])
        with self.chat([fact("us", content="小测和助理甲一起决定先留着开关")]):
            r = D.distill_one(self.row)
        self.assertEqual(r["kept"], [])
        self.assertTrue(str(r["dropped"][0]).startswith("subject-us"))

    def test_apply_never_writes_tester_even_if_kept_is_forged(self):
        bad = {"subject": "tester", "type": "preference", "content": "小测这周在准备考试", "quote": "她这周都在准备考试",
               "evidence_id": self.ev, "occurred_at": "2026-10-01T02:00:00+00:00", "confidence": 0.9}
        good = dict(bad, subject="claude", content="助理甲先留着开关")
        forged_us = dict(bad, subject="us", content="助理甲和小测商量先留着开关")
        res = [{"fact_id": self.row["id"], "cat": "K7", "route": "primary", "model": "m", "usage": {},
                "dropped": [], "kept": [bad, forged_us, good]}]
        stat = D.apply(self.con, [self.row], res, "BTEST-E1", Path(self.dir.name))
        self.assertEqual(stat["rejected_subject"], 2)
        self.assertEqual(stat["new_facts"], 1)
        subs = [x[0] for x in self.con.execute(
            "SELECT f.subject FROM facts f JOIN fact_class c ON c.fact_id=f.id WHERE c.reason=?", (D.EXTRACTOR,))]
        self.assertEqual(subs, ["claude"])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE subject='tester'").fetchone()[0], 0)



def setUpModule():
    # 这个文件测的是旧管线（值得记闸关）的行为：不管环境变量 ENGRAM_VALUE_GATE 开没开，都按闸关跑（闸开的行为在 test_value_gate*.py 里测）
    import os
    from unittest import mock
    p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "0"})
    p.start()
    unittest.addModuleCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
