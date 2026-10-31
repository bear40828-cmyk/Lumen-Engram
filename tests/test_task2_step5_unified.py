"""任务二·步骤5：gate + classify + keys → 一次 LLM 调用（ENGRAM_UNIFIED_GATE，默认关；规则兜底）"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
import unified  # noqa: E402
from classify import classify  # noqa: E402
from gate.gate import judge  # noqa: E402
from keys import match  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

OFF = {"ENGRAM_ENV_FILE": "/nonexistent"}
ON = {**OFF, "ENGRAM_UNIFIED_GATE": "1"}
VER = [{"verification_status": "verified"}]
SAMPLES = ["小测（2026-10-20 10:00）：我考某大学，报计算机技术",
           "小测（2026-10-20 10:00）：以后一律纯文本回复，这是说好的规矩",
           "小测（2026-10-20 10:00）：嗯嗯",
           "小测（2026-10-20 10:00）：我最爱吃米线",
           "小测（2026-10-20 10:00）：今天无聊",
           "小测（2026-10-20 10:00）：😂😂"]


def reply(**kw):
    base = {"promote": True, "fact_type": "preference", "subject": "tester",
            "fact_key": None, "slots": ["偏好"], "confidence": 0.8, "quote": "我最爱吃米线"}
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def decide(self, env, content=SAMPLES[3], chat=None, rows=VER, **kw):
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(llm, "chat", chat or mock.Mock(side_effect=AssertionError("不该调 LLM"))):
            return unified.decide(self.con, subject="tester", content=content, evidence_rows=rows, **kw)


class FlagOffIsOldBehaviour(Base):
    def test_equals_rules_chain(self):
        for c in SAMPLES:
            with self.subTest(c=c):
                u = self.decide(OFF, c)
                v = judge(self.con, subject="tester", content=c, evidence_rows=VER)
                self.assertEqual((u.promote, u.reason, u.code), (v.promote, v.reason, v.code))
                if v.promote:
                    k = classify(c, "tester")
                    self.assertEqual((u.fact_type, u.type_conf, u.type_why), k)
                    self.assertEqual((u.fact_key, u.value, u.polarity, u.eligible), match(c))
                    self.assertEqual((u.importance, u.confidence), (v.importance, v.confidence))
                self.assertEqual(u.source, "rules")


class LLMPath(Base):
    def test_llm_promote_with_registry_key(self):
        c = "小测（2026-10-20 10:00）：我考某大学，报计算机技术"
        r = reply(fact_type="semantic", fact_key="education.target_school", quote="我考某大学", slots=["专名"])
        u = self.decide(ON, c, chat=mock.Mock(return_value=r))
        self.assertTrue(u.promote)
        self.assertEqual((u.source, u.fact_type, u.fact_key, u.value), ("llm", "semantic", "education.target_school", "某大学"))
        self.assertTrue(u.eligible)

    def test_invented_key_becomes_null(self):
        u = self.decide(ON, chat=mock.Mock(return_value=reply(fact_key="made.up.key")))
        self.assertTrue(u.promote)
        self.assertIsNone(u.fact_key)

    def test_bad_quote_falls_back_to_rules(self):
        u = self.decide(ON, chat=mock.Mock(return_value=reply(quote="原话里根本没有的一句话")))
        self.assertEqual(u.source, "rules")
        self.assertIn("退回规则", u.reason)

    def test_bad_schema_falls_back(self):
        for bad in ("不是 JSON", '{"promote": "yes"}', reply(fact_type="weird"), reply(subject="stranger")):
            with self.subTest(bad=bad):
                u = self.decide(ON, chat=mock.Mock(return_value=bad))
                self.assertEqual(u.source, "rules")

    def test_llm_error_falls_back(self):
        u = self.decide(ON, chat=mock.Mock(side_effect=llm.LLMNotConfigured("x")))
        self.assertEqual(u.source, "rules")
        self.assertIn("LLM 不可用", u.reason)
        v = judge(self.con, subject="tester", content=SAMPLES[3], evidence_rows=VER)
        self.assertEqual(u.promote, v.promote)

    def test_llm_says_no(self):
        u = self.decide(ON, chat=mock.Mock(return_value='{"promote": false}'))
        self.assertFalse(u.promote)
        self.assertEqual(u.code, "llm_no")

    def test_slots_filtered_to_known_names(self):
        u = self.decide(ON, chat=mock.Mock(return_value=reply(slots=["偏好", "乱编的槽"])))
        self.assertEqual(u.slots, ("偏好",))


class Guardrails(Base):
    """护栏不让 LLM 参与：LLM 不被调用，结论跟规则一致"""
    def test_no_evidence(self):
        self.assertFalse(self.decide(ON, rows=[]).promote)

    def test_unverified_evidence(self):
        u = self.decide(ON, rows=[{"verification_status": "uncertain"}])
        self.assertFalse(u.promote)
        self.assertEqual(u.code, "evidence_unverified")

    def test_emoji_only(self):
        self.assertEqual(self.decide(ON, SAMPLES[5]).code, "no_text")

    def test_duplicate(self):
        c = SAMPLES[3]
        promote(self.con, subject="tester", fact_type="preference", content=c, importance=0.5, confidence=0.7)
        self.assertEqual(self.decide(ON, c).code, "duplicate")

    def test_pinned_goes_through_rules(self):
        u = self.decide(ON, SAMPLES[2], user_pinned=True)
        self.assertTrue(u.promote)
        self.assertEqual(u.code, "pinned")


class FromChannelEndToEnd(unittest.TestCase):
    def run_fc(self, env, chat=None):
        from extract import from_channel as fc
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            rec = {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "mcp__telegram__reply",
                 "input": {"text": "以后一律纯文本回复，不用 markdown，这是我跟小测说好的规矩。"}}]}}
            p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            con = connect(Path(d) / "t.db")
            orig_connect, orig_argv = fc.connect, sys.argv
            fc.connect = lambda *a, **k: con
            try:
                sys.argv = ["from_channel.py", str(p), "--promote-claude"]
                with mock.patch.dict(os.environ, env, clear=True), \
                        mock.patch.object(llm, "chat", chat or mock.Mock(side_effect=AssertionError("不该调"))):
                    fc.main()
            finally:
                fc.connect, sys.argv = orig_connect, orig_argv
            row = con.execute("""SELECT f.subject, f.fact_type, c.reason FROM facts f
                                 LEFT JOIN fact_class c ON c.fact_id = f.id""").fetchall()
            con.close()
            return [tuple(r) for r in row]

    def test_flag_off_rules(self):
        rows = self.run_fc(OFF)
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0][2], "unified-llm-v1")

    def test_flag_on_llm(self):
        r = json.dumps({"promote": True, "fact_type": "procedural", "subject": "claude", "fact_key": None,
                        "slots": ["规矩"], "confidence": 0.9, "quote": "以后一律纯文本回复"}, ensure_ascii=False)
        rows = self.run_fc(ON, chat=mock.Mock(return_value=r))
        self.assertEqual(rows, [("claude", "procedural", "unified-llm-v1")])


if __name__ == "__main__":
    unittest.main()
