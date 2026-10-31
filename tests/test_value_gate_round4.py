"""值得记闸第四轮（2026-10-28，用户同意「行，那你放吧」）：放宽 us 的成立条件。
us = 两方在窗口里各说过话：(a) 助手提、她随后确认；(b) 她先提要求/定规矩、助手随后明确答应（事实实质得出自她的话）。
另：模型把助手的建议写成她的决定、她随后明确同意 → 改记 us；没有她的同意仍被回声挡。闸关一字不差。"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import value_gate as VG  # noqa: E402
from extract import from_evidence as FE  # noqa: E402


def ev(i, actor, text):
    who = {"tester": "小测", "claude": "我", "gpt": "助手乙", "agentc": "助手丙"}[actor]
    return {"id": i, "source_actor": actor, "occurred_at": f"2026-10-01T00:{i:02d}:00Z", "path": "p", "line": f"L{i}",
            "content": f"{who}（2026-10-01 10:{i:02d}）：{text}"}


HER = "我想要你不催我背单词，我想背到几点就背到几点"
CLAUDE_OK = "好，不催了，你想背到几点就背到几点，我陪着"
CONTENT = "小测和助理甲约定：助理甲不再催小测背单词"


def us_fact(**kw):
    f = {"keep": True, "category": "agreement", "subject": "us", "type": "procedural", "content": CONTENT,
         "quote": "不催了，你想背到几点就背到几点，我陪着", "src": 2, "quote2": "我想要你不催我背单词", "src2": 1}
    f.update(kw)
    return f


def chunk_b(her=HER, claude=CLAUDE_OK, her_actor="tester", claude_actor="claude"):
    return [ev(0, "claude", "这么晚了你还没歇呀"), ev(1, her_actor, her), ev(2, claude_actor, claude)]


class UsRelaxedB(unittest.TestCase):
    def setUp(self):
        VG.REJECTS.clear()
        VG.REATTRIBUTED.clear()
        p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"})
        p.start()
        self.addCleanup(p.stop)

    def test_positive_she_requests_first_claude_accepts(self):
        ok, why = FE.check(us_fact(), chunk_b())
        self.assertIsNotNone(ok, why)
        self.assertEqual((ok["subject"], ok["category"], ok["confidence"]), ("us", "agreement", 0.6))
        self.assertEqual({ok["hit"]["source_actor"], ok["hit2"]["source_actor"]}, {"claude", "tester"})

    def test_positive_model_wording_with_pronouns_and_extra_clause(self):
        """真模型试跑里的写法：第三人称、换了人称，后半句带着助手答应的那部分。"""
        f = us_fact(content="测试姓名要求助理甲不要催她背单词，助理甲承诺不再催促并会陪伴她。", quote="我想要你不催我背单词", src=1,
                    quote2="不催了，你想背到几点就背到几点，我陪着", src2=2)
        f["quote"], f["quote2"] = "想要你不催我背单词", "不催了，你想背到几点就背到几点，我陪着"
        ok, why = FE.check(f, chunk_b())
        self.assertIsNotNone(ok, why)

    def test_positive_quote_is_her_line(self):
        f = us_fact(quote="我想要你不催我背单词", src=1, quote2="不催了，你想背到几点就背到几点，我陪着", src2=2)
        ok, why = FE.check(f, chunk_b())
        self.assertIsNotNone(ok, why)

    def test_still_ok_claude_proposes_she_confirms(self):
        chunk = [ev(0, "claude", "以后改配置之前我一定先问你一声"), ev(1, "tester", "好，就这么定了，你说到要做到")]
        f = us_fact(content="小测和助理甲约定改配置之前先问小测", quote="以后改配置之前我一定先问你一声", src=0,
                    quote2="好，就这么定了，你说到要做到", src2=1)
        ok, why = FE.check(f, chunk)
        self.assertIsNotNone(ok, why)

    def test_negative_claude_only(self):
        chunk = [ev(0, "claude", CLAUDE_OK)]
        f = us_fact(src=0, quote2="", src2=None)
        ok, why = FE.check(f, chunk)
        self.assertIsNone(ok)
        self.assertIn("subject-us", why)
        ok, why = FE.check({**f, "content": "助理甲答应不再催小测背单词"}, chunk)       # 落回 claude 也不是 agreement
        self.assertIsNone(ok)
        self.assertIn("role:agreement-not-us", why)

    def test_negative_her_line_is_joke(self):
        ok, why = FE.check(us_fact(quote2="我要你以后不催我背单词"), chunk_b(her="我要你以后不催我背单词，开玩笑的🤡"))
        self.assertIsNone(ok)
        self.assertIn("role:us-her-line-joke", why)

    def test_negative_her_line_is_question(self):
        ok, why = FE.check(us_fact(quote2="你是不是又要催我背单词"), chunk_b(her="你是不是又要催我背单词了？"))
        self.assertIsNone(ok)
        self.assertIn("role:us-her-line-question", why)

    def test_negative_gpt_as_claude_side(self):
        ok, why = FE.check(us_fact(), chunk_b(claude_actor="gpt"))
        self.assertIsNone(ok)
        self.assertTrue("role:us-no-user-confirmation" in why or "role:agreement-needs-claude" in why, why)

    def test_negative_gpt_as_her_side(self):
        ok, why = FE.check(us_fact(), chunk_b(her_actor="gpt"))
        self.assertIsNone(ok)
        self.assertTrue("subject-us" in why or "role:" in why, why)

    def test_negative_her_request_without_acceptance(self):
        ok, why = FE.check(us_fact(quote="那你也早点歇吧，别熬太晚"), chunk_b(claude="那你也早点歇吧，别熬太晚"))
        self.assertIsNone(ok)
        self.assertIn("role:us-no-explicit-accept", why)
        # 她一个人的请求：只能记成她自己的偏好（subject=user），不是 us
        chunk = [ev(0, "tester", HER)]
        ok, why = FE.check(us_fact(src=0, quote="我想要你不催我背单词", quote2="", src2=None), chunk)
        self.assertIsNone(ok)
        self.assertTrue("subject-us" in why or "role:agreement-not-us" in why, why)
        pref = {"keep": True, "category": "preference", "subject": "tester", "type": "preference",
                "content": "小测想要助理甲不催她背单词", "quote": "我想要你不催我背单词", "src": 0}
        ok, why = FE.check(pref, chunk)
        self.assertIsNotNone(ok, why)

    def test_negative_claude_hedges(self):
        ok, why = FE.check(us_fact(quote="我尽量吧，可能做不到", quote2="我想要你不催我背单词"),
                           chunk_b(claude="我尽量吧，可能做不到"))
        self.assertIsNone(ok)
        self.assertIn("role:us-no-explicit-accept", why)

    def test_negative_echo_substance_only_from_claude(self):
        """防回声：她只说了「你以后别这样了」，助手的答应里带出的新内容不能记成约定。"""
        chunk = [ev(0, "claude", "我刚才又提醒你十二点前关电脑了"), ev(1, "tester", "你以后别这样了"),
                 ev(2, "claude", "好，我保证每天十二点前提醒你关电脑，一个字都不会少")]
        f = us_fact(content="助理甲保证每天十二点前提醒小测关电脑", quote="我保证每天十二点前提醒你关电脑", src=2,
                    quote2="你以后别这样了", src2=1)
        ok, why = FE.check(f, chunk)
        self.assertIsNone(ok)
        self.assertIn("role:us-not-grounded-in-user-words", why)

    def test_negative_claude_line_is_english_thinking(self):
        chunk = [ev(0, "tester", HER), ev(1, "claude", "She wants me to stop nagging her about vocabulary. I should agree and promise it.")]
        f = us_fact(quote="I should agree and promise it", src=1, quote2="我想要你不催我背单词", src2=0)
        ok, why = FE.check(f, chunk)
        self.assertIsNone(ok)

    def test_negative_acceptance_too_far(self):
        chunk = [ev(0, "tester", HER)] + [ev(i, "claude", f"这是第{i}条无关的工具说明，正在整理文件") for i in range(1, 10)] + [ev(10, "claude", CLAUDE_OK)]
        ok, why = FE.check(us_fact(src=10, src2=0, quote2="我想要你不催我背单词"), chunk)
        self.assertIsNone(ok)
        self.assertIn("role:us-too-far", why)

    def test_third_party_cannot_be_the_party_in_helper(self):
        self.assertEqual(VG.role_check("us", "agreement", {"tester", "gpt"}, confirmed_by_user_after=True), "role:agreement-needs-claude")
        self.assertEqual(VG.role_check("us", "agreement", {"tester", "agentc"}, confirmed_by_user_after=True), "role:agreement-needs-claude")
        self.assertEqual(VG.role_check("us", "agreement", {"tester", "claude"}, confirmed_by_user_after=False,
                                       us_reject="role:us-her-line-joke"), "role:us-her-line-joke")
        self.assertIsNone(VG.role_check("us", "agreement", {"tester", "claude"}, confirmed_by_user_after=True))

    def test_helpers(self):
        self.assertIsNone(VG.user_request_line("小测（2026-10-01 10:00）：" + HER))
        self.assertIsNone(VG.user_request_line("能不能别催我背单词？"))
        self.assertEqual(VG.user_request_line("你怎么又催我背单词"), "role:us-her-line-question")
        self.assertEqual(VG.user_request_line("今天天气不错"), "role:us-her-line-not-request")
        self.assertIsNone(VG.user_request_line("你别催我背单词啦🤡"))             # 🤡 她天天用，不算玩笑
        self.assertEqual(VG.user_request_line("你别催我背单词啦，开玩笑的"), "role:us-her-line-joke")
        self.assertIsNone(VG.claude_accepts(CLAUDE_OK))
        self.assertIsNone(VG.claude_accepts("好的，以后我都不催你"))
        self.assertIsNotNone(VG.claude_accepts("那你说说为什么？"))
        self.assertTrue(VG.us_user_grounded(CONTENT, HER))
        self.assertFalse(VG.us_user_grounded("助理甲保证每天提醒小测喝水", HER))


class ReattributeToUs(unittest.TestCase):
    """12495 型：模型把助手的建议记成了她的决定。"""
    SUG = "我建议以后账本一律按你的日期来记，毕竟是我俩的账本，不是系统的"

    def setUp(self):
        VG.REJECTS.clear()
        VG.REATTRIBUTED.clear()
        p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"})
        p.start()
        self.addCleanup(p.stop)

    def fact(self):
        return {"keep": True, "category": "decision", "subject": "tester", "type": "procedural",
                "content": "小测决定以后账本一律按她的日期来记录", "quote": "以后账本一律按你的日期来记", "src": 0}

    def test_agreement_in_window_becomes_us(self):
        chunk = [ev(0, "claude", self.SUG), ev(1, "tester", "好呀，就这样")]
        ok, why = FE.check(self.fact(), chunk)
        self.assertIsNotNone(ok, why)
        self.assertEqual((ok["subject"], ok["category"], ok["reattributed"]), ("us", "agreement", True))
        self.assertEqual(ok["hit2"]["source_actor"], "tester")
        self.assertEqual(VG.REATTRIBUTED["us:reattributed"], 1)

    def test_no_agreement_still_blocked_as_echo(self):
        for tail in ([], [ev(1, "tester", "不要，还是按系统的吧")], [ev(1, "tester", "你确定吗？")],
                     [ev(1, "tester", "但是我想改成农历日期")], [ev(1, "gpt", "好呀，就这样")]):
            ok, why = FE.check(self.fact(), [ev(0, "claude", self.SUG)] + tail)
            self.assertIsNone(ok, tail)
            self.assertTrue("subject-echo" in why or "role:" in why, why)
        self.assertEqual(VG.REATTRIBUTED["us:reattributed"], 0)

    def test_agreement_must_follow_closely(self):
        chunk = [ev(0, "claude", self.SUG)] + [ev(i, "claude", "我在整理别的东西，稍等一下下") for i in range(1, 6)] + [ev(6, "tester", "好呀，就这样")]
        ok, why = FE.check(self.fact(), chunk)
        self.assertIsNone(ok)
        # 她先说了别的话，再说「好」：不算对这条建议的同意
        chunk = [ev(0, "claude", self.SUG), ev(1, "tester", "今天好累啊，不想动了"), ev(2, "tester", "好呀，就这样")]
        ok, why = FE.check(self.fact(), chunk)
        self.assertIsNone(ok)

    def test_her_own_line_unaffected(self):
        chunk = [ev(0, "tester", "以后账本一律按我的日期来记，别用系统的")]
        f = {**self.fact(), "quote": "以后账本一律按我的日期来记", "content": "小测要求以后账本一律按她的日期来记录"}
        ok, why = FE.check(f, chunk)
        self.assertIsNotNone(ok, why)
        self.assertEqual(ok["subject"], "tester")
        self.assertNotIn("reattributed", ok)

    def test_gate_off_unchanged(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            chunk = [ev(0, "claude", self.SUG), ev(1, "tester", "好呀，就这样")]
            ok, why = FE.check(self.fact(), chunk)
            self.assertIsNone(ok)
            self.assertIn("subject-echo", why)


if __name__ == "__main__":
    unittest.main()
