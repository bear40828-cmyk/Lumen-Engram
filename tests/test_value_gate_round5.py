"""值得记闸第五轮（2026-10-28，用户：委托式回复「算」同意）：(a) 路（助手提、她随后确认）和重归属路共用一个同意判定
VG.user_consents：明确同意（好/行/可以/就这样…）和委托式（都行/看你/听你的/你定/随你/你说了算/无所谓…）都算；
带「但/不过/改成」、问句、拒绝（不要/别/不行）的不算。窗口/紧随限制不变。"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import value_gate as VG  # noqa: E402
from extract import from_evidence as FE  # noqa: E402
from tests.test_value_gate_round4 import ev  # noqa: E402

SUG = "我建议以后账本一律按你的日期来记，毕竟是我俩的账本，不是系统的"
NEG = ["看你吧但我觉得还是系统的好", "随便你？", "不要"]


def reattr_fact():
    return {"keep": True, "category": "decision", "subject": "tester", "type": "procedural",
            "content": "小测决定以后账本一律按她的日期来记录", "quote": "以后账本一律按你的日期来记", "src": 0}


def a_fact(reply):
    return {"keep": True, "category": "agreement", "subject": "us", "type": "procedural",
            "content": "小测和助理甲约定以后账本一律按小测的日期来记", "quote": "以后账本一律按你的日期来记", "src": 0,
            "quote2": reply, "src2": 1}


class SharedConsent(unittest.TestCase):
    def setUp(self):
        VG.REJECTS.clear()
        VG.REATTRIBUTED.clear()
        p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"})
        p.start()
        self.addCleanup(p.stop)

    def test_detector(self):
        for ok in ["都行，看你", "好呀，就这样", "行", "可以", "听你的", "你定", "随你", "你说了算", "无所谓", "都可以", "好，不过改成周三"]:
            # 第六轮：只看第一个分句，「好」后面跟什么都算同意
            self.assertTrue(VG.user_consents(ok), ok)
        for bad in NEG + ["别这样", "不行", "不过改成周三", "改成明天吧", "你定吗"]:
            self.assertFalse(VG.user_consents(bad), bad)
        self.assertIs(VG.user_agrees, VG.user_consents)

    def test_reattribution_accepts_delegating(self):
        ok, why = FE.check(reattr_fact(), [ev(0, "claude", SUG), ev(1, "tester", "都行，看你")])
        self.assertIsNotNone(ok, why)
        self.assertEqual((ok["subject"], ok["category"]), ("us", "agreement"))

    def test_reattribution_negatives(self):
        for bad in NEG:
            ok, why = FE.check(reattr_fact(), [ev(0, "claude", SUG), ev(1, "tester", bad)])
            self.assertIsNone(ok, bad)

    def test_path_a_accepts_delegating(self):
        ok, why = FE.check(a_fact("都行，看你"), [ev(0, "claude", SUG), ev(1, "tester", "都行，看你")])
        self.assertIsNotNone(ok, why)
        self.assertEqual(ok["subject"], "us")

    def test_path_a_negatives(self):
        for bad in NEG:
            ok, why = FE.check(a_fact(bad), [ev(0, "claude", SUG), ev(1, "tester", bad)])
            self.assertIsNone(ok, bad)


if __name__ == "__main__":
    unittest.main()
