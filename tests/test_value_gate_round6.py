"""值得记闸第六轮（2026-10-28，用户：最后一次微调，之后冻结）：同意判定只看她那句话的「第一个分句」。
第一个分句以明确同意（好/行/可以/就这样/嗯嗯/OK…）开头，或是委托式回复（都行/看你/随你/听你的/你定/你说了算/无所谓…），
不管后面跟什么都算；她自己的拍板口吻（那先…吧/那就…吧/先…吧）在和事实对得上时也算；
开头是拒绝（不要/别/不行/不可以/不同意）、第一个分句本身是问句或带「但/不过/改成」的照拒。
夹具全是编的句子，不含真实聊天原文。"""
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

FACT_TXT = "小测和助理甲约定先不买新键盘，攒到月底再看"


class FirstClause(unittest.TestCase):
    def test_split_points(self):
        self.assertEqual(VG.first_clause("行👀 我不是有别的吗")[0], "行")
        self.assertEqual(VG.first_clause("可以🌚\n话说那个能不能换")[0], "可以")
        self.assertEqual(VG.first_clause("好！明天再说…")[0], "好")
        self.assertEqual(VG.first_clause("嗯 我想想")[0], "嗯")
        self.assertEqual(VG.first_clause("小测（2026-01-01 02:00）：行，没问题")[0], "行")
        self.assertEqual(VG.first_clause("好？"), ("好", "？"))

    def test_agreement_regardless_of_what_follows(self):
        for line in ["行👀 我不是还有一个吗，我在想能不能拿去改",
                     "可以🌚\n话说那个图标能不能换一个，好丑吗",
                     "好，不过我还有个问题你怎么看？",
                     "嗯嗯，" + "后面是很长的一段话" * 20,
                     "OK 但是我晚点才有空", "对，" + "我补充一下" * 5, "就这样吧，对了你吃饭了吗"]:
            self.assertTrue(VG.user_consents(line), line)

    def test_delegating_regardless_of_what_follows(self):
        for line in ["都行，另外你今天累不累？", "看你👀 我都可以", "随你，你说了算", "你定 我先去洗澡了", "无所谓，你高兴就好",
                     "听你的，" + "长" * 100]:
            self.assertTrue(VG.user_consents(line), line)

    def test_refusal_or_question_first_clause_rejects(self):
        for line in ["不要，还是旧的好", "别，我再想想", "不行", "不可以这样", "不同意", "好？", "行吗，我有点担心", "可以吗？",
                     "你定吗，我不确定", "但是我想改成周五", "看你吧但我觉得不太行", "为什么要这样，好吧", "对不起，我刚才没看到",
                     "其实是十点才吃的早饭🤣", "对于这件事我有点不一样的看法", "那就是说你还没做？", "我看到的是星号👀", ""]:
            self.assertFalse(VG.user_consents(line), line)

    def test_decision_phrasing_needs_fact_match(self):
        for line in ["那先不买吧，攒着再看看", "那就先不买新键盘吧", "先不买吧🤡 月底再说"]:
            self.assertTrue(VG.user_consents(line, FACT_TXT), line)
            self.assertFalse(VG.user_consents(line), line)                      # 没有事实正文：拍板口吻不算
        for line in ["那先去洗澡吧，一会儿聊", "先吃饭吧", "那先不聊这个吧"]:       # 和事实对不上
            self.assertFalse(VG.user_consents(line, FACT_TXT), line)
        self.assertFalse(VG.user_consents("那先不买吧？", FACT_TXT))              # 第一个分句是问句
        self.assertFalse(VG.user_consents("那先不买吗，我不确定", FACT_TXT))


def fact_a(quote2):
    return {"keep": True, "category": "agreement", "subject": "us", "type": "procedural",
            "content": FACT_TXT, "quote": "先不买新键盘，攒到月底再看", "src": 0, "quote2": quote2, "src2": 1}


def fact_reattr():
    return {"keep": True, "category": "decision", "subject": "tester", "type": "procedural",
            "content": FACT_TXT, "quote": "先不买新键盘，攒到月底再看", "src": 0}


SUG = "我建议先不买新键盘，攒到月底再看，那时候会有活动价"


class Wired(unittest.TestCase):
    def setUp(self):
        VG.REJECTS.clear()
        VG.REATTRIBUTED.clear()
        p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"})
        p.start()
        self.addCleanup(p.stop)

    def test_path_a_recovers(self):
        for reply in ["行👀 我最近有个别的想法", "可以🌚\n话说那个能不能换", "那先不买吧，月底看看", "都行，你定"]:
            ok, why = FE.check(fact_a(reply), [ev(0, "claude", SUG), ev(1, "tester", reply)])
            self.assertIsNotNone(ok, f"{reply}: {why}")
            self.assertEqual(ok["subject"], "us")

    def test_path_a_still_blocked(self):
        for reply in ["其实我最近在看别的🤣", "不要，我想现在买", "你确定吗？", "那先去吃饭吧"]:
            ok, why = FE.check(fact_a(reply), [ev(0, "claude", SUG), ev(1, "tester", reply)])
            self.assertIsNone(ok, reply)

    def test_reattribution_uses_first_clause_and_fact(self):
        for reply in ["行👀 我不是有别的吗", "那先不买吧，月底看看"]:
            ok, why = FE.check(fact_reattr(), [ev(0, "claude", SUG), ev(1, "tester", reply)])
            self.assertIsNotNone(ok, f"{reply}: {why}")
            self.assertEqual((ok["subject"], ok["category"]), ("us", "agreement"))
        for reply in ["不要，我想现在买", "其实我最近在看别的🤣", "你确定吗？"]:
            ok, why = FE.check(fact_reattr(), [ev(0, "claude", SUG), ev(1, "tester", reply)])
            self.assertIsNone(ok, reply)


if __name__ == "__main__":
    unittest.main()
