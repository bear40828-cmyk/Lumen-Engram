"""值得记闸第三轮：说话人标注 overlay、转述词表、agreement 角色限制、引文规范化/限长、
英文思考文本、召回补偿（旧候选并集）。除说明外都在闸开（ENGRAM_VALUE_GATE=1）下测；闸关的行为见 test_value_gate.py。"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import actor_fix  # noqa: E402
import echo  # noqa: E402
import flags  # noqa: E402
import speakers  # noqa: E402
import value_gate as VG  # noqa: E402
from extract import collect, distill_slice as DS, from_channel as FC, from_evidence as FE  # noqa: E402
from store.store import add_evidence, connect  # noqa: E402

GPT_BLOCK = ('<channel source="plugin:telegram:telegram" chat_id="-100123" message_id="12" user="AgentB_bot" '
             'user_id="1000000003" ts="2026-08-02T14:09:30.000Z">\n周报草稿写好了，直接复制：第 4 题：我想和队友一起完成这个小项目\n</channel>')
HER_BLOCK = ('<channel source="plugin:telegram:telegram" chat_id="1000000001" message_id="37" user="1000000001" '
             'user_id="1000000001" ts="2026-08-16T15:56:32.000Z">\n今天把周报整理完了，明天开始写测试\n</channel>')
HER_BLOCK2 = HER_BLOCK.replace("今天把周报整理完了，明天开始写测试", "这个周末想去海边走走，看看日出")


def _jsonl(path: Path, recs: list[dict]):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")


def _user(text, ts="2026-10-28T01:00:00Z"):
    return {"type": "user", "message": {"role": "user", "content": text}, "timestamp": ts}


class Retelling(unittest.TestCase):
    def t(self, content, ev, quote, subject="tester", category="preference"):
        return VG.postcheck({"subject": subject, "category": category, "content": content, "quote": quote}, ev, actor="tester")

    def test_relatives_said(self):
        ev = "小测（2026-10-01）：后来我表姐说水杯用一两年就好换新的，也没必要买贵的"
        self.assertEqual(self.t("小测认为水杯用一两年就应该换新的", ev, "水杯用一两年就好换新的"), "post:retelling")
        self.assertEqual(self.t("小测的表姐说水杯要一两年换一次", "x" + ev, "水杯用一两年就好换新的"), "post:retelling")

    def test_many_sources_and_verbs(self):
        for who in ("我妈", "妈妈", "我爸", "同事", "同学", "朋友", "老师", "我姐", "同事"):
            for verb in ("说", "觉得", "告诉我", "跟我说"):
                ev = f"小测（时间）：{who}{verb}应该早点把毕业设计的开题报告交上去才行"
                self.assertEqual(self.t("小测认为应该早点交开题报告", ev, "应该早点把毕业设计的开题报告交上去才行"),
                                 "post:retelling", (who, verb))

    def test_verbatim_line_with_embedded_report_is_kept_but_leading_report_is_not(self):
        ev = "小测（2026-09-09 13:27）：他想让我报某工科学院来着，可我就是只想学烘焙 他听完说那就随你吧（我同学说我爱折腾），主要是不想当会计"
        f = {"subject": "tester", "category": "decision", "content": ev, "quote": "可我就是只想学烘焙"}
        self.assertIsNone(VG.postcheck(f, ev, actor="tester"))
        ev2 = "小测（2026-09-09 13:27）：我表姐说今年夏天冷饮摊一天能赚好几千，我也不知道真假"
        f2 = {"subject": "tester", "category": "life_change", "content": ev2, "quote": "今年夏天冷饮摊一天能赚好几千"}
        self.assertEqual(VG.postcheck(f2, ev2, actor="tester"), "post:retelling")

    def test_own_stance_beats_content_level_retelling(self):
        ev = "小测（2026-10-06 03:20）：我叔叔说存款放着不动会缩水，我觉得他说得对，我打算学他的做法"
        f = {"subject": "tester", "category": "decision", "content": "小测说她叔叔认为存款放着不动会缩水，她表示认同，准备照她叔叔的办法做", "quote": "我打算学他的做法"}
        self.assertIsNone(VG.postcheck(f, ev, actor="tester"))

    def test_her_own_words_pass(self):
        ev = "小测（时间）：我觉得水杯用一两年就好换新的，也没必要买贵的"
        self.assertIsNone(self.t("小测认为水杯用一两年就应该换新的", ev, "水杯用一两年就好换新的"))
        ev2 = "小测（时间）：我跟朋友说我不想去了，太累了真的"            # 她自己对别人说，不是转述
        self.assertIsNone(self.t("小测不想去，觉得太累", ev2, "我不想去了，太累了真的", category="decision"))
        self.assertFalse(VG.reported_speech("小测希望助理甲像老师说话那样耐心"))

    def test_claude_subject_unaffected(self):
        ev = "助理甲（时间）：朋友说的那个办法我试过了，不行，我错了以后先验证"
        self.assertIsNone(VG.postcheck({"subject": "claude", "category": "self_correction", "content": "助理甲承认没验证就用了朋友的办法，以后先验证",
                                        "quote": "我错了以后先验证"}, ev, actor="claude"))


class AgreementRoles(unittest.TestCase):
    def test_agreement_only_for_us(self):
        for subj in ("gpt", "agentc", "claude", "tester"):
            self.assertEqual(VG.role_check(subj, "agreement", {subj, "tester"}), "role:agreement-not-us", subj)
        self.assertIsNone(VG.role_check("us", "agreement", {"tester", "claude"}))

    def chunk(self):
        return [{"id": 1, "source_actor": "agentc", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
                 "content": "助手丙（2026-10-01 10:00）：资料的事先放在文档里，群里不转发不截图不外传。"},
                {"id": 2, "source_actor": "tester", "occurred_at": "2026-10-01T00:01:00Z", "path": "p", "line": "L2",
                 "content": "小测（2026-10-01 10:01）：好，就这么办，记一下"},
                {"id": 3, "source_actor": "claude", "occurred_at": "2026-10-01T00:02:00Z", "path": "p", "line": "L3",
                 "content": "助理甲（2026-10-01 10:02）：好，资料的事先放在文档里，我也这么做。"}]

    def test_agentc_agreement_not_recorded_as_agentc_or_user(self):
        base = {"keep": True, "category": "agreement", "type": "procedural", "content": "助手丙与小测约定资料内容不在群里转发",
                "quote": "资料的事先放在文档里，群里不转发不截图不外传", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            for subj in ("agentc", "tester"):
                ok, why = FE.check({**base, "subject": subj}, self.chunk())
                self.assertIsNone(ok, (subj, why))
            ok, why = FE.check({**base, "subject": "us", "content": "小测和助手丙约定资料内容不在群里转发",
                                "quote2": "好，就这么办，记一下", "src2": 1}, self.chunk())
            self.assertIsNone(ok, why)                  # GLM 助手+用户两方原话也不行：约定只认用户和助手之间
            self.assertIn("agreement-needs-claude", why)

    def test_quote2_must_be_other_party_other_line(self):
        ch = self.chunk()
        f = {"keep": True, "category": "agreement", "subject": "us", "type": "procedural", "content": "小测和助理甲约定资料的事先放在文档里",
             "quote": "资料的事先放在文档里", "src": 2, "quote2": "好，就这么办，记一下", "src2": 1}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            hit = ch[2]
            self.assertIs(FE._second_hit(f, ch, hit), ch[1])                                  # 她另一行：成立
            same_line = {**f, "quote2": "资料的事先放在文档里，我也这么做", "src2": 2}
            self.assertIsNone(FE._second_hit(same_line, ch, hit))                             # 和第一句同一行：不行
            ch2 = ch + [{"id": 4, "source_actor": "tester", "occurred_at": "2026-10-01T00:03:00Z", "path": "p", "line": "L4",
                         "content": "小测（2026-10-01 10:03）：对，我也这么想，就这样吧"}]
            same_actor = {**f, "quote2": "对，我也这么想，就这样吧", "src2": 3}
            self.assertIsNone(FE._second_hit(same_actor, ch2, ch2[1]))                        # hit 是她、quote2 也是她另一行：不行（要另一方）
            dup_id = [{**ch[0]}, {**ch[1], "id": 1}]                                          # 同一个证据 id 的两段：不算两行
            self.assertIsNone(FE._second_hit({**f, "quote2": "好，就这么办，记一下", "src2": 1}, dup_id, dup_id[0]))

    def test_gate_off_second_hit_unchanged(self):
        ch = self.chunk()
        f = {"quote2": "好，就这么办，记一下", "src2": 1}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            self.assertIs(FE._second_hit(f, ch, ch[2]), ch[1])


class QuoteNormalization(unittest.TestCase):
    def chunk(self, text):
        return [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
                 "content": "助理甲（2026-10-01 10:00）：" + text}]

    def fact(self, quote, **kw):
        d = {"keep": True, "category": "self_correction", "subject": "claude", "type": "procedural",
             "content": "助理甲承认把「试一试」说成了事后重新定义，并为此道歉", "quote": quote, "src": 0}
        d.update(kw)
        return d

    def test_norm_quote_variants(self):
        a = '我把昨天的方案事后重新定义成"试一试",等于否定了你!'
        b = "我把昨天的方案事后重新定义成“试一试”，等于否定了你！"
        c = "我把昨天的方案事后重新定义成「试一试」，等于否定了你！"
        self.assertEqual(VG.norm_quote(a), VG.norm_quote(b))
        self.assertEqual(VG.norm_quote(b), VG.norm_quote(c))
        self.assertTrue(VG.quote_in("“试一试”，等于", "我把昨天的方案事后重新定义成\"试一试\",等于否定了你"))
        self.assertTrue(VG.quote_in("Chess 我玩不了", "chess我玩不了，玩了头晕"))
        self.assertTrue(VG.quote_in("我不爱看…", "我不爱看...他们吵架"))
        self.assertFalse(VG.quote_in("试一试吗", "我把昨天的方案事后重新定义成试一试"))

    def test_curly_quote_in_model_quote_matches_straight_in_evidence(self):
        ev = '我把昨天的方案事后重新定义成"试一试",等于否定了你的付出。这条我道歉'
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(self.fact("重新定义成“试一试”，等于否定了你的付出。这条我道歉"), self.chunk(ev))
            self.assertIsNotNone(ok, why)
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            ok, why = FE.check(self.fact("重新定义成“试一试”，等于否定了你的付出。这条我道歉"), self.chunk(ev))
            self.assertIsNone(ok)                                                             # 闸关：一字不差

    def test_fullwidth_halfwidth_punct_and_case(self):
        ev = "好的,我记住了!以后换模型先看价格再动.Chess我也删了"
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(self.fact("我记住了！以后换模型先看价格再动。chess我也删了", content="助理甲答应以后换模型先看价格再动，我记住了"), self.chunk(ev))
            self.assertIsNotNone(ok, why)

    def test_cap_quote_truncates_at_clause(self):
        q = "我把昨天的方案事后重新定义成「试一试」，等于连你当时的付出一起否定了。你人在项目里，我事后单方面改了它的性质——这条我道歉，而且以后不会再这样"
        c = VG.cap_quote(q)
        self.assertLessEqual(len(c), VG.QUOTE_MAX)
        self.assertGreaterEqual(len(c), 20)
        self.assertTrue(q.startswith(c))
        self.assertEqual(VG.cap_quote("短引文不用截"), "短引文不用截")
        hard = "字" * 100
        self.assertEqual(len(VG.cap_quote(hard)), VG.QUOTE_MAX)

    def test_long_quote_truncated_but_kept_and_very_long_rejected(self):
        body = "我把昨天的方案事后重新定义成「试一试」，等于连你当时的付出一起否定了。你人在项目里，我事后单方面改了它的性质——这条我道歉，而且以后不会再这样做了"
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(self.fact(body), self.chunk(body))
            self.assertIsNotNone(ok, why)
            self.assertLessEqual(len(ok["quote"]), VG.QUOTE_MAX)
            self.assertIn(ok["quote"].replace("「", "").replace("」", "")[:10], body.replace("「", "").replace("」", ""))
            huge = ("我错了，我以后先验证再回答。" * 20)
            ok, why = FE.check(self.fact(huge[:200]), self.chunk(huge))
            self.assertIsNone(ok)
            self.assertIn("quote-too-long", why)


class ThinkingText(unittest.TestCase):
    EN = "She's asking why I haven't been sending the weekly report. Let me check the history first before I answer."

    def test_prefilter_marks_english_thinking_for_claude_only(self):
        self.assertEqual(VG.prefilter("我（2026-10-01 10:00）：" + self.EN), "pre:thinking-text")
        self.assertEqual(VG.prefilter("助理甲（2026-10-01 10:00）：" + self.EN), "pre:thinking-text")
        self.assertEqual(VG.prefilter(self.EN, "claude"), "pre:thinking-text")
        self.assertIsNone(VG.prefilter("小测（2026-10-01 10:00）：" + self.EN))              # 她的英文不是思考文本
        self.assertIsNone(VG.prefilter(self.EN, "tester"))
        for ok in ("助理甲（2026-10-01 10:00）：好，我记住了，以后先问你再动", "助理甲（2026-10-01 10:00）：Let me 帮你查一下，稍等，查完马上告诉你"):
            self.assertIsNone(VG.prefilter(ok), ok)

    def test_not_in_hard_pre_so_it_stays_context(self):
        self.assertNotIn("pre:thinking-text", VG.HARD_PRE)
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            evs = [{"id": 1, "source_actor": "claude", "content": "我（2026-10-01 10:00）：" + self.EN}]
            self.assertEqual(FE.prefilter_evs(evs), evs)                                      # 留着当上下文

    def test_quote_cannot_come_from_thinking_sentence(self):
        text = "助理甲（2026-10-01 10:00）：" + self.EN + " 你说得对我该挨骂，聊了这么久一份周报都没发，以后我会主动发。"
        chunk = [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1", "content": text}]
        f = {"keep": True, "category": "commitment", "subject": "claude", "type": "procedural",
             "content": "助理甲承诺以后会主动发周报", "quote": "Let me check the history first before I answer", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(f, chunk)
            self.assertIsNone(ok)
            self.assertIn("pre:thinking-text", why)
            ok, why = FE.check({**f, "quote": "聊了这么久一份周报都没发，以后我会主动发"}, chunk)
            self.assertIsNotNone(ok, why)                                                     # 同一行里的中文回复照样能当引文

    def test_off_means_old_behavior(self):
        text = "助理甲（2026-10-01 10:00）：" + self.EN
        chunk = [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1", "content": text}]
        f = {"keep": True, "subject": "claude", "type": "procedural", "content": "助理甲会先查历史再回答", "quote": "Let me check the history first", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            ok, why = FE.check(f, chunk)
            self.assertIsNotNone(ok, why)


class UnionRecall(unittest.TestCase):
    CHUNK = [
        {"id": 10, "source_actor": "tester", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
         "content": "小测（2026-10-01 10:00）：以后你不要催我背单词，我自己会背，这个你记住，定下来了"},
        {"id": 11, "source_actor": "claude", "occurred_at": "2026-10-01T00:01:00Z", "path": "p", "line": "L2",
         "content": "助理甲（2026-10-01 10:01）：好，我答应你，以后不再催你背单词，我记住了"},
        {"id": 12, "source_actor": "tester", "occurred_at": "2026-10-01T00:02:00Z", "path": "p", "line": "L3",
         "content": "小测（2026-10-01 10:02）：我决定明年考公，报税务岗位，已经定了，你别劝我"},
        {"id": 13, "source_actor": "tester", "occurred_at": "2026-10-01T00:03:00Z", "path": "p", "line": "L4",
         "content": "小测（2026-10-01 10:03）：晚上煮了一锅汤，味道不错，下次还这样做"},
        {"id": 14, "source_actor": "tester", "occurred_at": "2026-10-01T00:04:00Z", "path": "p", "line": "L5",
         "content": "小测（2026-10-01 10:04）：我喜欢吃草莓味的冰淇淋，不喜欢香草味的那种"},
    ]

    def new(self, i, quote, content, cat="preference", subj="tester", typ="preference"):
        return {"keep": True, "category": cat, "subject": subj, "type": typ, "content": content, "quote": quote, "src": i}

    def old(self, i, quote, content, subj="tester", typ="procedural"):
        return {"subject": subj, "type": typ, "content": content, "quote": quote, "src": i, "confidence": 0.9, "_old": True}

    def test_old_big_facts_join_and_cap_raised_for_priority_only(self):
        newf = [self.new(3, "我喜欢吃草莓味的冰淇淋", "小测喜欢吃草莓味的冰淇淋"),
                self.new(4, "不喜欢香草味的那种", "小测不喜欢香草味"),
                self.new(3, "晚上煮了一锅汤", "小测晚上煮了一锅汤，觉得好喝", cat="life_change", typ="episodic")]
        oldf = [self.old(0, "以后你不要催我背单词", "小测定下规矩：助理甲以后不要催她背单词"),
                self.old(2, "我决定明年考公，报税务岗位", "小测决定明年考公，报税务岗位"),
                self.old(3, "味道不错，下次还这样做", "小测觉得那锅汤好喝，下次还煮")]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            res = FE.gate_pass(newf + oldf, self.CHUNK)
        kept = [ok["content"] for ok, _ in res if ok]
        self.assertEqual(len(kept), 5, [w for _, w in res])                                    # 3 新 + 2 个大事（超 3 条上限，≤5）
        self.assertTrue(any("催她背单词" in k for k in kept))
        self.assertTrue(any("考公" in k for k in kept))
        self.assertFalse(any("下次还去" in k for k in kept))                                    # 旧候选里的小事不进
        self.assertTrue(all(ok["old"] for ok, _ in res if ok and "考公" in ok["content"]))

    def test_normal_new_facts_still_capped_at_three(self):
        newf = [self.new(4, "我喜欢吃草莓味的冰淇淋", "小测喜欢吃草莓味的冰淇淋"),
                self.new(4, "不喜欢香草味的那种", "小测不喜欢香草味的冰淇淋", typ="preference"),
                self.new(3, "晚上煮了一锅汤", "小测晚上煮了一锅汤，觉得好喝", cat="life_change", typ="episodic"),
                self.new(3, "下次再去那家店", "小测下次还要去那家面条店吃饭", cat="decision", typ="procedural")]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            res = FE.gate_pass(newf, self.CHUNK)
        self.assertLessEqual(sum(1 for ok, _ in res if ok), 3)

    def test_dedupe_old_vs_new(self):
        newf = [self.new(2, "我决定明年考公，报税务岗位", "小测决定明年考公，报税务岗位", cat="decision", typ="procedural")]
        oldf = [self.old(2, "我决定明年考公", "小测已定明年考公，报税务岗位方向")]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            res = FE.gate_pass(newf + oldf, self.CHUNK)
        self.assertEqual([bool(ok) for ok, _ in res], [True, False])
        self.assertIn("union:dup", res[1][1])

    def test_old_facts_pass_same_postchecks_and_roles(self):
        oldf = [self.old(1, "以后不再催你背单词", "小测以后不再催助理甲睡觉", subj="tester"),                     # 说话人是助手：回声
                self.old(0, "我自己会背，这个你记住", "他们以后不要催背单词", subj="tester"),                       # 代词主语
                self.old(1, "我答应你，以后不再催你背单词", "助理甲答应小测以后不再催她背单词", subj="claude")]       # 对的
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            res = FE.gate_pass(oldf, self.CHUNK)
        self.assertEqual([bool(ok) for ok, _ in res], [False, False, True], [w for _, w in res])
        self.assertEqual(res[2][0]["category"], "commitment")

    def test_union_flag_default_off_and_gate_off_unchanged(self):
        self.assertFalse(flags.DEFAULTS["VALUE_GATE_UNION"])
        os.environ.pop("ENGRAM_VALUE_GATE_UNION", None)
        self.assertFalse(FE.union_on())
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "1"}):
            self.assertFalse(FE.union_on())                                                    # 闸没开，补偿也不开
            res = FE.gate_pass([self.new(4, "我喜欢吃草莓味的冰淇淋", "小测喜欢吃草莓味的冰淇淋")] * 5, self.CHUNK)
            self.assertEqual(sum(1 for ok, _ in res if ok), 5)                                 # 关闸：没有封顶/去重，一字不差

    def test_call_model_old_tags_and_survives_failure(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1", "ENGRAM_VALUE_GATE_UNION": "1"}):
            with mock.patch.object(FE, "chat", return_value=json.dumps({"facts": [{"subject": "tester", "content": "x"}]})) as m:
                got = FE.call_model_old("原话")
                self.assertEqual(got, [{"subject": "tester", "content": "x", "_old": True}])
                self.assertIn("过一个月还可能想得起来", m.call_args[0][0][1]["content"])      # 旧提示词
            with mock.patch.object(FE, "chat", side_effect=RuntimeError("boom")):
                self.assertEqual(FE.call_model_old("原话"), [])
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1", "ENGRAM_VALUE_GATE_UNION": "0"}):
            with mock.patch.object(FE, "chat") as m:
                self.assertEqual(FE.call_model_old("原话"), [])
                m.assert_not_called()

    def test_is_priority(self):
        self.assertTrue(VG.is_priority("tester", "decision", "小测定下规矩：助理甲不要催她背单词", "tester"))
        self.assertTrue(VG.is_priority("tester", "life_change", "小测决定搬出去住", "tester"))
        self.assertTrue(VG.is_priority("tester", "decision", "小测明年考公", "tester"))
        self.assertFalse(VG.is_priority("tester", "life_change", "小测今天煮了汤", "tester"))
        self.assertTrue(VG.is_priority("claude", "commitment", "助理甲保证以后先问小测", "claude"))
        self.assertFalse(VG.is_priority("claude", "self_correction", "助理甲记得有这回事", "claude"))
        self.assertFalse(VG.is_priority("gpt", "commitment", "助手乙保证以后先问", "gpt"))

    def test_is_priority_strict_for_old_candidates(self):
        # 旧抽取器候选只认规矩类强词；「喜欢某剧」「决定申请」这类泛词不算
        self.assertTrue(VG.is_priority("tester", "preference", "小测喜欢某部古装剧", "tester"))
        self.assertFalse(VG.is_priority("tester", "preference", "小测喜欢某部古装剧", "tester", strict=True))
        self.assertFalse(VG.is_priority("tester", "decision", "小测决定申请加入某社团", "tester", strict=True))
        self.assertTrue(VG.is_priority("tester", "decision", "小测要求以后不要催她背单词", "tester", strict=True))
        self.assertTrue(VG.is_priority("tester", "life_change", "小测决定搬出去住", "tester", strict=True))


class PromptTweaks(unittest.TestCase):
    def test_kept_tweaks_only_b(self):
        self.assertEqual(VG.EXTRA_RULES, VG.TWEAKS["B"])
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            p = FE.build_prompt("X")
        self.assertIn(VG.TWEAKS["B"], p)
        self.assertNotIn(VG.TWEAKS["A"], p)
        self.assertNotIn(VG.TWEAKS["C"], p)

    def test_extra_rules_prompt_has_no_placeholder(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            p = FE.build_prompt("X")
        self.assertNotIn("{extra}", p)
        self.assertEqual(p, VG.PROMPT_GATE.format(chunk="X", extra=VG.EXTRA_RULES))
        for k, v in VG.TWEAKS.items():
            self.assertNotIn("{", v)
            self.assertIn(v, VG.PROMPT_GATE.format(chunk="X", extra=v))

    def test_off_prompt_untouched(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            self.assertEqual(FE.build_prompt("X"), FE.PROMPT.format(chunk="X"))


if __name__ == "__main__":
    unittest.main()
