"""「值不值得记」闸：预过滤、角色规则、后检、置信度、提示词、接线（开关默认关）。
反面例子照着已经被丢弃的 GA2/GA3 那几类垃圾写（设定介绍、过程流水账、对第三方的评价、转述、指标）。"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import flags  # noqa: E402
import value_gate as VG  # noqa: E402
from extract import from_evidence as FE  # noqa: E402


class Prefilter(unittest.TestCase):
    def test_drops(self):
        for t, why in [("嗯", "pre:ack"), ("好的好的", "pre:ack"), ("收到！", "pre:ack"), ("哈哈哈哈", "pre:ack"), ("👍", "pre:ack"),
                       ("heartbeat", "pre:heartbeat"), ("HEARTBEAT_OK", "pre:heartbeat"), ("ping", "pre:heartbeat"),
                       ("<tool_result>ok</tool_result>", "pre:tool-output"), ("Exit code: 1 something", "pre:tool-output"),
                       ('{"a": 1, "b": [1,2,3], "c": "x"}', "pre:json-blob"),
                       ("Traceback (most recent call last):\n  File x\nError: boom", "pre:error-log"),
                       ("Error: connect ECONNREFUSED 127.0.0.1:80", "pre:error-log"),
                       ("2026-10-27 12:00:01 ERROR failed to connect", "pre:error-log"),
                       ("好", "pre:ack"), ("abc", "pre:too-short"), ("  。  ", "pre:too-short")]:
            self.assertEqual(VG.prefilter(t), why, t)

    def test_render_prefix_stripped(self):
        self.assertEqual(VG.prefilter("小测（2026-10-27 12:00）：嗯"), "pre:ack")
        self.assertEqual(VG.prefilter("助理甲（2026-10-27 12:00）：heartbeat"), "pre:heartbeat")

    def test_keeps_real_content(self):
        for t in ["我下周三要去考驾照，你到时候提醒我一下", "以后你每次改完配置都先告诉我，我不想再被吓到",
                  "我答应你，下次不会再没问你就把库清掉了", "小测说她最讨厌吃香菜", "I prefer tea over coffee in the morning"]:
            self.assertIsNone(VG.prefilter(t), t)
        self.assertIsNone(VG.prefilter("想玩跑酷"))             # 四个字也可能是一条偏好


class Roles(unittest.TestCase):
    def test_claude_only(self):
        self.assertIsNone(VG.role_check("claude", "commitment", {"claude"}))
        self.assertIsNone(VG.role_check("claude", "self_correction", {"claude"}))
        self.assertEqual(VG.role_check("claude", "preference", {"claude"}), "role:claude-only-category")
        self.assertEqual(VG.role_check("tester", "preference", {"claude"}), "role:claude-only-subject-mismatch")
        self.assertEqual(VG.role_check("gpt", "life_change", {"claude"}), "role:claude-only-subject-mismatch")

    def test_user_categories_only_from_user(self):
        for c in VG.USER_CATEGORIES:
            self.assertIsNone(VG.role_check("tester", c, {"tester"}))
            self.assertIsNotNone(VG.role_check("claude", c, {"claude"}))
        self.assertEqual(VG.role_check("tester", "commitment", {"tester"}), "role:category-not-allowed-for-speaker")

    def test_us_needs_user_confirmation_after(self):
        self.assertEqual(VG.role_check("us", "agreement", {"claude"}), "role:us-one-sided")
        self.assertEqual(VG.role_check("us", "agreement", {"tester"}), "role:us-one-sided")
        self.assertEqual(VG.role_check("us", "agreement", {"tester", "claude"}, confirmed_by_user_after=False),
                         "role:us-no-user-confirmation")
        self.assertIsNone(VG.role_check("us", "agreement", {"tester", "claude"}, confirmed_by_user_after=True))
        self.assertEqual(VG.role_check("us", "preference", {"tester", "claude"}), "role:us-category")

    def test_speaker_subject_mismatch(self):
        self.assertEqual(VG.role_check("gpt", "life_change", {"tester"}), "role:speaker-subject-mismatch")


class Postcheck(unittest.TestCase):
    def fact(self, content, **kw):
        d = {"keep": True, "category": "preference", "subject": "tester", "content": content, "quote": "我讨厌香菜的味道"}
        d.update(kw)
        return d

    EV = "我讨厌香菜的味道，每次吃到都想吐"

    def test_good_fact_passes(self):
        self.assertIsNone(VG.postcheck(self.fact("小测讨厌香菜的味道"), self.EV, actor="tester"))

    def test_quote_must_be_substring(self):
        self.assertEqual(VG.postcheck(self.fact("小测讨厌香菜", quote="我特别特别讨厌香菜"), self.EV, actor="tester"), "post:quote-not-substring")
        self.assertEqual(VG.postcheck(self.fact("小测讨厌香菜", quote="香"), self.EV, actor="tester"), "post:no-quote")

    def test_category_enum_and_role(self):
        self.assertEqual(VG.postcheck(self.fact("小测讨厌香菜", category="gossip"), self.EV, actor="tester"), "post:category-not-in-enum")
        self.assertEqual(VG.postcheck(self.fact("助理甲讨厌香菜", category="preference", subject="claude"), self.EV, actor="claude"),
                         "post:category-not-allowed-for-role")
        self.assertEqual(VG.postcheck(self.fact("x", keep=False), self.EV), "post:model-said-no")

    def test_vague_or_pronoun_subject(self):
        self.assertEqual(VG.postcheck(self.fact("她讨厌香菜的味道"), self.EV, actor="tester"), "post:pronoun-subject")
        self.assertEqual(VG.postcheck(self.fact("有人讨厌香菜的味道", subject="某人"), self.EV, actor="tester"), "post:vague-subject")
        self.assertEqual(VG.postcheck(self.fact("讨厌香菜", subject="他"), self.EV, actor="tester"), "post:vague-subject")

    def test_pronoun_check_is_not_fooled_by_qishi_and_skips_verbatim(self):
        self.assertIsNone(VG.postcheck(self.fact("小测其实讨厌香菜的味道"), self.EV, actor="tester"))
        self.assertIsNone(VG.postcheck(self.fact("小测（2026-10-01 10:00）：其实我讨厌香菜的味道"), self.EV, actor="tester"))
        self.assertIsNone(VG.postcheck(self.fact("这个人讨厌香菜的味道".replace("这个人", "小测")), self.EV, actor="tester"))

    def test_grounding_uses_context_when_given(self):
        f = {"keep": True, "category": "life_change", "subject": "tester", "content": "小测上架了第一个小程序"}
        self.assertEqual(VG.postcheck(f, "弄好了呀", actor="tester"), "post:ungrounded")
        self.assertIsNone(VG.postcheck(f, "弄好了呀", actor="tester", context_text="我把第一个小程序上架好了 弄好了呀"))

    def test_ungrounded(self):
        self.assertEqual(VG.postcheck(self.fact("小测最近迷上了在海边冲浪还买了新板子"), self.EV, actor="tester"), "post:ungrounded")

    def junk(self, content, why, ev=None, **kw):
        ev = ev or content
        f = {"keep": True, "category": "commitment", "subject": "claude", "content": content, "quote": ev[:12]}
        f.update(kw)
        self.assertEqual(VG.postcheck(f, ev, actor="claude"), why, content)

    def test_discarded_junk_types(self):
        """GA2/GA3 丢掉的那几类（照形状写的同类句子，不是原句）"""
        self.junk("某助手是某团队开发的智能体，并非真人。", "post:setup-background")
        self.junk("测试群里的机器人跑在云主机上，靠小测的账号登录，并且吃账号额度。", "post:setup-background")
        self.junk("凌晨某助手压缩旧会话失败，额度很快烧完。", "post:process-log")
        self.junk("某助手的访问证书失效，更新后恢复正常。", "post:process-log")
        self.junk("群聊里因插件授权被收回而报错，助理甲暂时停用该插件。", "post:process-log")
        self.junk("某助手把摘要任务误判成外发动作，随后被拦截。", "post:process-log")
        self.junk("迁移脚本里的一项配置漏改了，结果对话被过滤成0条。", "post:process-log")
        self.junk("GPT 一开始嘴硬说成功了，被追问才承认失败。", "post:third-party-eval", category="commitment")
        self.junk("据说那边的接口下周会改版。", "post:retelling")
        self.junk("本次汇报：今天一共处理了三件事。", "post:summary-report")
        self.junk("建议你以后换成方案二。", "post:recommendation")
        self.assertIsNone(VG.postcheck({"subject": "tester", "category": "life_change", "content": "小测中午吃了焖锅排骨，是助理甲建议的"},
                                       "我中午吃了焖锅排骨啊，是助理甲建议的呀", actor="tester"))
        self.junk("加油，你一定可以的，别怕。", "post:encouragement")
        self.junk("目前服务器正在重新构建索引。", "post:transient-state", ev="目前服务器正在重新构建索引。")

    def test_metric(self):
        self.junk("会话占用了 79 万 token。", "post:metric")
        self.junk("命中率提升到百分之八十。", "post:metric")


class Confidence(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(VG.confidence_for("preference", "tester", "tester"), 0.9)
        self.assertEqual(VG.confidence_for("commitment", "claude", "claude"), 0.6)
        self.assertEqual(VG.confidence_for("agreement", "us", "tester"), 0.6)
        self.assertEqual(VG.confidence_for(None, "tester", "tester"), 0.5)


class Wiring(unittest.TestCase):
    def setUp(self):
        VG.REJECTS.clear()

    def chunk(self):
        return [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
                 "content": "助理甲（2026-10-01 10:00）：我答应你，以后改配置之前一定先问你一声"},
                {"id": 2, "source_actor": "tester", "occurred_at": "2026-10-01T00:01:00Z", "path": "p", "line": "L2",
                 "content": "小测（2026-10-01 10:01）：好，就这么定了，你说到要做到"}]

    def test_flag_default_off_and_registered(self):
        self.assertFalse(flags.DEFAULTS["VALUE_GATE"])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENGRAM_VALUE_GATE", None)
            self.assertFalse(VG.enabled())

    def test_off_means_old_prompt_and_old_check(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            self.assertIs(FE.build_prompt("X").startswith("下面是一段按时间排好的聊天原话"), True)
            self.assertNotIn("预计会拒绝", FE.build_prompt("X"))
            ok, why = FE.check({"subject": "claude", "type": "procedural", "content": "助理甲答应以后改配置前先问小测",
                                "quote": "以后改配置之前一定先问你一声", "src": 0, "confidence": 0.9}, self.chunk())
            self.assertIsNotNone(ok)
            self.assertEqual(ok["confidence"], 0.9)

    def test_on_prompt_and_commitment_passes_with_06(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            self.assertIn("预计会拒绝", FE.build_prompt("X"))
            self.assertIn("30 天", FE.build_prompt("X"))
            f = {"keep": True, "category": "commitment", "subject": "claude", "type": "procedural",
                 "content": "助理甲答应小测以后改配置之前一定先问她", "quote": "以后改配置之前一定先问你一声", "src": 0, "confidence": 0.9}
            ok, why = FE.check(f, self.chunk())
            self.assertIsNotNone(ok, why)
            self.assertEqual(ok["confidence"], 0.6)
            self.assertEqual(ok["category"], "commitment")

    def test_on_rejects_and_counts(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            base = {"keep": True, "category": "commitment", "subject": "claude", "type": "procedural",
                    "content": "助理甲答应小测以后改配置之前一定先问她", "quote": "以后改配置之前一定先问你一声", "src": 0}
            ok, why = FE.check({**base, "category": "preference"}, self.chunk())
            self.assertIsNone(ok)
            self.assertIn("role:claude-only-category", why)
            ok, why = FE.check({k: v for k, v in base.items() if k != "category"}, self.chunk())
            self.assertIn("post:category-not-in-enum", why)
            ok, why = FE.check({**base, "keep": False}, self.chunk())
            self.assertIn("post:model-said-no", why)
            self.assertEqual(VG.REJECTS["role:claude-only-category"], 1)
            self.assertEqual(sum(VG.REJECTS.values()), 3)

    def test_us_agreement_needs_user_after(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            f = {"keep": True, "category": "agreement", "subject": "us", "type": "procedural",
                 "content": "小测和助理甲约定改配置之前先问小测", "quote": "以后改配置之前一定先问你一声", "src": 0,
                 "quote2": "好，就这么定了，你说到要做到", "src2": 1}
            ok, why = FE.check(f, self.chunk())
            self.assertIsNotNone(ok, why)
            self.assertEqual((ok["subject"], ok["confidence"]), ("us", 0.6))
            rev = self.chunk()[::-1]                      # 她的话在前、助手在后：她没有「随后确认」
            f2 = {**f, "quote": "好，就这么定了，你说到要做到", "src": 0, "quote2": "以后改配置之前一定先问你一声", "src2": 1}
            ok2, why2 = FE.check(f2, rev)
            self.assertIsNone(ok2)
            # 第四轮放宽后：她在前的情形多了一条「她那句得是请求」的检查，这句只是确认、不是请求 → 原因更具体，仍然拒绝
            self.assertTrue("role:us-no-user-confirmation" in why2 or "role:us-her-line-not-request" in why2, why2)

    def test_short_ack_cannot_be_the_quote_source_in_check(self):
        chunk = [{"id": 1, "source_actor": "tester", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
                  "content": "小测（2026-10-01 10:00）：我把第一个小程序上架好了"},
                 {"id": 2, "source_actor": "tester", "occurred_at": "2026-10-01T00:01:00Z", "path": "p", "line": "L2",
                  "content": "小测（2026-10-01 10:01）：好的好的"}]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            f = {"keep": True, "category": "life_change", "subject": "tester", "type": "episodic",
                 "content": "小测上架了第一个小程序", "quote": "好的好的", "src": 1}
            ok, why = FE.check(f, chunk)
            self.assertIsNone(ok)
            self.assertIn("pre:", why)
            ok, why = FE.check({**f, "quote": "第一个小程序上架好了", "src": 0}, chunk)
            self.assertIsNotNone(ok, why)

    def test_prefilter_evs(self):
        evs = [{"content": "嗯"}, {"content": "heartbeat"}, {"content": "我下周三要去考驾照，到时候提醒我一下"}]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            self.assertEqual(len(FE.prefilter_evs(evs)), 2)          # 只丢心跳；应答留着当上下文
            self.assertEqual(VG.REJECTS["pre:heartbeat"], 1)
            self.assertEqual(VG.REJECTS["pre:ack"], 0)
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            self.assertEqual(len(FE.prefilter_evs(evs)), 3)


class GroundingWindow(unittest.TestCase):
    """修 1：依据看前后窗口，不是只看引文那一条；她的偏好仍要出自她自己的话，别人的上下文只许解释指代"""
    LINES = [("claude", "助理甲（10:00）：我把那个把天气预报搬到手表上的小工具看了一遍，创意很特别"),
             ("tester", "小测（10:01）：这个我是真喜欢"),
             ("claude", "助理甲（10:02）：那我记一下"),
             ("tester", "小测（10:30）：今天去超市买了牛奶")]

    def test_window_slices_and_splits_user_text(self):
        allt, ut = VG.window(self.LINES, 1, 1)
        self.assertIn("天气预报", allt)
        self.assertIn("记一下", allt)
        self.assertNotIn("超市", allt)
        self.assertEqual(ut, "小测（10:01）：这个我是真喜欢")
        allt0, _ = VG.window(self.LINES, 0, 8)                 # 越界不报错
        self.assertIn("超市", allt0)

    def test_referent_comes_from_context_but_verdict_from_user(self):
        content = "小测喜欢一个把天气预报搬到手表上的工具，觉得很有趣"
        fact = {"subject": "tester", "category": "preference", "content": content}
        ev = "小测（10:01）：这个我是真喜欢"
        allt, ut = VG.window(self.LINES, 1, 8)
        self.assertEqual(VG.postcheck(fact, ev, actor="tester"), "post:ungrounded")            # 只看那一条：挡（旧行为）
        self.assertIsNone(VG.postcheck(fact, ev, actor="tester", context_text=allt, user_text=ut))   # 看窗口：放

    def test_user_preference_not_made_up_from_claudes_context(self):
        content = "小测讨厌助理甲把天气预报搬到手表上的工具，觉得很有意思"
        fact = {"subject": "tester", "category": "preference", "content": content}
        allt, ut = VG.window(self.LINES, 2, 8)
        # 她自己的话（今天去超市买了牛奶）和这条偏好毫无关系，偏好是靠助手的话拼出来的
        why = VG.postcheck(fact, "小测（10:30）：今天去超市买了牛奶", actor="tester",
                           context_text=allt, user_text="小测（10:30）：今天去超市买了牛奶")
        self.assertIn(why, ("post:ungrounded-in-user-words", "post:ungrounded"))
        self.assertIsNotNone(why)

    def test_user_text_check_ignored_for_claude_and_when_not_given(self):
        f = {"subject": "claude", "category": "commitment", "content": "助理甲答应每次改配置先问小测"}
        self.assertIsNone(VG.postcheck(f, "我答应你每次改配置先问你", actor="claude", context_text="我答应你每次改配置先问你", user_text="无关的话"))

    def test_paraphrase_with_pronoun_shift_is_grounded_by_char_cover(self):
        f = {"subject": "tester", "category": "preference", "content": "小测不喜欢助理甲管她穿衣服，觉得烦"}
        ev = "被你吵烦了都 一个机器人管人类穿衣服，还总是不看天气就乱提醒"
        self.assertGreaterEqual(VG.char_cover(f["content"], ev), VG.CHAR_COVER_THR)
        self.assertIsNone(VG.postcheck(f, ev, actor="tester", context_text=ev))
        junk = {"subject": "tester", "category": "preference", "content": "小测最近迷上了在海边冲浪还买了新板子"}
        self.assertEqual(VG.postcheck(junk, ev, actor="tester", context_text=ev), "post:ungrounded")


class WordlistPriority(unittest.TestCase):
    """修 2：承诺/自我纠正赢过鼓励、过程、状态、总结词；她明确的决定/要求赢过过程、状态、总结和光秃秃的「额度」"""
    def claude(self, content, cat="commitment"):
        return VG.postcheck({"subject": "claude", "category": cat, "content": content}, content, actor="claude")

    def test_apology_plus_promise_beats_encouragement(self):
        self.assertIsNone(self.claude("助理甲说我错了，保证下次先查记录再说，抱抱小测", "self_correction"))
        self.assertIsNone(self.claude("助理甲承认重启后又说错了，以后重启前先告诉小测", "self_correction"))
        self.assertIsNone(self.claude("助理甲决定以后不再给小测报错误计数，不再把认错做成总结报表", "commitment"))

    def test_plain_comfort_and_process_still_blocked(self):
        self.assertEqual(self.claude("助理甲抱抱小测，别担心，会好的"), "post:encouragement")
        self.assertEqual(self.claude("助理甲重启了服务，报错消失"), "post:process-log")

    def test_commitment_beats_bare_quota_word(self):
        self.assertIsNone(self.claude("助理甲决定像装额度报警一样给自己装个技术护栏，以后命令写到生产库就拦下"))

    def test_commit_win_does_not_cover_hard_lists(self):
        self.assertEqual(self.claude("助理甲保证下次额度命中率达到 90%"), "post:metric")
        self.assertEqual(self.claude("据说助理甲保证下次不再说错"), "post:retelling")

    def user(self, content):
        return VG.postcheck({"subject": "tester", "category": "decision", "content": content}, content, actor="tester")

    def test_user_durable_statement_beats_process_summary_quota(self):
        self.assertIsNone(self.user("小测同意看板加“当前版本”一栏，额度那块先不动"))
        self.assertIsNone(self.user("小测要求证书到期的事别天天报，临近前一两天提醒她就行"))
        self.assertIsNone(self.user("小测决定以后让大模型写摘要，不再手写"))

    def test_user_chitchat_with_quota_words_still_blocked_and_numbers_always(self):
        self.assertEqual(self.user("小测说今天额度又掉了好快"), "post:metric")
        self.assertEqual(self.user("小测决定留到周日再用，周额度只剩 8%"), "post:metric")
        self.assertEqual(self.user("小测说今天服务正在重启"), "post:process-log")

    def test_verbatim_copy_gets_no_durable_override(self):
        self.assertEqual(self.user("小测（2026-09-12 11:44）：好，确认了：不是缓存的问题，改完要重启才行"), "post:process-log")

    def test_bare_summary_word_is_topic_not_report(self):
        self.assertIsNone(self.user("小测希望助理甲读完周报后主动告诉她，别不吭声"))
        self.assertEqual(VG.postcheck({"subject": "claude", "category": "commitment", "content": "本次汇报：摘要如下，三件事"},
                                      "本次汇报：摘要如下，三件事", actor="claude"), "post:summary-report")


class ReplayParity(unittest.TestCase):
    """修 3：应答/太短留作上下文，只是不能当引文来源（和真管线一致）"""
    LINES = [("tester", "小测（09:50）：我把第一个正经小程序在应用商店上架了，还写了六条更新说明"),
             ("claude", "助理甲（09:51）：不错，描述里有个错别字"),
             ("tester", "小测（09:52）：弄好了🙉")]

    def test_ack_line_cannot_be_source_but_neighbor_can(self):
        self.assertEqual(VG.prefilter(self.LINES[2][1]), "pre:too-short")
        content = "小测在应用商店上架了第一个正经小程序，写了六条更新说明"
        self.assertEqual(VG.quote_source(content, self.LINES, 2, "tester"), 0)

    def test_no_neighbor_with_content_means_block(self):
        lines = [("tester", "小测（09:50）：嗯嗯"), ("tester", "小测（09:52）：弄好了🙉")]
        self.assertIsNone(VG.quote_source("小测在应用商店上架了第一个正经小程序", lines, 1, "tester"))
        # 邻居是别人说的：不算她的话
        lines2 = [("claude", "助理甲（09:50）：我在应用商店上架了第一个正经小程序，写了六条更新说明"), ("tester", "小测（09:52）：弄好了🙉")]
        self.assertIsNone(VG.quote_source("小测在应用商店上架了第一个正经小程序，写了六条更新说明", lines2, 1, "tester"))

    def test_fact_that_is_itself_an_ack_is_not_rescued(self):
        self.assertIsNone(VG.quote_source("小测（2026-09-17 21:58）：666", self.LINES, 2, "tester"))

    def test_ack_stays_in_the_slice_prefilter_evs(self):
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            evs = [{"content": "小测（09:52）：弄好了🙉"}, {"content": "heartbeat"}]
            self.assertEqual(len(FE.prefilter_evs(evs)), 1)


class InferCategory(unittest.TestCase):
    """修 4：助手自己立的规矩/承诺算 commitment，承认并改正算 self_correction；复述别人、描述动作仍是 other"""
    def inf(self, c):
        return VG.infer_category(c, {"claude"}, "claude")

    def test_self_set_rules_are_commitments(self):
        for c in ["助理甲以后会注意把想让小测看到的话通过邮件发送，不留在终端自言自语。",
                  "助理甲以后发消息每段会控制短一点，避免句子被硬切。",
                  "助理甲定下的做法：如果真看不懂小测在说什么，会问一句，不会瞎顺着回答。",
                  "助理甲定下以后会议一散当场就记。",
                  "助理甲（2026-09-15 04:49）：立规矩——凡是要写“昨天”“上周”这种相对日期，先查再写。",
                  "助理甲和助手丙约定：话题与自己无关、也没被点名时，助理甲就不插话。",
                  "以后小测发项目截图，助理甲先看文档再说话。"]:
            self.assertEqual(self.inf(c), "commitment", c)
        self.assertEqual(VG.role_check("claude", self.inf("助理甲以后会注意先查记录"), {"claude"}), None)

    def test_self_correction(self):
        for c in ["助理甲（2026-09-17 00:17）：承认时间说错了，查了是零点二十，说“是我又随口说的”",
                  "助理甲认错，把小测的意思从做插件纠正为做独立应用。",
                  "助理甲（2026-09-21 16:51）：小测说过会议在周三，助理甲记成了周四，自认“错题本+1”"]:
            self.assertEqual(self.inf(c), "self_correction", c)

    def test_descriptions_relays_and_banter_stay_other(self):
        for c in ["助理甲发现同步服务从3月5日起一直认为自己是离线状态。",
                  "助理甲（2026-09-02 04:29）：替它说一句，它很老实——它承认“第三步我没把握还说得那么笃定”",
                  "助理甲（2026-09-16 14:04）：不过昨天晚上你可是说我“偷懒来的”——现在改口了？",
                  "助理甲方便以后做看板或预警时查阅，把接口用法存入了文件。",
                  "助理甲（2026-10-15 15:31）：换成定时以后才是真正的到点提醒，一天就花那一次"]:
            self.assertEqual(self.inf(c), "other", c)
        self.assertEqual(VG.role_check("claude", "other", {"claude"}), "role:claude-only-category")

    def test_user_and_us(self):
        self.assertEqual(VG.infer_category("小测喜欢蓝色", {"tester"}, "tester"), "preference")
        self.assertEqual(VG.infer_category("x", {"tester", "claude"}, "us"), "agreement")
        self.assertIsNone(VG.infer_category("小测在吃饭", {"tester"}, "tester"))


class LowValueLeaks(unittest.TestCase):
    """修 5：整句照抄原话、又只是个问句的闲聊；转述别的窗口里的话。只收窄，不误伤带决定/事实的原话"""
    def user(self, c):
        return VG.postcheck({"subject": "tester", "content": c}, c, actor="tester")

    def test_verbatim_chitchat_question(self):
        self.assertEqual(self.user("小测（2026-09-16 11:19）：这个项目才做了三周吗，感觉像做了一整年🤡"), "post:verbatim-question")
        self.assertEqual(self.user("小测（2026-09-17 23:39）：整个报表系统你做完啦？🤡"), "post:verbatim-question")

    def test_verbatim_with_decision_or_not_a_question_or_long_is_kept(self):
        self.assertIsNone(self.user("小测（2026-09-16 11:19）：以后你改配置之前先问我好吗"))
        self.assertIsNone(self.user("小测（2026-09-16 11:19）：每月预算大部分拿来买书和茶叶用。"))
        self.assertIsNone(self.user("小测（2026-09-16 11:19）：" + "我今天一整天都在仓库帮忙整理东西，" * 6 + "你说怎么办呢？"))
        self.assertIsNone(self.user("小测想问助理甲要不要换窗口，她决定明天再换"))      # 不是整句照抄

    def test_other_window_relay(self):
        c = "助理甲（2026-09-24 14:33）：说那是第一个窗口，同一条线上的他，但“那次是他答应的，不是我答应的”"
        self.assertEqual(VG.postcheck({"subject": "claude", "category": "commitment", "content": c}, c, actor="claude"), "post:retelling")
        c2 = "助理甲答应小测以后重启前先说一声，这是在第一个窗口里定下的做法"
        self.assertIsNone(VG.postcheck({"subject": "claude", "category": "commitment", "content": c2}, c2, actor="claude"))

    def test_wiring_user_text_passed_to_postcheck(self):
        chunk = [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
                  "content": "助理甲（2026-10-01 10:00）：这个工具是把天气预报搬到手表上的，创意很特别"},
                 {"id": 2, "source_actor": "tester", "occurred_at": "2026-10-01T00:01:00Z", "path": "p", "line": "L2",
                  "content": "小测（2026-10-01 10:01）：这个我是真喜欢"}]
        f = {"keep": True, "category": "preference", "subject": "tester", "type": "semantic",
             "content": "小测喜欢把天气预报搬到手表上的工具，觉得很有意思", "quote": "这个我是真喜欢", "src": 1}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(f, chunk)
            self.assertIsNotNone(ok, why)
            # 引文是她的话，但整段里她自己一句有内容的话都没有：偏好是助手的话拼出来的
            f2 = {**f, "content": "小测讨厌把天气预报搬到手表上的工具，觉得很无聊", "quote": "这个我是真喜欢"}
            ok2, why2 = FE.check(f2, [chunk[0], {**chunk[1], "content": "小测（2026-10-01 10:01）：嗯嗯这个我是真喜欢"}])
            self.assertIsNotNone(ok2 or why2)



class TrialFixes(unittest.TestCase):
    """真模型试跑（2026-10-28）发现的两处：PROMPT_GATE 不输出 type、模型引文的空白和原话不一致"""
    CHUNK = [{"id": 1, "source_actor": "claude", "occurred_at": "2026-10-01T00:00:00Z", "path": "p", "line": "L1",
              "content": "助理甲（2026-10-01 10:00）：是我拿着一条为平板定的规矩，套到手表上了。  那我收回。"}]

    def test_missing_type_is_filled_from_category_when_gate_on(self):
        f = {"keep": True, "category": "self_correction", "subject": "claude",
             "content": "助理甲承认把为平板定的规矩套到了手表上，并收回", "quote": "套到手表上了", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(f, self.CHUNK)
            self.assertIsNotNone(ok, why)
            self.assertEqual(ok["type"], "procedural")
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            ok, why = FE.check(f, self.CHUNK)
            self.assertIsNone(ok)
            self.assertIn("type 不对", why)               # 关闸时一字不差

    def test_quote_whitespace_insensitive_only_when_gate_on(self):
        f = {"keep": True, "category": "self_correction", "subject": "claude", "type": "procedural",
             "content": "助理甲承认把为平板定的规矩套到了手表上，并收回", "quote": "套到手表上了。那我收回", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(f, self.CHUNK)
            self.assertIsNotNone(ok, why)
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0"}):
            ok, why = FE.check(f, self.CHUNK)
            self.assertIsNone(ok)
            self.assertIn("quote 在原话里找不到", why)


if __name__ == "__main__":
    unittest.main()
