"""2026-10-29 真模型实测后的修补：说话人前缀不是主体 / 生日同义写法 / 原子拆分候选 / 重复与同日屏蔽。全部虚构夹具。"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maintenance"))
import metamem  # noqa: E402
import persona  # noqa: E402
import router  # noqa: E402
import split_atomic as SA  # noqa: E402
from extract import timeline  # noqa: E402
from store.store import add_evidence, connect, promote, record_class, search  # noqa: E402

U, C = persona.USER, persona.CLAUDE
NU, NC = persona.N_USER, persona.N_CLAUDE


def hit(subject, content, ext="x"):
    return {"ext_id": ext, "subject": subject, "content": content, "score": 0.9}


class SpeakerPrefixIsNotSubject(unittest.TestCase):
    def test_prefixed_fact_filed_under_speaker_still_belongs_to_user_when_text_is_about_user(self):
        h = hit(C, f"{NC}（2026-08-20 13:57）：28号是{NU}生日")
        self.assertTrue(metamem.belongs_to(h, U))
        self.assertFalse(metamem.belongs_to(h, C))

    def test_prefixed_fact_about_claude_stays_claude_s(self):
        h = hit(C, f"我（2026-08-20 13:57）：{NC}生日7月21日，陪同的人是{NU}")
        self.assertFalse(metamem.belongs_to(h, U))

    def test_mention_far_from_the_start_does_not_reattribute(self):
        """助理的长句里末尾顺带提到她，不能因此算成她的事（大样本实测发现的误提升）。"""
        h = hit(C, f"{NC}（2026-08-20 13:57）：说图中爪印五个脚趾，像用户爪子，不上色就素色，上色可以问商家，最后留给{NU}自己挑")
        self.assertFalse(metamem.belongs_to(h, U))

    def test_non_prefixed_wrong_subject_unchanged(self):
        self.assertFalse(metamem.belongs_to(hit(C, f"{NU}的生日是5月20日"), U))

    def test_judge_and_prefer_use_it(self):
        hs = [hit(C, f"{NC}的生日是3月3日", "a"), hit(C, f"{NC}（2026-08-20 13:57）：28号是{NU}生日", "b")]
        self.assertEqual([h["ext_id"] for h in metamem.prefer_subject("我哪天生日", hs)], ["b", "a"])
        self.assertTrue(metamem.judge("我哪天生日", [hs[1]]).enough)


class BirthdayAlias(unittest.TestCase):
    def test_expand_appends_only(self):
        self.assertEqual(router.expand_aliases("我哪天生日", ["生日"]), ["生日", "日生", "出生"])
        self.assertEqual(router.expand_aliases("今天吃什么", ["吃什么"]), ["吃什么"])

    def test_keyword_path_finds_birth_phrasing_via_alias(self):
        with tempfile.TemporaryDirectory() as d:
            con = connect(Path(d) / "t.db")
            eid = add_evidence(con, source_kind="chat", source_actor=U, source_locator="conv:t#msg:1",
                               content="小林2000年5月20日生，属龙", occurred_at="2026-09-01")
            promote(con, subject=U, fact_type="semantic", content="小林2000年5月20日生，属龙",
                    evidence_ids=[eid], valid_from="2026-09-01", importance=0.5, confidence=0.7)
            self.assertEqual(search(con, "生日"), [])           # 原词字面对不上
            self.assertEqual(len(search(con, "日生")), 1)        # 同义写法能找到
            con.close()

    def test_retrieve_uses_alias_only_when_metamem_on(self):
        import retrieve
        with tempfile.TemporaryDirectory() as d:
            con = connect(Path(d) / "t.db")
            eid = add_evidence(con, source_kind="chat", source_actor=U, source_locator="conv:t#msg:1",
                               content="小林2000年5月20日生，属龙", occurred_at="2026-09-01")
            promote(con, subject=U, fact_type="semantic", content="小林2000年5月20日生，属龙",
                    evidence_ids=[eid], valid_from="2026-09-01", importance=0.5, confidence=0.7)
            with mock.patch.dict("os.environ", {"ENGRAM_METAMEM": "1"}):
                on = retrieve.retrieve(con, "我哪天生日")
            with mock.patch.dict("os.environ", {"ENGRAM_METAMEM": "0"}):
                off = retrieve.retrieve(con, "我哪天生日")
            self.assertTrue(any("2000年5月20日生" in h["content"] for h in on.hits))
            self.assertFalse(any("2000年5月20日生" in h["content"] for h in off.hits))
            con.close()


class AliasBeforeRarity(unittest.TestCase):
    def test_alias_terms_enter_before_rarity_sort_and_survive_light_cut(self):
        import retrieve
        seen = {}

        def spy(con, terms):
            seen["terms"] = list(terms)
            return list(terms)
        with tempfile.TemporaryDirectory() as d:
            con = connect(Path(d) / "t.db")
            with mock.patch.dict("os.environ", {"ENGRAM_METAMEM": "1"}), mock.patch.object(retrieve, "by_rarity", spy):
                retrieve.retrieve(con, "我哪天生日")
            con.close()
        self.assertIn("日生", seen.get("terms", []))


class AtomicCandidates(unittest.TestCase):
    OWN = "小林周三要去牙科复诊；小林更喜欢用纸质笔记本；下个月打算把自行车卖掉，价格大概两百块左右，到时候看情况再说"

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.n = 0

    def add(self, subject, content):
        self.n += 1
        eid = add_evidence(self.con, source_kind="chat", source_actor=subject, source_locator=f"conv:t#msg:{self.n}",
                           content=content, occurred_at="2026-09-01")
        promote(self.con, subject=subject, fact_type="semantic", content=content, evidence_ids=[eid],
                valid_from="2026-09-01", importance=0.5, confidence=0.7)

    def test_only_users_own_facts(self):
        self.add(U, self.OWN)
        self.add(C, self.OWN.replace("小林", "助手"))
        got = SA.candidates(self.con)
        self.assertEqual([r["subject"] for r in got], [U])

    def test_other_speakers_raw_line_excluded(self):
        self.add(U, f"{NC}（2026-09-01 10:00）：" + self.OWN)
        self.assertEqual(SA.candidates(self.con), [])

    def test_own_speaker_line_kept(self):
        self.add(U, f"{NU}（2026-09-01 10:00）：" + self.OWN)
        self.assertEqual(len(SA.candidates(self.con)), 1)

    def test_multi_turn_transcript_excluded(self):
        t = (f"{NU}（2026-09-01 10:00）：今天去了牙科；顺便买了纸质笔记本\n"
             f"{NC}（2026-09-01 10:01）：好的，记下了；下次提醒你复诊；还有别的吗")
        self.add(U, t)
        self.assertEqual(SA.candidates(self.con), [])
        self.assertTrue(SA.is_transcript(t))

    def test_work_report_excluded(self):
        t = "已完成路由重构；修复了 split.py 的报错；部署到测试环境并且全部通过回归测试"
        self.add(U, t)
        self.assertTrue(SA.is_work_report(t))
        self.assertEqual(SA.candidates(self.con), [])

    def test_pasted_task_prompts_for_other_ais_excluded(self):
        for t in ("返修 util.ts:12–15：读取游标必须和写入走同一条队列；补一个并发测试；不要改其他代码；完成后跑编译",
                  "先备份配置，只摘掉旧 thread 的绑定；不要删除旧文件；改完只重启一次服务；群里的保持不动"):
            self.assertTrue(SA.is_work_report(t), t)
        self.assertFalse(SA.is_work_report(self.OWN))

    def test_prompt_has_rules_and_fictional_examples_only(self):
        p = SA.build_prompt(self.OWN)
        for k in ("脱离原文也看得懂", "不要为了凑数拆", "例2", "atoms"):
            self.assertIn(k, p)
        self.assertIn(self.OWN, p)


class TimelineWindow(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        self.n = 0

    def old(self, content, when):
        self.n += 1
        eid = add_evidence(self.con, source_kind="chat", source_actor=U, source_locator=f"conv:t#msg:{self.n}",
                           content=content, occurred_at=when)
        fid = promote(self.con, subject=U, fact_type="semantic", content=content, evidence_ids=[eid],
                      valid_from=when, importance=0.5, confidence=0.7)
        record_class(self.con, fid, "semantic", 0.8, "extract-llm-v1")

    def cands(self, new, when):
        with mock.patch.dict("os.environ", {"ENGRAM_TIMELINE": "1", "ENGRAM_ENV_FILE": "/nonexistent"}):
            return [c["content"] for c in timeline.candidates(self.con, U, new, when)]

    def test_same_day_old_is_not_a_candidate_but_earlier_day_is(self):
        self.old("小林手机用的是蓝色那台", "2026-09-10 08:00")
        self.old("小林手机屏幕有个裂缝", "2026-09-09 08:00")
        with mock.patch.object(timeline, "keywords", return_value=["小林手机"]):
            got = self.cands("小林手机换了新的", "2026-09-10 20:00")
        self.assertEqual(got, ["小林手机屏幕有个裂缝"])

    def test_near_duplicate_dropped_but_changed_number_kept(self):
        self.old("小林最后一天上课是9月23日", "2026-09-01")
        self.assertEqual(self.cands("小林最后一天上课是9月24日", "2026-09-05"), ["小林最后一天上课是9月23日"])
        self.assertEqual(self.cands("小林最后一天上课是9月23日", "2026-09-05"), [])

    def test_near_duplicate_function(self):
        self.assertTrue(timeline.near_duplicate("小林周三去牙科复诊", "小林周三去牙科复诊了"))
        self.assertFalse(timeline.near_duplicate("小林周三去牙科复诊", "小林周五去牙科复诊"))

    def test_prompt_has_refinement_and_same_day_rules(self):
        for k in ("enriches", "updates", "永远不会因为后来的事而变假", "虚构"):
            self.assertIn(k, timeline.PROMPT)


if __name__ == "__main__":
    unittest.main()
