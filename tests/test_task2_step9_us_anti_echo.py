"""task2-09：us 防回声（入库要双方原话、召回单方 us 当转述）+ 数字校验。模型调用全 mock；没有开关。"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cards  # noqa: E402
import distill_self as D  # noqa: E402
import echo  # noqa: E402
import hybrid  # noqa: E402
import llm  # noqa: E402
import mcp_server  # noqa: E402
import store.store as S  # noqa: E402
from extract import from_evidence as FE  # noqa: E402
from store.store import add_evidence, connect, promote, reinforce  # noqa: E402


def mkfact(con, subject, content, actors, *, status="candidate", pinned=0, importance=0.6):
    ts = "2026-09-01T00:00:00+00:00"
    cur = con.execute(
        """INSERT INTO facts (ext_id, subject, fact_type, content, grams, content_sha256, status, valid_from,
             user_pinned, importance, confidence, layer, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"fact_{uuid.uuid4().hex[:12]}", subject, "episodic", content, S.bigrams(content), S.sha256(content),
         status, ts, pinned, importance, 0.8, "active", ts, ts))
    fid = cur.lastrowid
    for a in actors:
        ev = add_evidence(con, source_kind="chat", source_actor=a, source_locator=f"L{uuid.uuid4().hex[:8]}",
                          content=f"{a}说：{content}", verification_status="verified")
        con.execute("INSERT INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,?)", (fid, ev, "support"))
    con.commit()
    return con.execute("SELECT ext_id FROM facts WHERE id=?", (fid,)).fetchone()[0]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def ev(self, actor, body="一句原话内容足够长"):
        return add_evidence(self.con, source_kind="chat", source_actor=actor,
                            source_locator=f"L{uuid.uuid4().hex[:8]}", content=body, verification_status="verified")


class UsRule(unittest.TestCase):
    def test_us_ok_needs_tester_and_another(self):
        self.assertTrue(echo.us_ok({"tester", "claude"}))
        self.assertTrue(echo.us_ok(["tester", "gpt"]))
        for a in ({"claude"}, {"tester"}, {"claude", "gpt"}, {"gpt", "agentc"}, set(), None):
            self.assertFalse(echo.us_ok(a), a)

    def test_resolve_us_falls_back_to_speaker_or_drops(self):
        self.assertEqual(echo.resolve_us("claude", "助理甲答应陪她考完"), "claude")
        self.assertEqual(echo.resolve_us("tester", "小测今天去了XX"), "tester")
        self.assertIsNone(echo.resolve_us("claude", "小测和助理甲约定每天打卡"))
        self.assertIsNone(echo.resolve_us("claude", "小测说她喜欢XX"))       # 助手转述她的事：不落回 claude
        self.assertIsNone(echo.resolve_us(None, "助理甲……"))

    def test_evidence_allows_us(self):
        con = connect(Path(tempfile.mkdtemp()) / "t.db")
        e = lambda a: add_evidence(con, source_kind="chat", source_actor=a, source_locator="L" + uuid.uuid4().hex[:6],
                                   content="x" * 12, verification_status="verified")
        c, l = e("claude"), e("tester")
        self.assertFalse(echo.evidence_allows(con, "us", [c])[0])
        self.assertFalse(echo.evidence_allows(con, "us", [l])[0])
        self.assertTrue(echo.evidence_allows(con, "us", [c, l])[0])
        self.assertTrue(echo.evidence_allows(con, "us", [])[0])             # 没证据不判（同旧规矩）
        self.assertIn("subject-us", echo.evidence_allows(con, "us", [c])[1])


CHUNK = [
    {"id": 1, "source_actor": "claude", "content": "我：约定好了，周末一起整理相册，你来选照片，我来排版。",
     "occurred_at": "2026-10-01T02:00:00+00:00"},
    {"id": 2, "source_actor": "tester", "content": "好，周末就这么定了，我来选照片。",
     "occurred_at": "2026-10-01T02:01:00+00:00"},
]


def f(subj="us", content="小测和助理甲约定周末一起整理相册", **kw):
    d = {"subject": subj, "type": "procedural", "content": content, "quote": "周末一起整理相册", "src": 0, "confidence": 0.9}
    d.update(kw)
    return d


class CheckUs(unittest.TestCase):
    def test_us_with_both_sides_kept_and_links_both(self):
        got, why = FE.check(f(quote2="周末就这么定了", src2=1), CHUNK)
        self.assertIsNotNone(got, why)
        self.assertEqual(got["subject"], "us")
        self.assertEqual((got["evidence_id"], got["evidence_id2"]), (1, 2))

    def test_us_claude_only_is_dropped_or_becomes_claude(self):
        got, why = FE.check(f(), CHUNK)                                     # 没给 quote2
        self.assertIsNone(got)
        self.assertTrue(why.startswith("subject-us"), why)
        got, why = FE.check(f(content="助理甲提议周末一起整理相册"), CHUNK)    # 以助手为主语 → 落回 claude
        self.assertEqual(got["subject"], "claude")
        self.assertIsNone(got["evidence_id2"])

    def test_us_second_quote_from_same_speaker_does_not_count(self):
        chunk = [CHUNK[0], dict(CHUNK[0], id=3, content="我：我来排版，你来选照片。")]
        got, why = FE.check(f(quote2="我来排版", src2=1), chunk)
        self.assertIsNone(got)
        self.assertTrue(why.startswith("subject-us"))

    def test_forged_quote2_not_in_evidence_does_not_count(self):
        got, why = FE.check(f(quote2="她其实答应了这件事情", src2=1), CHUNK)
        self.assertIsNone(got)

    def test_user_solo_us_falls_back_to_tester_only_when_she_is_the_subject(self):
        chunk = [{"id": 2, "source_actor": "tester", "content": "我周末要去整理相册", "occurred_at": "x"}]
        got, _ = FE.check({"subject": "us", "type": "episodic", "content": "小测周末要整理相册", "quote": "周末要去整理相册",
                           "src": 0, "confidence": 0.9}, chunk)
        self.assertEqual(got["subject"], "tester")
        got, why = FE.check({"subject": "us", "type": "episodic", "content": "周末要整理相册", "quote": "周末要去整理相册",
                             "src": 0, "confidence": 0.9}, chunk)
        self.assertIsNone(got)

    def test_prompts_mention_both_sides(self):
        self.assertIn("quote2", FE.PROMPT)
        self.assertIn("src2", FE.PROMPT)
        self.assertIn("绝对不要写 tester / gpt / agentc / us", D.PROMPT)
        self.assertIn("提议、打算不要写成「约定」", D.PROMPT)
        self.assertIn("年份、日期、数字", D.PROMPT)


class NumberCheck(unittest.TestCase):
    def test_unsupported_numbers(self):
        self.assertEqual(FE.numbers_unsupported("事发在2023年6月26日", "6/26 收到第一封"), ["2023"])
        self.assertEqual(FE.numbers_unsupported("在12月11-12日交叉", "她的(1211)和我的(1212)错开了"), ["12", "11", "12"])
        self.assertEqual(FE.numbers_unsupported("知识截止到2025年5月", "我的知识截止在今年五月"), ["2025"])
        self.assertEqual(FE.numbers_unsupported("在8月13日完成", "8 月 14 号 19:00"), ["13"])

    def test_supported_numbers_pass(self):
        self.assertEqual(FE.numbers_unsupported("2026-06 底停机", "是 2026-06 底那次停机"), [])
        self.assertEqual(FE.numbers_unsupported("8月6日下午7:24", "8/6 下午 07:24 掉到"), [])      # 前导零
        self.assertEqual(FE.numbers_unsupported("换线6次", "换线六次"), [])                          # 个位数↔中文数字
        self.assertEqual(FE.numbers_unsupported("没有数字的一句话", "任何原话"), [])
        self.assertEqual(FE.numbers_unsupported("302 专业课", "我报 302"), [])

    def test_check_drops_invented_year(self):
        chunk = [{"id": 1, "source_actor": "tester", "content": "第一封（6/26）说的是政策问题",
                  "occurred_at": "x"}]
        got, why = FE.check({"subject": "tester", "type": "episodic", "content": "小测在2023年6月26日收到第一封通知",
                             "quote": "第一封（6/26）说的是政策问题", "src": 0, "confidence": 0.9}, chunk)
        self.assertIsNone(got)
        self.assertTrue(why.startswith("number-unsupported"), why)
        got, why = FE.check({"subject": "tester", "type": "episodic", "content": "小测在6月26日收到第一封通知",
                             "quote": "第一封（6/26）说的是政策问题", "src": 0, "confidence": 0.9}, chunk)
        self.assertIsNotNone(got, why)


class PromoteBackstop(Base):
    def p(self, subject, actors, **kw):
        ids = [self.ev(a) for a in actors]
        return promote(self.con, subject=subject, fact_type="episodic",
                       content=f"{subject}的事{uuid.uuid4().hex[:6]}", evidence_ids=ids, **kw)

    def test_us_needs_both(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertIsNone(self.p("us", ["claude"]))
            self.assertIsNone(self.p("us", ["tester"]))
            self.assertIsNone(self.p("us", ["claude", "gpt"]))
        self.assertIn("[anti-echo]", err.getvalue())
        self.assertIsNotNone(self.p("us", ["claude", "tester"]))
        self.assertIsNotNone(self.p("us", ["gpt", "tester"]))

    def test_confirmed_pinned_exempt(self):
        self.assertIsNotNone(self.p("us", ["claude"], status="confirmed"))
        self.assertIsNotNone(self.p("us", ["claude"], user_pinned=True))


class RecallUs(Base):
    def test_us_one_sided_is_echo_only(self):
        one = mkfact(self.con, "us", "我们一号", ["claude"])
        both = mkfact(self.con, "us", "我们二号", ["claude", "tester"])
        solo = mkfact(self.con, "us", "我们三号", ["tester"])
        conf = mkfact(self.con, "us", "我们四号", ["claude"], status="confirmed")
        pin = mkfact(self.con, "us", "我们五号", ["claude"], pinned=1)
        got = echo.echo_sources(self.con, [one, both, solo, conf, pin])
        self.assertEqual(set(got), {one, solo})
        self.assertEqual(echo.label(got[one]), "〔转述·来自助理甲〕")
        self.assertEqual(echo.label(got[solo]), "〔转述·来自小测〕")

    def test_claude_subject_from_her_words_is_echo_only(self):
        """更早的缺口：她说的话被记在 claude 名下——召回时也当转述。"""
        her = mkfact(self.con, "claude", "助理甲被她说成XX", ["tester"])
        mine = mkfact(self.con, "claude", "助理甲自己做了XX", ["claude"])
        mixed = mkfact(self.con, "claude", "助理甲和她都说过XX", ["claude", "tester"])
        got = echo.echo_sources(self.con, [her, mine, mixed])
        self.assertEqual(set(got), {her})
        self.assertEqual(echo.label(got[her]), "〔转述·来自小测〕")

    def test_hybrid_downweights_us_one_sided(self):
        one = mkfact(self.con, "us", "整理相册约定一号", ["claude"])
        both = mkfact(self.con, "us", "整理相册约定二号", ["claude", "tester"])
        src = echo.echo_sources(self.con, [one, both])                       # hybrid.run 里就是这样取转述信号的
        self.assertEqual(set(src), {one})

        def H(ext):
            return {"ext_id": ext, "subject": "us", "content": ext, "status": "candidate",
                    "user_pinned": 0, "layer": "active", "score": 1.0}
        m = hybrid.rrf_merge({"fts": [H(one), H(both)]})                      # one 排第一
        out = {h["ext_id"]: h for h in hybrid.finalize(m, echo=src)}
        self.assertEqual(out[one]["echo_from"], ["claude"])
        self.assertEqual(out[both]["echo_from"], [])
        self.assertLess(out[one]["hybrid_score"], out[both]["hybrid_score"])  # 本来排第一，降权后落到后面

    def test_cards_and_mcp_tags(self):
        one = mkfact(self.con, "us", "整理相册约定一号", ["claude"])
        tags = mcp_server.echo_tags(self.con, [one]) if hasattr(mcp_server, "echo_tags") else {}
        self.assertEqual(tags.get(one), "〔转述·来自助理甲〕")
        src = echo.echo_sources(self.con, [one])
        self.assertEqual(src[one], ["claude"])

    def test_reinforce_does_not_promote_us_echo(self):
        one = mkfact(self.con, "us", "整理相册约定一号", ["claude"])
        both = mkfact(self.con, "us", "整理相册约定二号", ["claude", "tester"])
        reinforce(self.con, one)
        reinforce(self.con, both)
        n = lambda e: self.con.execute("SELECT access_count FROM facts WHERE ext_id=?", (e,)).fetchone()[0]
        self.assertEqual(n(one), 0)
        self.assertEqual(n(both), 1)

    def test_hook_label_uses_echo(self):
        spec = importlib_util_load()
        self.assertEqual(spec.echo_prefix({"echo_from": ["claude"]}), "〔转述·来自助理甲〕")
        self.assertEqual(spec.echo_prefix({"echo_from": []}), "")


def importlib_util_load():
    import importlib.util
    sp = importlib.util.spec_from_file_location("hook_t9", ROOT / "hooks_engram_recall.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m



def setUpModule():
    # 这个文件测的是旧管线（值得记闸关）的行为：不管环境变量 ENGRAM_VALUE_GATE 开没开，都按闸关跑（闸开的行为在 test_value_gate*.py 里测）
    import os
    from unittest import mock
    p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "0"})
    p.start()
    unittest.addModuleCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
