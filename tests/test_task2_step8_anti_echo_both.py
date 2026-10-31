"""task2-08：防回声——入库（X 的证据只能产出 X / us 的事实）与召回（只有转述的降权、标注、不强化）两处都要有。
模型调用全 mock；没有开关，默认就是开。"""
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cards  # noqa: E402
import consolidate  # noqa: E402
import echo  # noqa: E402
import hybrid  # noqa: E402
import llm  # noqa: E402
import mcp_server  # noqa: E402
import retrieve  # noqa: E402
import store.store as S  # noqa: E402
import unified  # noqa: E402
from extract import from_evidence as FE  # noqa: E402
from store.store import add_evidence, connect, promote, reinforce  # noqa: E402

ENV = {"ENGRAM_ENV_FILE": "/nonexistent"}


def mkfact(con, subject, content, actors, *, status="candidate", pinned=0, role="support", layer="active",
           importance=0.6):
    """绕过 promote 的入库兜底，造「库里已有的」事实（用来测召回侧）。actors：挂几条证据、各是谁说的。"""
    ts = "2026-09-01T00:00:00+00:00"
    cur = con.execute(
        """INSERT INTO facts (ext_id, subject, fact_type, content, grams, content_sha256, status, valid_from,
             user_pinned, importance, confidence, layer, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (f"fact_{uuid.uuid4().hex[:12]}", subject, "episodic", content, S.bigrams(content), S.sha256(content),
         status, ts, pinned, importance, 0.8, layer, ts, ts))
    fid = cur.lastrowid
    for a in actors:
        ev = add_evidence(con, source_kind="chat", source_actor=a, source_locator=f"L{uuid.uuid4().hex[:8]}",
                          content=f"{a}说：{content}", verification_status="verified")
        con.execute("INSERT INTO fact_evidence (fact_id, evidence_id, role) VALUES (?,?,?)", (fid, ev, role))
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


# ───────────────────────── 入库 ─────────────────────────
class IngestRule(unittest.TestCase):
    def test_owner_ok_matrix(self):
        for actor in ("tester", "claude", "gpt", "agentc"):
            for subj in ("tester", "claude", "gpt", "agentc"):
                self.assertEqual(echo.owner_ok(actor, subj), actor == subj, (actor, subj))
            self.assertFalse(echo.owner_ok(actor, "us"))        # task2-09：单个说话人证明不了「两个人的事」
        self.assertTrue(echo.owner_ok(None, "tester"))          # 说话人未知：不判，交给 promote 兜底

    def test_check_in_from_evidence(self):
        def f(subj):
            return {"subject": subj, "type": "episodic", "content": "小测考某大学计算机", "quote": "我考某大学，报计算机", "src": 0,
                    "confidence": 0.9}
        for actor, ok_subj, bad_subj in (("tester", ("tester", "us"), ("claude", "gpt", "agentc")),
                                         ("claude", ("claude", "us"), ("tester", "gpt", "agentc")),
                                         ("gpt", ("gpt", "us"), ("tester", "claude", "agentc"))):
            chunk = [{"id": 1, "source_actor": actor, "content": "小测（2026-10-01 10:00）：我考某大学，报计算机技术",
                      "occurred_at": "2026-10-01T02:00:00+00:00"}]
            for s in (ok_subj[0],):
                got, why = FE.check(f(s), chunk)
                self.assertIsNotNone(got, (actor, s, why))
            for s in bad_subj + ("us",):
                if s == "us" and actor == "tester":
                    continue                       # 她单方的 us、内容以「用户」为主语 → 落回 user（见 step9）
                got, why = FE.check(f(s), chunk)
                self.assertIsNone(got, (actor, s))
                self.assertTrue(why.startswith("subject-echo") or why.startswith("subject-us"), why)

    def test_prompts_state_the_rule(self):
        self.assertIn("subject 只能是 quote 所在那句话的说话人本人", FE.PROMPT)
        self.assertIn("us 要双方的原话", FE.PROMPT)
        self.assertIn("只能是说话人本人", unified.PROMPT)


class PromoteGuard(Base):
    """store.promote 是所有入库路径（from_evidence/from_channel/from_digest 走 SQL 除外见审计）的最后一道。"""

    def p(self, subject, actors, **kw):
        ids = [self.ev(a) for a in actors]
        return promote(self.con, subject=subject, fact_type="episodic",
                       content=f"{subject}的事{uuid.uuid4().hex[:6]}", evidence_ids=ids, **kw)

    def test_blocks_other_speakers_words(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for subj, actor in (("tester", "claude"), ("gpt", "claude"), ("agentc", "claude"),
                                ("claude", "tester"), ("gpt", "tester")):
                self.assertIsNone(self.p(subj, [actor]), (subj, actor))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)
        self.assertIn("[anti-echo]", err.getvalue())

    def test_allows_owner_and_us(self):
        self.assertIsNotNone(self.p("tester", ["tester"]))
        self.assertIsNotNone(self.p("claude", ["claude"]))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertIsNone(self.p("us", ["claude"]))                 # task2-09：us 要双方原话
            self.assertIsNone(self.p("us", ["tester"]))
        self.assertIsNotNone(self.p("us", ["claude", "tester"]))
        self.assertIsNotNone(self.p("tester", ["tester", "claude"]))     # 有本人一条即可
        self.assertIsNotNone(self.p("project:sims", ["claude"]))           # 非人物主体不受这条约束

    def test_confirmed_or_pinned_exempt_corrections_path(self):
        # 她自己的纠正（corrections.py：confirmed + user_pinned）不拦
        self.assertIsNotNone(self.p("tester", ["claude"], status="confirmed", user_pinned=True))
        self.assertIsNotNone(self.p("tester", ["claude"], user_pinned=True))
        self.assertIsNotNone(self.p("tester", ["claude"], status="confirmed"))

    def test_no_evidence_not_judged(self):
        self.assertIsNotNone(promote(self.con, subject="tester", fact_type="semantic", content="巩固来的抽象"))

    def test_no_flag_can_turn_it_off(self):
        env = {k: "1" for k in ("ENGRAM_METAMEM", "ENGRAM_LAYERED_RETRIEVE", "ENGRAM_CONSOLIDATE",
                                "ENGRAM_INVALIDATE_CHAIN", "ENGRAM_TIMELINE", "ENGRAM_UNIFIED_GATE")}
        with mock.patch.dict(os.environ, {**env, "ENGRAM_ANTI_ECHO": "0", "ENGRAM_ECHO": "0"}), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(self.p("tester", ["claude"]))

    def test_every_fact_creating_module_is_known(self):
        """新增一个会 promote 的模块就红：逼着先做防回声审计。"""
        callers = set()
        for path in list(ROOT.glob("*.py")) + list((ROOT / "extract").glob("*.py")):
            if "promote(" in path.read_text(encoding="utf-8") and path.name != "echo.py":
                callers.add(str(path.relative_to(ROOT)))
        audited = {"store/store.py", "consolidate.py", "corrections.py", "distill_self.py", "ingest.py",
                   "ingest_mailbox.py", "unified.py", "extract/from_channel.py", "extract/from_evidence.py",
                   # 任务三：新增的两个入口，防回声/数字校验由 tests/test_ingest_slim.py 逐条覆盖
                   "extract/distill_slice.py", "remember.py"}
        callers.discard("store/store.py")
        self.assertLessEqual(callers, audited - {"store/store.py"}, callers - audited)


class UnifiedGate(Base):
    def test_llm_cannot_refile_claudes_words_under_tester(self):
        body = "小测下周三要去己市考试，地点在某理工大学大学"
        content = f"助理甲（2026-10-01 10:00）：{body}"
        reply = json.dumps({"promote": True, "fact_type": "episodic", "subject": "tester", "fact_key": None,
                            "slots": ["专名"], "confidence": 0.9, "quote": "下周三要去己市考试"}, ensure_ascii=False)
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_UNIFIED_GATE": "1"}, clear=True), \
                mock.patch.object(llm, "chat", mock.Mock(return_value=reply)):
            u = unified.decide(self.con, subject="claude", content=content,
                               evidence_rows=[{"verification_status": "verified"}])
        self.assertEqual(u.subject, "claude")
        self.assertEqual(u.source, "rules")
        self.assertIn("防回声", u.reason)

    def test_llm_may_choose_us_and_self(self):
        content = "助理甲（2026-10-01 10:00）：我和小测说好了周末一起整理相册，这是约定"
        both = [{"verification_status": "verified", "source_actor": "claude"},
                {"verification_status": "verified", "source_actor": "tester"}]
        one = [{"verification_status": "verified", "source_actor": "claude"}]
        for subj, rows, want in (("us", both, ("llm", "us")), ("claude", one, ("llm", "claude")),
                                 ("us", one, ("rules", "claude"))):          # task2-09：us 只有一方证据 → 退回规则
            reply = json.dumps({"promote": True, "fact_type": "procedural", "subject": subj, "fact_key": None,
                                "slots": ["规矩"], "confidence": 0.9, "quote": "周末一起整理相册"}, ensure_ascii=False)
            with mock.patch.dict(os.environ, {**ENV, "ENGRAM_UNIFIED_GATE": "1"}, clear=True), \
                    mock.patch.object(llm, "chat", mock.Mock(return_value=reply)):
                u = unified.decide(self.con, subject="claude", content=content, evidence_rows=rows)
            self.assertEqual((u.source, u.subject), want, (subj, len(rows)))


# ───────────────────────── 召回 ─────────────────────────
class Provenance(Base):
    def test_signal(self):
        e1 = mkfact(self.con, "tester", "回声一号", ["claude"])
        e2 = mkfact(self.con, "gpt", "回声二号", ["claude", "tester"])
        e3 = mkfact(self.con, "agentc", "回声三号", ["claude"])
        own = mkfact(self.con, "tester", "本人一号", ["tester"])
        mixed = mkfact(self.con, "tester", "本人加转述", ["claude", "tester"])
        us = mkfact(self.con, "us", "我们一号", ["claude"])
        me = mkfact(self.con, "claude", "我自己的事", ["claude"])
        noev = mkfact(self.con, "tester", "没证据的", [])
        conf = mkfact(self.con, "tester", "确认过的转述", ["claude"], status="confirmed")
        pin = mkfact(self.con, "tester", "钉住的转述", ["claude"], pinned=1)
        contra = mkfact(self.con, "tester", "本人只是反证", ["claude"])
        ev = self.ev("tester")
        self.con.execute("INSERT INTO fact_evidence (fact_id, evidence_id, role) SELECT id, ?, 'contradict' "
                         "FROM facts WHERE ext_id=?", (ev, contra))
        self.con.commit()
        got = echo.echo_sources(self.con, [e1, e2, e3, own, mixed, us, me, noev, conf, pin, contra])
        self.assertEqual(set(got), {e1, e2, e3, contra, us})     # task2-09：只有一方证据的 us 也是转述
        self.assertEqual(got[e1], ["claude"])
        self.assertEqual(echo.label(got[e1]), "〔转述·来自助理甲〕")
        self.assertEqual(echo.label(["claude", "gpt"]), "〔转述·来自助理甲、助手乙〕")
        self.assertEqual(echo.label([]), "")
        # e2：gpt 的事只有助手和用户说过，没有 gpt 本人 → 也是转述
        self.assertEqual(echo.echo_sources(self.con, [e2])[e2], ["claude", "tester"])
        self.assertEqual(echo.tag_for(self.con, own), "")

    def test_weight_matches_consolidate(self):
        self.assertEqual(echo.ECHO_WEIGHT, consolidate.ECHO_WEIGHT)
        spec = importlib.util.spec_from_file_location("h", ROOT / "hooks_engram_recall.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        self.assertEqual(hook.ECHO_W, echo.ECHO_WEIGHT)


class HybridRanking(Base):
    def test_finalize_downweights_echo_but_not_exempt(self):
        def H(ext, **kw):
            return dict({"ext_id": ext, "subject": "tester", "content": ext, "status": "candidate",
                         "user_pinned": 0, "layer": "active", "score": 1.0}, **kw)
        m = hybrid.rrf_merge({"fts": [H("e"), H("a"), H("b")]})            # e 排第一
        out = hybrid.finalize(m, echo={"e": ["claude"]})
        self.assertEqual([h["ext_id"] for h in out], ["a", "b", "e"])
        self.assertEqual(out[-1]["echo_from"], ["claude"])
        self.assertEqual(out[0]["echo_from"], [])
        m = hybrid.rrf_merge({"fts": [H("e"), H("a")]})
        s_before = hybrid.finalize(m)[0]["hybrid_score"]
        m = hybrid.rrf_merge({"fts": [H("e"), H("a")]})
        s_after = [h for h in hybrid.finalize(m, echo={"e": ["claude"]}) if h["ext_id"] == "e"][0]["hybrid_score"]
        self.assertAlmostEqual(s_after / s_before, echo.ECHO_WEIGHT, places=3)

    def test_retrieve_end_to_end(self):
        e = mkfact(self.con, "tester", "星光湖那天小测说她很喜欢脚印会亮的海", ["claude"], importance=0.95)
        o = mkfact(self.con, "tester", "星光湖那天小测说她想再去一次看脚印会亮", ["tester"], importance=0.3)
        c = mkfact(self.con, "tester", "星光湖是小测钉住的约定要再去", ["claude"], status="confirmed", pinned=1,
                   importance=0.2)
        with mock.patch.dict(os.environ, ENV, clear=True), mock.patch.object(hybrid, "_vsearch", return_value=[]):
            r = retrieve.retrieve(self.con, "上次那个星光湖是哪年的事")
        by = {h["ext_id"]: h for h in r.hits}
        self.assertEqual(by[e]["echo_from"], ["claude"])
        self.assertEqual(by[o]["echo_from"], [])
        self.assertEqual(by[c]["echo_from"], [])                          # 钉住/确认豁免
        ids = [h["ext_id"] for h in r.hits]
        self.assertLess(ids.index(o), ids.index(e))                        # 本人的排在转述前面，尽管转述激活度更高


class Cards(Base):
    def test_discover_and_expand_carry_and_downweight(self):
        e = mkfact(self.con, "tester", "星光湖转述版本很长很长很长的一段话", ["claude"], importance=0.99)
        o = mkfact(self.con, "tester", "星光湖本人说的一段话", ["tester"], importance=0.2)
        cards.build(self.con, quiet=True)
        hits = cards.discover(self.con, "星光湖", limit=5)
        self.assertEqual([h["ext_id"] for h in hits], [o, e])
        self.assertEqual(hits[1]["echo_from"], ["claude"])
        rows = cards.expand(self.con, [h["fact_id"] for h in hits])
        self.assertEqual({r["ext_id"]: r["echo_from"] for r in rows}, {o: [], e: ["claude"]})


class NoReinforce(Base):
    def count(self, ext):
        return self.con.execute("SELECT access_count FROM facts WHERE ext_id=?", (ext,)).fetchone()[0]

    def test_reinforce_skips_echo_only(self):
        e = mkfact(self.con, "tester", "回声事实", ["claude"])
        o = mkfact(self.con, "tester", "本人事实", ["tester"])
        c = mkfact(self.con, "tester", "确认过的转述", ["claude"], status="confirmed")
        for x in (e, o, c):
            reinforce(self.con, x)
            reinforce(self.con, x)
        self.assertEqual((self.count(e), self.count(o), self.count(c)), (0, 2, 2))
        self.assertIsNotNone(self.con.execute("SELECT last_accessed FROM facts WHERE ext_id=?", (e,)).fetchone()[0])

    def test_mcp_expand_labels_and_does_not_reinforce(self):
        e = mkfact(self.con, "tester", "助理甲转述的小测的事", ["claude"])
        out = mcp_server.handle("tools/call", {"name": "engram_expand", "arguments": {"ext_id": e}}, self.con)
        text = out["content"][0]["text"]
        self.assertTrue(text.startswith("〔转述·来自助理甲〕"), text)
        self.assertEqual(self.count(e), 0)

    def test_mcp_recall_labels_only_echo(self):
        e = mkfact(self.con, "tester", "星光湖转述的一段话", ["claude"])
        o = mkfact(self.con, "tester", "星光湖本人说的一段话", ["tester"])
        text = mcp_server.handle("tools/call", {"name": "engram_recall", "arguments": {"query": "星光湖"}},
                                 self.con)["content"][0]["text"]
        lines = [ln for ln in text.splitlines() if re.match(r"\d+\. ", ln)]
        tagged = [ln for ln in lines if "〔转述·来自助理甲〕" in ln]
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(tagged), 1)
        self.assertIn("转述的一段话", tagged[0])


class HookInjection(Base):
    """整条 hook 走一遍（rerank / newer 模型调用 mock）：精排看到标注、注入带标注、转述排在本人后面。"""

    def run_hook(self, scores):
        spec = importlib.util.spec_from_file_location("hk", ROOT / "hooks_engram_recall.py")
        hk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hk)
        hk.ENGRAM = str(ROOT)
        hk.LOG = str(Path(self.dir.name) / "hook.log")
        import rerank
        seen = {}

        def fake_judge(query, cands):
            seen["cands"] = list(cands)
            return {"need": True, "scores": scores(cands)}

        out = io.StringIO()
        db = str(Path(self.dir.name) / "t.db")
        with mock.patch.dict(os.environ, ENV, clear=True), \
                mock.patch.object(hybrid, "_vsearch", return_value=[]), \
                mock.patch.object(rerank, "judge", fake_judge), \
                mock.patch.object(sys.modules["newer"] if "newer" in sys.modules else __import__("newer"),
                                  "superseded_by", return_value=None), \
                mock.patch.object(S.connect, "__defaults__", (db,)), \
                mock.patch.object(sys, "stdin", io.StringIO(json.dumps({"prompt": "你还记得上次那个星光湖是哪年的事吗"}))), \
                contextlib.redirect_stdout(out):
            hk.main()
        return out.getvalue(), seen.get("cands", [])

    def test_label_rerank_and_order(self):
        mkfact(self.con, "tester", "星光湖那天小测说她想再去一次看脚印会亮的海滩", ["claude"], importance=0.95)
        mkfact(self.con, "tester", "星光湖那天小测自己说脚印会亮她很开心", ["tester"], importance=0.3)
        text, cands = self.run_hook(lambda cands: [9 for _ in cands])
        self.assertTrue(any(c.startswith("〔转述·来自助理甲〕") for c in cands), cands)
        self.assertEqual(sum(1 for c in cands if c.startswith("〔转述")), 1)
        pieces = [ln for ln in text.splitlines() if ln.startswith("· [")]
        self.assertEqual(len(pieces), 2, text)
        self.assertNotIn("〔转述", pieces[0])                      # 本人的在前、不带标
        self.assertIn("〔转述·来自助理甲〕", pieces[1])
        self.assertIn("〔转述〕＝别人转述的话", text)

    def test_rerank_order_downweights_equal_scores(self):
        mkfact(self.con, "tester", "星光湖转述版本小测想去看脚印会亮的海", ["claude"], importance=0.95)
        mkfact(self.con, "tester", "星光湖本人版本小测说脚印会亮真好看", ["tester"], importance=0.3)
        # 精排对转述打 10、对本人打 8：转述 10/3≈3.3 < 8，本人排前；阈值仍按原分，转述 10 分照样能进
        text, cands = self.run_hook(lambda cands: [10 if c.startswith("〔转述") else 8 for c in cands])
        pieces = [ln for ln in text.splitlines() if ln.startswith("· [")]
        self.assertEqual(len(pieces), 2, text)
        self.assertNotIn("〔转述", pieces[0])



def setUpModule():
    # 这个文件测的是旧管线（值得记闸关）的行为：不管环境变量 ENGRAM_VALUE_GATE 开没开，都按闸关跑（闸开的行为在 test_value_gate*.py 里测）
    import os
    from unittest import mock
    p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "0"})
    p.start()
    unittest.addModuleCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
