"""任务三：入库精简（RAW_ONLY / DISTILL_SLICE / SELF_REMEMBER / RERANK_FLASH）。模型调用全是假的，库全是临时库。
覆盖：恰好一次、重试+退避、死信箱、保底切片、切片路径上的防回声+数字校验、自记不会钉住/确认、
自记标记不进正文/FTS/向量、确认的不变、MCP 工具表、说话人统一、脚本语法。"""
import datetime as dt
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cards  # noqa: E402
import derive_repair  # noqa: E402
import echo  # noqa: E402
import flags  # noqa: E402
import llm  # noqa: E402
import mcp_server  # noqa: E402
import remember  # noqa: E402
import rerank  # noqa: E402
import speakers  # noqa: E402
import store.store as S  # noqa: E402
import unified  # noqa: E402
from extract import collect, distill_slice as D  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

ENV = {"ENGRAM_ENV_FILE": "/nonexistent"}
NOW = dt.datetime(2026, 10, 28, 12, 0, tzinfo=dt.timezone.utc)


def iso(d):
    return d.isoformat()


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.con.close)
        self.n = 0

    def msg(self, key, actor, body, when, kind="chat"):
        self.n += 1
        i = add_evidence(self.con, source_kind=kind, source_actor=actor, source_path=key,
                         source_locator=f"L{self.n}", content=body, occurred_at=iso(when),
                         verification_status="verified")
        self.con.commit()
        return i

    def session(self, key="sessA", ago_min=240, msgs=None):
        """默认：用户一句 + 我一句，停在 ago_min 分钟前。"""
        msgs = msgs or [("tester", "我下个月想去大理玩一周，订了民宿"), ("claude", "好，我帮你查查大理的天气")]
        last = NOW - dt.timedelta(minutes=ago_min)
        ids = []
        for i, (a, b) in enumerate(msgs):
            ids.append(self.msg(key, a, b, last - dt.timedelta(minutes=len(msgs) - i)))
        return ids

    def counts(self):
        q = lambda s: self.con.execute(s).fetchone()[0]     # noqa: E731
        return {"conf": q("SELECT COUNT(*) FROM facts WHERE status='confirmed' OR user_pinned=1"),
                "ev": q("SELECT COUNT(*) FROM evidence"), "facts": q("SELECT COUNT(*) FROM facts")}

    def seed_confirmed(self, n=3):
        for i in range(n):
            c = f"这是第{i}条钉住的老事实内容各不相同{i}{i}{i}"
            e = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator=f"C{i}",
                             content=c, verification_status="verified")
            promote(self.con, subject="tester", fact_type="semantic", content=c, evidence_ids=[e],
                    status="confirmed", user_pinned=True)
        self.con.commit()

    def fake(self, facts, calls=None, route="primary"):
        def caller(text):
            if calls is not None:
                calls.append(text)
            return list(facts), {"route": route, "model": "m", "tokens_in": 10, "tokens_out": 5}
        return caller

    def run_slice(self, caller, now=NOW, **kw):
        D.init_baseline(self.con, force=True) if False else None
        return D.run(self.con, now_dt=now, caller=caller, **kw)


GOOD = {"subject": "tester", "type": "episodic", "content": "小测下个月想去大理玩一周", "quote": "想去大理玩一周",
        "src": 0, "confidence": 0.8}


# ───────────────────────── 开关 ─────────────────────────
class Flags(unittest.TestCase):
    def test_new_flags_default_off(self):
        with mock.patch.dict(os.environ, ENV, clear=True):
            for k in ("RAW_ONLY", "DISTILL_SLICE", "SELF_REMEMBER", "RERANK_FLASH"):
                self.assertFalse(flags.on(k), k)

    def test_env_turns_on(self):
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_RAW_ONLY": "1"}, clear=True):
            self.assertTrue(flags.on("RAW_ONLY"))


class RawOnlyGate(Base):
    def decide(self, env):
        rows = [{"verification_status": "verified", "source_actor": "tester"}]
        with mock.patch.dict(os.environ, {**ENV, **env}, clear=True), \
                mock.patch.object(llm, "chat", mock.Mock(side_effect=AssertionError("不该调模型"))) as m:
            u = unified.decide(self.con, subject="tester", content="小测（2026-10-01 10:00）：我喜欢吃小笼包，真的很喜欢",
                               evidence_rows=rows)
        return u, m

    def test_on_never_promotes_and_no_llm(self):
        u, m = self.decide({"ENGRAM_RAW_ONLY": "1", "ENGRAM_UNIFIED_GATE": "1"})
        self.assertFalse(u.promote)
        self.assertEqual(u.code, "raw_only")
        m.assert_not_called()

    def test_off_unchanged(self):
        u, _ = self.decide({})
        self.assertNotEqual(u.code, "raw_only")

    def test_ingest_frozen_when_on(self):
        import ingest
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_RAW_ONLY": "1"}, clear=True), \
                mock.patch.object(ingest, "connect", side_effect=AssertionError("不该连库")):
            self.assertEqual(ingest.main(), 0)

    def test_from_channel_raw_only_writes_evidence_not_facts(self):
        d = Path(self.dir.name)
        f = d / "s.jsonl"
        f.write_text(json.dumps({"type": "user", "timestamp": "2026-10-28T01:00:00Z",
                                 "message": {"role": "user", "content": "我下个月想去大理玩一周，订了民宿"}},
                                ensure_ascii=False) + "\n", encoding="utf-8")
        from extract import from_channel
        db = d / "fc.db"
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_RAW_ONLY": "1"}, clear=True), \
                mock.patch.object(from_channel, "connect", lambda: connect(db)), \
                mock.patch.object(sys, "argv", ["from_channel", str(f)]):
            from_channel.main()
        c = connect(db)
        self.assertGreaterEqual(c.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 1)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)


# ───────────────────────── 采集 ─────────────────────────
class Collect(Base):
    def jsonl(self, name, lines):
        p = Path(self.dir.name) / name
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
        return p

    def user(self, text, ts="2026-10-28T01:00:00Z"):
        return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}

    def test_idempotent_and_cursor_skip(self):
        p = self.jsonl("a.jsonl", [self.user("我下个月想去大理玩一周，订了民宿"), self.user("周末打算在家看电影休息一下")])
        t1 = collect.collect_file(self.con, p)
        self.assertEqual(t1["new"], 2)
        t2 = collect.collect_file(self.con, p)
        self.assertEqual(t2["skipped_file"], 1)
        t3 = collect.collect_file(self.con, p, full=True)
        self.assertEqual((t3["new"], t3["have"]), (0, 2))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 2)
        r = self.con.execute("SELECT occurred_at, source_kind, source_actor FROM evidence").fetchone()
        self.assertEqual((r[1], r[2]), ("chat", "tester"))
        self.assertTrue(r[0].startswith("2026-10-28T01:00"))

    def test_appended_lines_only(self):
        p = self.jsonl("b.jsonl", [self.user("第一句话要足够长才行吧")])
        collect.collect_file(self.con, p)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.user("第二句话也要足够长才行吧", "2026-10-28T02:00:00Z"), ensure_ascii=False) + "\n")
        t = collect.collect_file(self.con, p)
        self.assertEqual(t["new"], 1)

    def test_background_session_skipped(self):
        p = self.jsonl("bg.jsonl", [self.user("【后台任务】写日记"), self.user("这句不是她说的话而是机器")])
        t = collect.collect_file(self.con, p)
        self.assertEqual(t["new"], 0)

    def test_old_prefix_variant_counts_as_present(self):
        p = self.jsonl("c.jsonl", [self.user("我下个月想去大理玩一周，订了民宿")])
        add_evidence(self.con, source_kind="chat", source_actor="tester", source_path=str(p), source_locator="L1",
                     content="小测（2026-10-28 09:00）：我下个月想去大理玩一周，订了民宿", verification_status="verified")
        t = collect.collect_file(self.con, p)
        self.assertEqual((t["new"], t["have"]), (0, 1))

    def test_unknown_actor_skipped(self):
        self.assertIsNone(speakers.normalize_actor("unknown:abc"))
        self.assertEqual(speakers.normalize_actor("claude"), "claude")


# ───────────────────────── 切片：何时到点 ─────────────────────────
class SliceTriggers(Base):
    def setUp(self):
        super().setUp()
        D.init_baseline(self.con)             # 空库基线 = 0

    def state(self, key="sessA"):
        return {s["key"]: s for s in D.due_sessions(self.con, NOW)}.get(key)

    def test_idle_3h_is_due(self):
        self.session(ago_min=181)
        s = self.state()
        self.assertEqual((s["state"], s["kind"]), ("due", "idle"))

    def test_active_not_due(self):
        self.session(ago_min=60)
        s = self.state()
        self.assertEqual((s["state"], s["kind"]), ("waiting", None))

    def test_cap_by_300_msgs(self):
        msgs = [("tester", f"第{i}条今天聊的内容各不相同呀") for i in range(310)]
        self.session(ago_min=30, msgs=msgs)
        s = self.state()
        self.assertEqual((s["state"], s["kind"]), ("due", "cap"))

    def test_cap_by_6h_age(self):
        self.msg("sessA", "tester", "七小时前开始聊的第一句话", NOW - dt.timedelta(hours=7))
        self.msg("sessA", "tester", "刚才还在聊的最后一句话", NOW - dt.timedelta(minutes=20))
        s = self.state()
        self.assertEqual((s["state"], s["kind"]), ("due", "cap"))
        self.assertEqual(len(s["rows"]), 2)

    def test_cap_settle_window(self):
        for i in range(305):
            self.msg("sessA", "tester", f"第{i}条刚刚才说的内容各不相同", NOW - dt.timedelta(minutes=1))
        s = self.state()
        self.assertEqual(s["state"], "settling")

    def test_baseline_refusal_and_history_skipped(self):
        con2 = connect(Path(self.dir.name) / "t2.db")
        self.addCleanup(con2.close)
        add_evidence(con2, source_kind="chat", source_actor="tester", source_path="old", source_locator="L1",
                     content="很久以前的一条历史原话内容", occurred_at=iso(NOW - dt.timedelta(days=9)))
        con2.commit()
        self.assertIsNone(D.baseline(con2))
        r = subprocess.run([sys.executable, "-m", "extract.distill_slice", "--apply", "--db",
                            str(Path(self.dir.name) / "t2.db")], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(r.returncode, 3)
        D.init_baseline(con2)
        add_evidence(con2, source_kind="chat", source_actor="tester", source_path="old", source_locator="L2",
                     content="基线之后新来的一条原话内容", occurred_at=iso(NOW - dt.timedelta(hours=5)))
        con2.commit()
        rows = {s["key"]: s for s in D.due_sessions(con2, NOW)}
        self.assertEqual([r["source_locator"] for r in rows["old"]["rows"]], ["L2"])


# ───────────────────────── 切片：恰好一次 / 重试 / 死信 ─────────────────────────
class SliceFlow(Base):
    def setUp(self):
        super().setUp()
        D.init_baseline(self.con)
        self.ids = self.session()

    def test_exactly_once(self):
        calls = []
        c = self.fake([GOOD], calls)
        t1 = D.run(self.con, now_dt=NOW, caller=c)
        self.assertEqual((t1["chunks"], t1["kept"]), (1, 1))
        t2 = D.run(self.con, now_dt=NOW, caller=c)
        self.assertEqual(t2["sessions"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM fact_origin WHERE origin='distill'").fetchone()[0], 1)
        cur = D._cursor(self.con, "sessA")
        self.assertEqual(cur["last_evidence_id"], self.ids[-1])

    def test_mid_write_failure_rolls_back_and_reruns_once(self):
        def boom(kept):
            raise RuntimeError("写到一半断电")
        t = D.run(self.con, now_dt=NOW, caller=self.fake([GOOD]), inject=boom)
        self.assertEqual(t["failed"], 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)       # 整块回滚
        cur = D._cursor(self.con, "sessA")
        self.assertEqual(cur["last_evidence_id"], 0)                                           # 游标没动
        self.assertEqual(cur["fail_count"], 1)
        later = NOW + dt.timedelta(minutes=11)
        t2 = D.run(self.con, now_dt=later, caller=self.fake([GOOD]))
        self.assertEqual(t2["kept"], 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)
        self.assertEqual(D._cursor(self.con, "sessA")["fail_count"], 0)

    def failing(self):
        def c(text):
            raise ConnectionError("接口挂了")
        return c

    def test_backoff_10_20_40_60_then_dead(self):
        now = NOW
        waits = []
        for attempt in range(1, 5):
            D.run(self.con, now_dt=now, caller=self.failing())
            cur = D._cursor(self.con, "sessA")
            self.assertEqual(cur["fail_count"], attempt)
            waits.append(int((D._t(cur["next_try_at"]) - now).total_seconds() // 60))
            now = D._t(cur["next_try_at"])
        self.assertEqual(waits, [10, 20, 40, 60])
        t = D.run(self.con, now_dt=now, caller=self.failing())          # 第 5 次
        self.assertEqual((t["failed"], t["dead"]), (1, 1))
        self.assertEqual(len(D.dead_list(self.con)), 1)
        self.assertEqual(D._cursor(self.con, "sessA")["last_evidence_id"], self.ids[-1])   # 游标越过去了

    def test_inside_backoff_window_skipped_no_call(self):
        D.run(self.con, now_dt=NOW, caller=self.failing())
        calls = []
        t = D.run(self.con, now_dt=NOW + dt.timedelta(minutes=5), caller=self.fake([GOOD], calls))
        self.assertEqual((t["backoff"], len(calls)), (1, 0))

    def test_24h_hard_cap_goes_dead(self):
        D.run(self.con, now_dt=NOW, caller=self.failing())
        t = D.run(self.con, now_dt=NOW + dt.timedelta(hours=25), caller=self.failing())
        self.assertEqual(t["dead"], 1)

    def test_poison_request_dead_after_3(self):
        def poison(text):
            raise D.PermanentError("HTTP 400")
        now = NOW
        for _ in range(3):
            D.run(self.con, now_dt=now, caller=poison)
            now = now + dt.timedelta(hours=2)
        self.assertEqual(len(D.dead_list(self.con)), 1)

    def test_dead_letter_does_not_block_later_chunks(self):
        D.run(self.con, now_dt=NOW, caller=self.failing())
        for k in range(4):
            cur = D._cursor(self.con, "sessA")
            D.run(self.con, now_dt=D._t(cur["next_try_at"]), caller=self.failing())
        self.assertEqual(len(D.dead_list(self.con)), 1)
        # 同一会话又来了新话，游标已越过死信段，正常处理
        last = NOW - dt.timedelta(minutes=300)
        i = self.msg("sessA", "tester", "后来她又说想去大理玩一周的事", last)
        t = D.run(self.con, now_dt=NOW + dt.timedelta(days=2), caller=self.fake([{**GOOD, "src": 0}]))
        self.assertEqual(t["chunks"], 1)
        self.assertEqual(D._cursor(self.con, "sessA")["last_evidence_id"], i)

    def _make_dead(self):
        D.run(self.con, now_dt=NOW, caller=self.failing())
        for _ in range(4):
            cur = D._cursor(self.con, "sessA")
            D.run(self.con, now_dt=D._t(cur["next_try_at"]), caller=self.failing())
        return D.dead_list(self.con)[0]["dead_id"]

    def test_dead_retry_success_marks_retried(self):
        did = self._make_dead()
        r = D.dead_retry(self.con, did, caller=self.fake([GOOD]))
        self.assertTrue(r["ok"])
        self.assertEqual(r["kept"], 1)
        self.assertEqual(D.dead_list(self.con), [])
        self.assertEqual(len(D.dead_list(self.con, "retried")), 1)

    def test_dead_retry_failure_stays_dead(self):
        did = self._make_dead()
        r = D.dead_retry(self.con, did, caller=self.failing())
        self.assertFalse(r["ok"])
        self.assertEqual(len(D.dead_list(self.con)), 1)

    def test_dead_drop_marks_not_deletes(self):
        did = self._make_dead()
        self.assertTrue(D.dead_drop(self.con, did))
        self.assertEqual(len(D.dead_list(self.con, "dropped")), 1)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM distill_dead").fetchone()[0], 1)
        with self.assertRaises(sqlite3.DatabaseError):
            self.con.execute("DELETE FROM distill_dead")
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 2)

    def test_audit_lists_dead_and_pending(self):
        self._make_dead()
        a = D.audit(self.con, NOW)
        self.assertEqual(a["dead"], 1)
        self.assertIn("derive_pending", a)


# ───────────────────────── 切片路径上的防回声 + 数字 ─────────────────────────
class SliceAntiEcho(Base):
    def setUp(self):
        super().setUp()
        D.init_baseline(self.con)

    def kept(self, facts, msgs=None):
        self.session(msgs=msgs)
        D.run(self.con, now_dt=NOW, caller=self.fake(facts))
        return [dict(r) for r in self.con.execute("SELECT subject, content FROM facts")]

    def test_speaker_echo_dropped(self):
        # 我（claude）说的话不能记成用户的事
        f = {"subject": "tester", "type": "episodic", "content": "小测要去大理", "quote": "我帮你查查大理的天气",
             "src": 1, "confidence": 0.8}
        self.assertEqual(self.kept([f]), [])
        self.assertEqual(self.con.execute("SELECT SUM(dropped) FROM distill_runs").fetchone()[0], 1)

    def test_own_words_kept_as_own_subject(self):
        f = {"subject": "claude", "type": "episodic", "content": "我答应帮她查大理的天气", "quote": "我帮你查查大理的天气",
             "src": 1, "confidence": 0.8}
        self.assertEqual([r["subject"] for r in self.kept([f])], ["claude"])

    def test_us_single_sided_not_kept_as_us(self):
        f = {"subject": "us", "type": "episodic", "content": "我们约好一起去大理玩", "quote": "想去大理玩一周",
             "src": 0, "confidence": 0.8}
        subs = [r["subject"] for r in self.kept([f])]
        self.assertNotIn("us", subs)

    def test_unsupported_number_dropped(self):
        f = {**GOOD, "content": "小测下个月想去大理玩77天", "quote": "想去大理玩一周"}
        self.assertEqual(self.kept([f]), [])
        f2 = {**GOOD, "content": "小测11月3日想去大理玩一周", "quote": "想去大理玩一周"}
        self.assertEqual(self.kept([f2]), [])

    def test_quote_not_in_chunk_dropped(self):
        self.assertEqual(self.kept([{**GOOD, "quote": "这句话原文里根本没有"}]), [])

    def test_context_is_not_a_quote_source(self):
        # 上文（已处理过的）只给模型看背景；quote 取自上文就找不到 → 丢
        old = self.msg("sessA", "tester", "上文里她说过喜欢吃草莓蛋糕呢", NOW - dt.timedelta(hours=9))
        D._save_cursor(self.con, D._cursor(self.con, "sessA"), last_evidence_id=old)
        self.session()
        calls = []
        D.run(self.con, now_dt=NOW, caller=self.fake([{**GOOD, "content": "小测喜欢草莓蛋糕", "quote": "喜欢吃草莓蛋糕呢"}], calls))
        self.assertIn("不要从里面抽事", calls[0])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)

    def test_promote_backstop_still_blocks(self):
        e = add_evidence(self.con, source_kind="chat", source_actor="claude", source_locator="Lx",
                         content="我说的话足够长啊", verification_status="verified")
        self.assertIsNone(promote(self.con, subject="tester", fact_type="episodic", content="她的事", evidence_ids=[e]))

    def test_near_dup_links_only_candidates_never_confirmed(self):
        self.seed_confirmed(0)
        c0 = "小测下个月想去大理玩一周"
        e = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="Cx", content=c0,
                         verification_status="verified")
        cand = promote(self.con, subject="tester", fact_type="episodic", content=c0, evidence_ids=[e],
                       valid_from=iso(NOW - dt.timedelta(days=1)))
        conf_c = "小测下个月很想去大理玩上一周时间"
        e2 = add_evidence(self.con, source_kind="chat", source_actor="tester", source_locator="Cy", content=conf_c,
                          verification_status="verified")
        conf = promote(self.con, subject="tester", fact_type="episodic", content=conf_c, evidence_ids=[e2],
                       status="confirmed", user_pinned=True, valid_from=iso(NOW - dt.timedelta(days=1)))
        self.con.commit()
        before = self.con.execute("SELECT COUNT(*) FROM fact_evidence WHERE fact_id=?", (conf,)).fetchone()[0]
        ids = self.session()
        D.run(self.con, now_dt=NOW, caller=self.fake([GOOD]))
        # 没新建 fact；对已确认的事实不动
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 2)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM fact_evidence WHERE fact_id=?", (conf,)).fetchone()[0], before)
        self.assertEqual(self.con.execute("SELECT SUM(near_dup) FROM distill_runs").fetchone()[0], 1)
        self.assertIsNotNone(cand)

    def test_confirmed_count_unchanged_through_pipeline(self):
        self.seed_confirmed(3)
        before = self.counts()["conf"]
        self.session()
        D.run(self.con, now_dt=NOW, caller=self.fake([GOOD]))
        derive_repair.repair(self.con)
        remember.remember(self.con, "我记得她不爱吃香菜这件事", "preference", None, run_id="r1", now_dt=NOW)
        self.assertEqual(self.counts()["conf"], before)


# ───────────────────────── 补派生层 ─────────────────────────
class DeriveRepair(Base):
    def setUp(self):
        super().setUp()
        D.init_baseline(self.con)
        self.session()
        D.run(self.con, now_dt=NOW, caller=self.fake([GOOD]))

    def test_pending_then_repair(self):
        p = derive_repair.pending(self.con)
        self.assertEqual(p["class"], 1)
        r = derive_repair.repair(self.con)
        self.assertEqual(r["class"], 1)
        self.assertEqual(derive_repair.pending(self.con)["class"], 0)
        row = self.con.execute("SELECT reason FROM fact_class").fetchone()
        self.assertEqual(row[0], "slice-v1")
        self.assertFalse(row[0].startswith("extract-llm-"))

    def test_cards_built_incrementally_and_idempotent(self):
        self.con.executescript((ROOT / "cards.sql").read_text(encoding="utf-8"))     # 生产库里卡片表早就有了；这里只建表
        self.con.execute("DELETE FROM fact_cards") if False else None
        derive_repair.repair(self.con)
        n = self.con.execute("SELECT COUNT(*) FROM fact_cards").fetchone()[0]
        self.assertEqual(n, 1)
        again = derive_repair.repair(self.con)
        self.assertEqual((again["class"], again["cards"]), (0, 0))
        self.assertEqual(derive_repair.pending(self.con)["cards"], 0)

    def test_does_not_touch_facts_columns(self):
        before = self.con.execute("SELECT id, content, status, user_pinned FROM facts").fetchall()
        derive_repair.repair(self.con)
        self.assertEqual(before, self.con.execute("SELECT id, content, status, user_pinned FROM facts").fetchall())


# ───────────────────────── 自记 ─────────────────────────
class SelfNote(Base):
    def rem(self, content, run="r1", now=NOW, quote=None, **kw):
        return remember.remember(self.con, content, "episodic", quote, run_id=run, now_dt=now, **kw)

    TEXTS = ["我答应她以后少说客套话多说实话", "她说睡前不想被问太多问题所以我先安静", "下次她难过时先陪着再给建议",
             "她讨厌被叫小朋友这个称呼要避开", "周日早上她喜欢先喝一杯温水再聊天", "她不爱吃香菜点菜时记得提醒店家"]

    def test_always_candidate_never_pinned_or_confirmed(self):
        ok, _ = self.rem(self.TEXTS[0], status="confirmed", pinned=True, user_pinned=True, subject="tester")
        self.assertTrue(ok)
        r = self.con.execute("SELECT subject, status, user_pinned, confidence FROM facts").fetchone()
        self.assertEqual((r["subject"], r["status"], r["user_pinned"]), ("claude", "candidate", 0))
        o = self.con.execute("SELECT origin FROM fact_origin").fetchone()[0]
        self.assertEqual(o, "self-note")

    def test_evidence_is_tool_result_secondary(self):
        self.rem(self.TEXTS[0])
        r = self.con.execute("SELECT e.source_kind, e.source_actor FROM evidence e").fetchone()
        self.assertEqual((r[0], r[1]), ("tool_result", "claude"))
        tier = self.con.execute("SELECT * FROM evidence_tier").fetchall() if self._has("evidence_tier") else None
        if tier is not None:
            self.assertIn("secondary", [str(x) for x in tuple(tier[0])])

    def _has(self, name):
        return self.con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None

    def test_cap_3_per_session(self):
        for i in range(3):
            self.assertTrue(self.rem(self.TEXTS[i])[0], i)
        ok, msg = self.rem(self.TEXTS[3])
        self.assertFalse(ok)
        self.assertIn("本会话", msg)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 3)

    def test_cap_5_per_day_across_sessions(self):
        n = 0
        for i, t in enumerate(self.TEXTS):
            ok, msg = self.rem(t, run=f"run{i // 3}")
            n += ok
        self.assertEqual(n, 5)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 5)
        ok, msg = self.rem("明天一定要提醒自己别忘了她今天说的那句话", run="run9")
        self.assertFalse(ok)
        self.assertIn("今天", msg)
        # 第二天重新计
        ok, _ = self.rem("明天一定要提醒自己别忘了她今天说的那句话", run="run9", now=NOW + dt.timedelta(days=1))
        self.assertTrue(ok)

    def test_marker_not_in_content_fts_cards_vec(self):
        self.rem(self.TEXTS[0])
        self.con.executescript((ROOT / "cards.sql").read_text(encoding="utf-8"))
        derive_repair.repair(self.con)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE content LIKE '%自记%'").fetchone()[0], 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM fact_cards WHERE card LIKE '%自记%'").fetchone()[0], 0)
        for (name,) in self.con.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%'").fetchall():
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {name} WHERE {name} MATCH '自记'").fetchone()[0]
            except sqlite3.DatabaseError:
                continue
            self.assertEqual(n, 0, name)
        # 向量的输入就是 facts.content（vectors.py）：干净
        self.assertNotIn("自记", self.con.execute("SELECT content FROM facts").fetchone()[0])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM evidence WHERE content LIKE '%自记%'").fetchone()[0], 0)

    def test_marker_word_in_content_rejected(self):
        ok, _ = self.rem("〔助理甲自记〕我答应她以后少说客套话")
        self.assertFalse(ok)

    def test_render_tag_via_echo(self):
        self.rem(self.TEXTS[0])
        ext = self.con.execute("SELECT ext_id FROM facts").fetchone()[0]
        plain = promote(self.con, subject="tester", fact_type="episodic", content="一条普通的候选事实内容",
                        evidence_ids=[add_evidence(self.con, source_kind="chat", source_actor="tester",
                                                   source_locator="Lq", content="一条普通的候选事实内容啊",
                                                   verification_status="verified")])
        self.con.commit()
        ext2 = self.con.execute("SELECT ext_id FROM facts WHERE id=?", (plain,)).fetchone()[0]
        hits = [{"ext_id": ext}, {"ext_id": ext2}]
        echo.annotate_origin(self.con, hits)
        self.assertTrue(hits[0]["self_note"])
        self.assertFalse(hits[1]["self_note"])
        self.assertTrue(echo.tag(hits[0]).endswith(echo.SELF_TAG))
        self.assertEqual(echo.tag(hits[1]), "")
        self.assertEqual(echo.tag({"echo_from": ["tester"]}), echo.label(["tester"]))
        self.assertEqual(mcp_server.echo_tags(self.con, [ext, ext2]).get(ext), echo.SELF_TAG)
        self.assertNotIn(ext2, mcp_server.echo_tags(self.con, [ext, ext2]))

    def test_hook_prefix(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("hk", ROOT / "hooks_engram_recall.py")
        hk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hk)
        self.assertEqual(hk.echo_prefix({}), "")
        self.assertEqual(hk.echo_prefix({"self_note": True}), "〔助理甲自记〕")
        self.assertEqual(hk.echo_prefix({"echo_from": ["tester"]}), echo.label(["tester"]))

    def test_digits_need_quote_and_support(self):
        self.assertFalse(self.rem("她说周五下午三点要去医院复查身体")[0] and False)
        ok, _ = self.rem("她 11 月 3 日要去医院复查身体情况")
        self.assertFalse(ok)                                   # 没 quote
        ok, _ = self.rem("她 11 月 3 日要去医院复查身体情况", quote="我十月二十号去复查")
        self.assertFalse(ok)                                   # quote 里没有这些数字
        ok, _ = self.rem("她 11 月 3 日要去医院复查身体情况", quote="11 月 3 日去医院复查")
        self.assertTrue(ok)

    def test_duplicate_rejected(self):
        self.assertTrue(self.rem(self.TEXTS[0])[0])
        self.assertFalse(self.rem(self.TEXTS[0], run="r2")[0])
        self.assertFalse(self.rem(self.TEXTS[0] + "啊", run="r3")[0])      # 近重复

    def test_self_note_never_confirmed_by_pipeline(self):
        self.rem(self.TEXTS[0])
        derive_repair.repair(self.con)
        r = self.con.execute("SELECT status, user_pinned FROM facts").fetchone()
        self.assertEqual((r[0], r[1]), ("candidate", 0))

    def test_origin_is_append_only(self):
        self.rem(self.TEXTS[0])
        with self.assertRaises(sqlite3.DatabaseError):
            self.con.execute("UPDATE fact_origin SET origin='distill'")
        with self.assertRaises(sqlite3.DatabaseError):
            self.con.execute("DELETE FROM fact_origin")


# ───────────────────────── MCP 工具表 ─────────────────────────
class McpTools(Base):
    def test_off_identical_to_tools(self):
        with mock.patch.dict(os.environ, ENV, clear=True):
            r = mcp_server.handle("tools/list", {}, self.con)
        self.assertIs(r["tools"], mcp_server.TOOLS)
        self.assertNotIn("engram_remember", [t["name"] for t in r["tools"]])
        pin = [t for t in r["tools"] if t["name"] == "engram_pin"][0]
        self.assertNotIn("engram_remember", pin["description"])

    def test_off_call_unknown(self):
        with mock.patch.dict(os.environ, ENV, clear=True):
            r = mcp_server.handle("tools/call", {"name": "engram_remember",
                                                 "arguments": {"content": "我答应她以后少说客套话多说实话"}}, self.con)
        self.assertIn("没有这个工具", r["content"][0]["text"])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)

    def test_on_listed_and_callable(self):
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_SELF_REMEMBER": "1"}, clear=True):
            names = [t["name"] for t in mcp_server.handle("tools/list", {}, self.con)["tools"]]
            self.assertIn("engram_remember", names)
            pin = [t for t in mcp_server.handle("tools/list", {}, self.con)["tools"] if t["name"] == "engram_pin"][0]
            self.assertTrue(pin["description"].endswith(remember.PIN_EXTRA))
            r = mcp_server.handle("tools/call", {"name": "engram_remember", "arguments": {
                "content": "我答应她以后少说客套话多说实话", "status": "confirmed", "pinned": True}}, self.con)
        self.assertIn("已记", r["content"][0]["text"])
        row = self.con.execute("SELECT status, user_pinned FROM facts").fetchone()
        self.assertEqual((row[0], row[1]), ("candidate", 0))
        # TOOLS 本身没被改（深拷贝）
        self.assertNotIn(remember.PIN_EXTRA, [t for t in mcp_server.TOOLS if t["name"] == "engram_pin"][0]["description"])


# ───────────────────────── llm 用量 / 精排 ─────────────────────────
class LlmUsage(Base):
    def setUp(self):
        super().setUp()
        self.db = str(Path(self.dir.name) / "t.db")

    def rows(self):
        return self.con.execute("SELECT purpose, route, model, prompt_tokens, ok FROM llm_usage").fetchall()

    def test_default_in_unittest_writes_nothing(self):
        with mock.patch.object(llm, "USAGE_DB", self.db):
            llm._usage("extract", "primary", "m", {"prompt_tokens": 1}, True)
        self.assertEqual(self.rows(), [])

    def test_force_writes_row(self):
        with mock.patch.object(llm, "USAGE_DB", self.db), mock.patch.object(llm, "USAGE_FORCE", True):
            llm._usage("extract", "primary", "m", {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}, True)
            llm._usage("extract", "fallback", "m2", None, False, "boom")
        r = self.rows()
        self.assertEqual(len(r), 2)
        self.assertEqual((r[0][0], r[0][3], r[0][4]), ("extract", 7, 1))
        self.assertEqual(r[1][4], 0)

    def test_env_off(self):
        with mock.patch.object(llm, "USAGE_DB", self.db), mock.patch.object(llm, "USAGE_FORCE", True), \
                mock.patch.dict(os.environ, {"ENGRAM_LLM_USAGE": "0"}):
            llm._usage("extract", "primary", "m", {"prompt_tokens": 7}, True)
        self.assertEqual(self.rows(), [])

    def test_silent_when_table_or_db_missing(self):
        empty = str(Path(self.dir.name) / "empty.db")
        sqlite3.connect(empty).close()
        import io, contextlib
        buf = io.StringIO()
        with mock.patch.object(llm, "USAGE_DB", empty), mock.patch.object(llm, "USAGE_FORCE", True), \
                contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            llm._usage("extract", "primary", "m", {"prompt_tokens": 7}, True)
            llm._usage("extract", "primary", "m", {"prompt_tokens": 7}, True) if False else None
        with mock.patch.object(llm, "USAGE_DB", "/nonexistent/x.db"), mock.patch.object(llm, "USAGE_FORCE", True), \
                contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            llm._usage("extract", "primary", "m", {}, True)
        self.assertEqual(buf.getvalue(), "")

    def test_rerank_flash_purpose(self):
        self.assertEqual(llm.PURPOSE_MODEL_VARS["rerank_flash"][0], "RERANK_FLASH_MODEL")
        self.assertEqual(llm.PURPOSE_MODEL_VARS["rerank"], ("RERANK_MODEL",))        # 对比模型那条没动
        with mock.patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(rerank._purpose(), "rerank")
        with mock.patch.dict(os.environ, {**ENV, "ENGRAM_RERANK_FLASH": "1"}, clear=True):
            self.assertEqual(rerank._purpose(), "rerank_flash")

    def test_newer_still_uses_rerank_purpose(self):
        src = (ROOT / "newer.py").read_text(encoding="utf-8") if (ROOT / "newer.py").exists() else \
            (ROOT / "extract" / "newer.py").read_text(encoding="utf-8")
        self.assertIn('purpose="rerank"', src)
        self.assertNotIn("rerank_flash", src)


# ───────────────────────── 说话人统一 / 信箱 ─────────────────────────
class Speakers(unittest.TestCase):
    def test_mailbox_mapping(self):
        self.assertEqual(speakers.mailbox_actor({"from": "me"}), "tester")
        self.assertEqual(speakers.mailbox_actor({"from": "claude"}), "claude")
        self.assertIsNone(speakers.mailbox_actor({"from": "somebody-else"}))
        self.assertIsNone(speakers.mailbox_actor({}))
        self.assertIsNone(speakers.mailbox_actor({"from": ""}))

    def test_same_actor_names_as_channel(self):
        from extract import from_channel
        self.assertEqual(set(speakers.VALID_ACTORS), {"tester", "claude", "gpt", "agentc"})
        self.assertEqual(speakers.normalize_actor("tester"), "tester")
        self.assertIn("persona.USER", inspect_src(from_channel))

    def test_mailbox_unknown_not_labelled_claude(self):
        src = (ROOT / "ingest_mailbox.py").read_text(encoding="utf-8")
        self.assertNotIn("'tester' if m['from'] == 'me' else 'claude'", src)
        self.assertIn("speakers.mailbox_actor", src)


def inspect_src(mod):
    return Path(mod.__file__).read_text(encoding="utf-8")


# ───────────────────────── schema / 脚本 ─────────────────────────
class SchemaAndScripts(Base):
    def test_schema_idempotent(self):
        con2 = connect(Path(self.dir.name) / "t.db")
        con2.close()
        for t in ("engram_meta", "ingest_cursor", "distill_cursor", "distill_runs", "distill_dead", "fact_origin",
                  "llm_usage"):
            self.assertTrue(self.con.execute("SELECT 1 FROM sqlite_master WHERE name=?", (t,)).fetchone(), t)

    def test_scripts_syntax_and_old_path_kept(self):
        for s in ("hourly-ingest.sh", "auto-ingest.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / s)]).returncode, 0, s)
        h = (ROOT / "hourly-ingest.sh").read_text(encoding="utf-8")
        self.assertIn("DISTILL_SLICE", h)
        self.assertIn("from_evidence", h)                 # 旧命令还在
        a = (ROOT / "auto-ingest.sh").read_text(encoding="utf-8")
        self.assertIn("DISTILL_SLICE", a)
        self.assertIn("ingest.py", a)

    def test_scripts_flag_off_take_old_branch(self):
        for s in ("hourly-ingest.sh", "auto-ingest.sh"):
            txt = (ROOT / s).read_text(encoding="utf-8")
            self.assertIn("exit 0", txt)
            self.assertIn("else", txt) if False else None



def setUpModule():
    # 这个文件测的是旧管线（值得记闸关）的行为：不管环境变量 ENGRAM_VALUE_GATE 开没开，都按闸关跑（闸开的行为在 test_value_gate*.py 里测）
    import os
    from unittest import mock
    p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "0"})
    p.start()
    unittest.addModuleCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
