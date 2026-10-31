"""说话人标注：collect/harvest 严格按名单认 channel 用户 + 历史错标的 overlay（evidence_actor_fix，不改 evidence）。"""
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
             'user_id="1000000003" ts="2026-08-02T14:09:30.000Z">\n第一版草稿写好了，直接复制：第 4 题：我想和一个 AI 一起体验这个小游戏\n</channel>')
HER_BLOCK = ('<channel source="plugin:telegram:telegram" chat_id="1000000001" message_id="3786" user="1000000001" '
             'user_id="1000000001" ts="2026-08-16T15:56:32.000Z">\n今天把周报整理完了，明天开始写测试\n</channel>')
HER_BLOCK2 = HER_BLOCK.replace("今天把周报整理完了，明天开始写测试", "这个周末想去海边走走，看看日出")


def _jsonl(path: Path, recs: list[dict]):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")


def _user(text, ts="2026-10-28T01:00:00Z"):
    return {"type": "user", "message": {"role": "user", "content": text}, "timestamp": ts}


class SpeakerLabels(unittest.TestCase):
    def test_channel_actor_strict(self):
        self.assertEqual(speakers.channel_actor("AgentB_bot", "1000000003"), "gpt")
        self.assertEqual(speakers.channel_actor("AgentB_bot", ""), "gpt")
        self.assertEqual(speakers.channel_actor("AgentC_bot", "1000000002"), "agentc")
        self.assertEqual(speakers.channel_actor("1000000001", "1000000001"), "tester")     # user 属性直接是她的数字 id
        self.assertEqual(speakers.channel_actor("whatever", "1000000001"), "tester")       # id 优先
        self.assertIsNone(speakers.channel_actor("SomeOtherGPT_bot", "999"))                # 名单外的 bot 不认
        self.assertIsNone(speakers.channel_actor("AgentB_bot", "1000000001"))         # id 和名字对不上：宁可不认
        self.assertIsNone(speakers.channel_actor("", ""))

    def test_whose_uses_the_same_table(self):
        self.assertEqual(FC.whose('source="plugin:telegram:telegram" user="AgentB_bot" user_id="1000000003"'), "gpt")
        self.assertEqual(FC.whose('source="x" user="Stranger_bot" user_id="1"'), "unknown:Stranger_bot")
        self.assertEqual(FC.BY_NAME, speakers.CHANNEL_BY_NAME)                              # 老别名还在

    def test_reconcile_leading_wrapper(self):
        self.assertEqual(speakers.reconcile("tester", GPT_BLOCK), ("gpt", "relabeled"))
        self.assertEqual(speakers.reconcile("tester", HER_BLOCK), ("tester", None))
        self.assertEqual(speakers.reconcile("tester", "<channel source=\"x\" user=\"Foo_bot\" user_id=\"5\">hi</channel>"),
                         (None, "unknown-channel-user"))
        self.assertEqual(speakers.reconcile("tester", "普通一句话"), ("tester", None))
        self.assertEqual(speakers.reconcile("claude", "已回。\n\nuser" + HER_BLOCK), ("claude", None))   # 夹在中间的不动（归 overlay）

    def test_harvest_any_source_block_is_not_her(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.jsonl"
            _jsonl(f, [_user(GPT_BLOCK), _user(HER_BLOCK),
                       _user('<channel source="plugin:discord:x" user="Foo_bot" user_id="5">别的来源的 bot 说话说话</channel>'),
                       _user("她自己在 cc-connect 里打的一句普通话，足够长")])
            got = [(a, b[:6]) for _, a, b, _ in FC.harvest(f)]
        self.assertEqual([a for a, _ in got], ["gpt", "tester", "unknown:Foo_bot", "tester"])

    def test_collect_labels_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.jsonl"
            _jsonl(f, [_user(GPT_BLOCK), _user(HER_BLOCK, "2026-10-28T01:05:00Z"),
                       _user('<channel source="plugin:telegram:telegram" user="Foo_bot" user_id="5">陌生 bot 的一句话呀</channel>', "2026-10-28T01:06:00Z")])
            con = connect(Path(d) / "t.db")
            t = collect.collect_file(con, f, full=True)
            rows = {r["source_actor"] for r in con.execute("select source_actor from evidence")}
        self.assertEqual(rows, {"gpt", "tester"})
        self.assertEqual((t["new"], t["unknown"]), (2, 1))

    def test_collect_reconcile_relabels_when_harvest_says_her(self):
        """harvest 以外的路径（或老版本）把转发包装标成她：collect 最后一道按包装改标，认不出的跳过。"""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.jsonl"
            _jsonl(f, [_user("x")])
            con = connect(Path(d) / "t.db")
            fake = [(1, "tester", GPT_BLOCK, "2026-10-28T01:00:00Z"),
                    (2, "tester", '<channel source="s" user="Foo_bot" user_id="5">陌生 bot 的一句话呀</channel>', "2026-10-28T01:01:00Z")]
            with mock.patch.object(collect, "harvest", return_value=iter(fake)):
                t = collect.collect_file(con, f, full=True)
            acts = [r["source_actor"] for r in con.execute("select source_actor from evidence")]
        self.assertEqual(acts, ["gpt"])
        self.assertEqual((t["relabeled"], t["unknown"], t["new"]), (1, 1, 1))


class ActorFixOverlay(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.con = connect(Path(self._d.name) / "t.db")
        P = "/x/s1.jsonl"
        self.e_lead = add_evidence(self.con, source_kind="chat", source_actor="tester", source_path=P, source_locator="L1",
                                   content=GPT_BLOCK, occurred_at="2026-08-02T14:09:30+00:00")
        self.e_her_plain = add_evidence(self.con, source_kind="chat", source_actor="tester", source_path=P, source_locator="L2",
                                        content="小测（时间）：下课了，回去了🌚", occurred_at="2026-08-16T11:33:00+00:00")
        self.e_emb_ok = add_evidence(self.con, source_kind="chat", source_actor="claude", source_path=P, source_locator="L3",
                                     content="已回。\n\nuser" + HER_BLOCK.replace("今天把周报整理完了，明天开始写测试", "下课了，回去了🌚"),
                                     occurred_at="2026-08-16T11:34:00+00:00")
        self.e_emb_fake = add_evidence(self.con, source_kind="chat", source_actor="claude", source_path=P, source_locator="L4",
                                       content="已回。\n\nuser" + HER_BLOCK, occurred_at="2026-08-16T15:55:00+00:00")
        self.e_fine = add_evidence(self.con, source_kind="chat", source_actor="tester", source_path=P, source_locator="L5",
                                   content=HER_BLOCK2, occurred_at="2026-08-16T15:56:00+00:00")
        self.con.commit()

    def tearDown(self):
        self.con.close()
        self._d.cleanup()

    def test_compute_classifies(self):
        fx = actor_fix.compute_all(self.con)
        self.assertEqual(set(fx), {self.e_lead, self.e_emb_ok, self.e_emb_fake})            # 对的（她的转发、普通话）不动
        self.assertEqual((fx[self.e_lead]["kind"], fx[self.e_lead]["effective_actor"]), ("leading-block", "gpt"))
        self.assertEqual((fx[self.e_emb_ok]["block_actor"], fx[self.e_emb_ok]["corroborated"]), ("tester", 1))
        self.assertEqual((fx[self.e_emb_fake]["block_actor"], fx[self.e_emb_fake]["corroborated"]), ("tester", 0))

    def test_build_is_append_only_overlay_and_idempotent(self):
        before = self.con.execute("select id, source_actor, content, content_sha256 from evidence order by id").fetchall()
        r1 = actor_fix.build(self.con)
        r2 = actor_fix.build(self.con)
        after = self.con.execute("select id, source_actor, content, content_sha256 from evidence order by id").fetchall()
        self.assertEqual([tuple(x) for x in before], [tuple(x) for x in after])              # evidence 一字不动
        self.assertEqual((r1["inserted"], r2["inserted"]), (3, 0))
        n = self.con.execute("select count(*) from evidence_actor_fix").fetchone()[0]
        self.assertEqual(n, 3)
        with self.assertRaises(sqlite3.DatabaseError):                                       # evidence 本身仍不可改
            self.con.execute("update evidence set source_actor='gpt' where id=?", (self.e_lead,))
        v = {r["evidence_id"]: r["effective_actor"] for r in self.con.execute("select * from evidence_effective")}
        self.assertEqual((v[self.e_lead], v[self.e_her_plain]), ("gpt", "tester"))
        self.assertTrue(self.con.execute("select reason from evidence_actor_fix where evidence_id=?", (self.e_lead,)).fetchone()[0])

    def test_echo_actors_use_overlay(self):
        self.assertEqual(echo.evidence_actors(self.con, [self.e_lead]), {"tester"})         # 没 build：原标注
        actor_fix.build(self.con)
        self.assertEqual(echo.evidence_actors(self.con, [self.e_lead]), {"gpt"})
        self.assertEqual(echo.evidence_actors(self.con, [self.e_emb_ok]), {"claude", "tester"})
        self.assertEqual(echo.evidence_actors(self.con, [self.e_emb_fake]), {"claude"})      # 查不到佐证的块不算她的话
        ok, why = echo.evidence_allows(self.con, "tester", [self.e_lead])
        self.assertFalse(ok, why)                                                            # bot 转发不能再记成她的事
        self.assertTrue(echo.evidence_allows(self.con, "gpt", [self.e_lead])[0])

    def test_provenance_uses_overlay(self):
        from store.store import promote
        fid = promote(self.con, subject="tester", fact_type="episodic", content="小测决定在入镇申请里写不装腔版",
                      evidence_ids=[self.e_lead], valid_from="2026-08-02T14:09:30+00:00", confidence=0.5)
        ext = self.con.execute("select ext_id from facts where id=?", (fid,)).fetchone()[0] if fid else None
        if ext is None:                       # promote 的防回声在 overlay 之前可能放行（旧标注是 user），这里只测召回侧
            self.skipTest("promote refused")
        self.assertFalse(echo.is_echo_only(echo.provenance(self.con, [ext])[ext]))           # 旧标注：看起来有本人证据
        actor_fix.build(self.con)
        p = echo.provenance(self.con, [ext])[ext]
        self.assertTrue(echo.is_echo_only(p))                                                # overlay 后：只有 gpt 的话 → 转述
        self.assertEqual(p["actors"], {"gpt"})

    def test_distill_render_follows_overlay_without_build(self):
        rows = self.con.execute("select id, occurred_at, captured_at, source_actor, content, source_path, source_locator"
                                " from evidence where id in (?,?,?) order by id", (self.e_lead, self.e_emb_ok, self.e_emb_fake)).fetchall()
        out = DS._render_rows(self.con, rows)
        by = [(o["id"], o["source_actor"], o["content"].split("：", 1)[0][:3]) for o in out]
        self.assertEqual(by, [(self.e_lead, "gpt", "助手乙"), (self.e_emb_ok, "claude", "我（2"), (self.e_emb_ok, "tester", "小测（"),
                              (self.e_emb_fake, "claude", "我（2")])                          # 假块整段不进片段
        self.assertNotIn("channel", "".join(o["content"] for o in out))

    def test_check_blocks_user_fact_from_bot_forward(self):
        rows = self.con.execute("select id, occurred_at, captured_at, source_actor, content, source_path, source_locator"
                                " from evidence where id=?", (self.e_lead,)).fetchall()
        chunk = DS._render_rows(self.con, rows)
        f = {"keep": True, "category": "decision", "subject": "tester", "content": "小测决定在入镇申请里写不装腔版，想和在意的 AI 一起体验这个小镇",
             "quote": "我想和一个 AI 一起体验这个小游戏", "src": 0}
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok, why = FE.check(f, chunk)
        self.assertIsNone(ok)
        self.assertIn("subject-echo", why)                                                   # gpt 的话不能记成她的
        # 对照：没有 overlay 时（旧标注 user）会被放行——这就是 S25 的漏网
        stale = [{**chunk[0], "source_actor": "tester"}]
        with mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "1"}):
            ok2, _ = FE.check(f, stale)
        self.assertIsNotNone(ok2)


if __name__ == "__main__":
    unittest.main()
