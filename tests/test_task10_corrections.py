"""任务四·步骤1：(A) 她当面纠正 → engram_remember(kind=correction) 当场入「她纠正过我」层；(B) 召回时把我自己说过的话降权。"""
import datetime as dt
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import echo  # noqa: E402
import mcp_server  # noqa: E402
import persona  # noqa: E402
import remember  # noqa: E402
from extract import collect  # noqa: E402
from store.store import add_evidence, connect, promote, record_tier  # noqa: E402

SPEC = importlib.util.spec_from_file_location("engram_hook_t10", ROOT / "hooks_engram_recall.py")
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)

NOW = dt.datetime(2026, 10, 28, 12, 0, tzinfo=dt.timezone.utc)
HER = "我不在家，也不在宿舍啊，我在图书馆"
KW = dict(content="她今天不在宿舍，在图书馆", wrong="她在宿舍", topic="所在位置", quote="我不在家，也不在宿舍啊")


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tmp = Path(self.dir.name)
        self.con = connect(self.tmp / "t.db")
        self.addCleanup(self.con.close)
        self.cj = self.tmp / "corrections.jsonl"
        self.tdir = self.tmp / "sessions"
        self.tdir.mkdir()

    def her_evidence(self, body=HER, actor=None, when="2026-10-28T10:30:00.000Z", loc="L1"):
        ev = add_evidence(self.con, source_kind="chat", source_actor=actor or persona.USER,
                          source_path="s.jsonl", source_locator=loc, content=body,
                          occurred_at=when, verification_status="verified")
        record_tier(self.con, ev, "primary", "s.jsonl:" + loc)
        self.con.commit()
        return ev

    def corr(self, **over):
        kw = {**KW, **over}
        return remember.remember(self.con, kw["content"], "correction", kw["quote"], wrong=kw["wrong"],
                                 topic=kw["topic"], now_dt=NOW, transcript_dir=self.tdir,
                                 corrections_path=self.cj)

    def facts(self):
        return self.con.execute("SELECT * FROM facts").fetchall()


class CorrectionWrite(Base):
    def test_her_words_in_db_land_confirmed_pinned_in_corrections_layer(self):
        ev = self.her_evidence()
        ok, msg = self.corr()
        self.assertTrue(ok, msg)
        f = self.facts()[0]
        self.assertEqual((f["status"], f["user_pinned"], f["subject"]), ("confirmed", 1, persona.USER))
        self.assertTrue(f["content"].startswith("【她纠正过我·2026-10-28·所在位置】我以为她在宿舍；其实她今天不在宿舍"), f["content"])
        link = self.con.execute("SELECT evidence_id FROM fact_evidence WHERE fact_id=?", (f["id"],)).fetchall()
        self.assertEqual([r[0] for r in link], [ev])
        # 不是自记：没有 fact_origin 行，也不占自记名额
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM fact_origin").fetchone()[0], 0)
        row = json.loads(self.cj.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((row["topic"], row["keep"], row["via"]), ("所在位置", True, "engram_remember"))
        self.assertIn("不在宿舍", row["right"])

    def test_found_in_transcript_stores_her_evidence_and_collect_rescan_is_noop(self):
        f = self.tdir / "s.jsonl"
        f.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in [
            {"type": "assistant", "message": {"role": "assistant", "content": "好，回宿舍再说"}, "timestamp": "2026-10-28T11:00:00Z"},
            {"type": "user", "message": {"role": "user", "content": HER}, "timestamp": "2026-10-28T11:01:00Z"},
        ]) + "\n", encoding="utf-8")
        ok, msg = self.corr()
        self.assertTrue(ok, msg)
        ev = self.con.execute("SELECT source_actor, source_kind, content FROM evidence").fetchall()
        self.assertEqual([(r[0], r[1]) for r in ev], [(persona.USER, "chat")])
        self.assertEqual(self.con.execute("SELECT tier FROM evidence_tier").fetchone()[0], "primary")
        t = collect.collect_file(self.con, f, full=True)
        self.assertGreaterEqual(t["have"], 1)                    # 她那句已在库里，重扫认得「已有」
        self.assertEqual(self.con.execute(
            "SELECT COUNT(*) FROM evidence WHERE source_actor=? AND content LIKE ?", (persona.USER, "%图书馆%")).fetchone()[0], 1)

    def test_only_assistant_said_it_is_refused(self):
        f = self.tdir / "s.jsonl"
        f.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant",
                                 "content": "你今天不在家，也不在宿舍啊，我记住了"}, "timestamp": "2026-10-28T11:00:00Z"},
                                ensure_ascii=False) + "\n", encoding="utf-8")
        self.her_evidence(body="今天天气好", actor="claude")
        ok, msg = self.corr(quote="不在家，也不在宿舍啊")
        self.assertFalse(ok)
        self.assertIn("找不到", msg)
        self.assertEqual(self.facts(), [])

    def test_assistant_evidence_row_with_same_text_is_refused(self):
        self.her_evidence(actor="claude")
        ok, _ = self.corr()
        self.assertFalse(ok)
        self.assertEqual(self.facts(), [])

    def test_stale_message_is_refused(self):
        self.her_evidence(when="2026-10-20T10:30:00.000Z")
        ok, _ = self.corr()
        self.assertFalse(ok)

    def test_content_not_grounded_in_quote_is_refused(self):
        self.her_evidence()
        ok, msg = self.corr(content="她最近心情很差，需要人陪")
        self.assertFalse(ok)
        self.assertIn("对不上", msg)

    def test_invented_number_is_refused(self):
        self.her_evidence()
        ok, msg = self.corr(content="她今天不在宿舍，在图书馆三楼305")
        self.assertFalse(ok)
        self.assertIn("数字", msg)

    def test_missing_fields_refused(self):
        self.her_evidence()
        self.assertFalse(self.corr(quote="")[0])
        self.assertFalse(self.corr(wrong="")[0])
        self.assertFalse(self.corr(topic="")[0])
        self.assertEqual(self.facts(), [])

    def test_same_correction_twice_is_one_fact(self):
        self.her_evidence()
        self.assertTrue(self.corr()[0])
        ok, msg = self.corr()
        self.assertFalse(ok)
        self.assertIn("已有相近", msg)
        self.assertEqual(len(self.facts()), 1)

    def test_daily_cap(self):
        cases = [("咖啡早就戒掉了", "她早就戒掉咖啡", "我以为她天天喝咖啡", "咖啡"),
                 ("耳机是蓝牙的不是有线的", "她的耳机是蓝牙的", "我以为耳机有线", "耳机"),
                 ("高铁票已经改签了啊", "她的高铁票已经改签", "我以为没改签", "高铁"),
                 ("考试周只有三门不是五门", "她考试周三门课", "我以为要考很多门", "考试"),
                 ("同事周末才回来", "同事周末才回来", "我以为同事在", "同事"),
                 ("快递放在菜鸟驿站了", "快递放在菜鸟驿站", "我以为放门口", "快递"),
                 ("奶茶只喝少糖的", "她奶茶只喝少糖", "我以为全糖", "奶茶"),
                 ("课表周四全天没课", "她周四全天没课", "我以为周四有课", "课表"),
                 ("房租是押一付三的", "房租押一付三", "我以为月付", "房租"),
                 ("体检结果完全正常", "她体检结果正常", "我以为有问题", "体检"),
                 ("游戏早就卸载了", "她早就卸载游戏", "我以为还在玩", "游戏")]
        for i, (body, content, wrong, topic) in enumerate(cases):
            self.her_evidence(body=body, loc=f"L{i}")
            r = remember.remember(self.con, content + "，别再搞错", "correction", body, wrong=wrong, topic=topic, now_dt=NOW,
                                  transcript_dir=self.tdir, corrections_path=self.cj)
            if i < remember.CORR_PER_DAY:
                self.assertTrue(r[0], (i, r))
        self.assertFalse(r[0])
        self.assertIn("不再收", r[1])

    def test_self_note_quotas_untouched_by_corrections(self):
        self.her_evidence()
        self.assertTrue(self.corr()[0])
        self.assertEqual(remember.counts(self.con, remember.RUN_ID, NOW), (0, 0))

    def test_correction_is_found_by_hook_pinned_layer(self):
        self.her_evidence()
        self.assertTrue(self.corr()[0])
        got = hook.pinned_candidates(self.con, "我今天不在宿舍你怎么又说回宿舍再说")
        self.assertEqual(len(got), 1)
        self.assertIn("她纠正过我", got[0]["content"])


class McpTool(Base):
    ENV = {"ENGRAM_ENV_FILE": "/nonexistent"}

    def test_schema_has_correction_only_when_on(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            names = [t["name"] for t in mcp_server.handle("tools/list", {}, self.con)["tools"]]
        self.assertNotIn("engram_remember", names)
        self.assertTrue(remember.TOOL["inputSchema"]["properties"]["kind"]["enum"][-1] == "correction")
        self.assertIn("wrong", remember.TOOL["inputSchema"]["properties"])

    def test_on_call_passes_wrong_topic_and_refuses_ungrounded(self):
        with mock.patch.dict(os.environ, {**self.ENV, "ENGRAM_SELF_REMEMBER": "1"}, clear=True):
            r = mcp_server.handle("tools/call", {"name": "engram_remember", "arguments": {
                "content": KW["content"], "kind": "correction", "quote": KW["quote"],
                "wrong": KW["wrong"], "topic": KW["topic"]}}, self.con)
        self.assertEqual(self.facts(), [])                       # 库里没有她这句话 → 拒绝
        self.assertIn("找不到", r["content"][0]["text"])


class AssistantSay(unittest.TestCase):
    def h(self, subject, content, **kw):
        return {"subject": subject, "content": content, "ext_id": "f" + str(abs(hash(content)) % 9999), **kw}

    def test_detects_utterance_form_of_claude_only(self):
        c = persona.CLAUDE
        self.assertTrue(echo.is_assistant_say(self.h(c, "助手（2026-09-01 14:02）：说开会别折腾电脑，给了一个脚本文件，让回家再用")))
        self.assertTrue(echo.is_assistant_say(self.h(c, "我：这个想法不错 反正自己掌握的技能谁也拿不走")))
        self.assertFalse(echo.is_assistant_say(self.h(c, "助手定下硬规矩：只有她说话才算唤醒源")))
        self.assertFalse(echo.is_assistant_say(self.h(persona.USER, "用户（2026-09-15 12:33）：上个月同事吵起来")))
        self.assertFalse(echo.is_assistant_say(self.h(persona.USER, "我：不在宿舍")))

    def test_annotate_flags_prefix_and_demote_is_stable_and_lossless(self):
        c = persona.CLAUDE
        hits = [self.h(c, "助手（2026-09-01 14:02）：让回家再用", ext_id="a"),
                self.h(persona.USER, "用户平时待在公司，住处只是晚上回去睡觉", ext_id="b"),
                self.h(c, "助手决定不做套一层 OpenCode 的方案", ext_id="c"),
                self.h(persona.USER, "用户表示不住宿舍", ext_id="d")]
        echo.annotate_say(hits)
        self.assertEqual([h["assistant_say"] for h in hits], [True, False, False, False])
        self.assertIn("〔我说的〕", hook.echo_prefix(hits[0]))
        self.assertNotIn("〔我说的〕", hook.echo_prefix(hits[1]))
        echo.demote_say(hits)
        self.assertEqual([h["ext_id"] for h in hits], ["b", "c", "d", "a"])   # 稳定；一条都没丢

    def test_pinned_correction_is_never_assistant_say(self):
        fix = self.h(persona.USER, "【她纠正过我·2026-10-28·所在位置】我以为她在宿舍；其实她不在宿舍", user_pinned=1)
        echo.annotate_say([fix])
        self.assertFalse(fix["assistant_say"])


if __name__ == "__main__":
    unittest.main()
