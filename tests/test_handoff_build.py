"""跨窗接力交接笔记生成（handoff_build.py）。全部用虚构的会话文件。"""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import tests  # noqa: F401  虚构人设
import handoff_build as H


def _iso(ts):
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_session(d: Path, sid: str, turns, mtime: float, summary: str | None = None):
    lines = []
    if summary:
        lines.append({"type": "user", "timestamp": _iso(turns[0][1] - 1 if turns else mtime),
                      "message": {"content": H.COMPACT_PREFIX + "\n\nSummary:\n" + summary}})
    for who, ts, text in turns:
        if who == "u":
            lines.append({"type": "user", "timestamp": _iso(ts), "message": {"content": text}})
        else:
            lines.append({"type": "assistant", "timestamp": _iso(ts),
                          "message": {"content": [{"type": "text", "text": text}]}})
    p = d / f"{sid}.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def convo(n, start, step=60):
    return [("u" if i % 2 == 0 else "a", start + i * step, f"第{i}句：周末去果园摘苹果的安排") for i in range(n)]


class HandoffBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name) / "t"
        self.d.mkdir()
        self.out = Path(self.tmp.name) / "out"
        self.now = time.time()

    def tearDown(self):
        self.tmp.cleanup()

    def test_last_n_turns_verbatim_and_compact_summary_reused(self):
        write_session(self.d, "prev", convo(40, self.now - 7200), self.now - 3000, summary="话题\n- 果园——定了周六去")
        write_session(self.d, "cur", [("u", self.now, "新窗口第一句")], self.now)
        text = H.run(self.d, self.out, "cur", call=lambda s, u: self.fail("有 /compact 摘要时不该调模型"))
        self.assertIn("【上个窗口的 /compact 摘要】", text)
        self.assertIn("果园——定了周六去", text)
        self.assertIn(f"最后 {H.KEEP_TURNS} 轮原话（一字未改）", text)
        self.assertIn("第39句：周末去果园摘苹果的安排", text)
        self.assertNotIn("第29句", text)
        self.assertNotIn("新窗口第一句", text)
        self.assertTrue((self.out / "handoff.txt").exists())

    def test_model_summary_when_no_compact_and_cached(self):
        write_session(self.d, "prev", convo(40, self.now - 7200), self.now - 3000)
        calls = []
        def fake(system, user):
            calls.append(user)
            return "话题\n" + "\n".join(f"- 话题{i}" for i in range(15)) + "\n悬着的事\n- 订门票\n近况\n- 无\n一句话概要\n- 摘苹果"
        t1 = H.run(self.d, self.out, "cur", call=fake)
        self.assertIn("模型生成", t1)
        self.assertIn("- 话题9", t1)
        self.assertNotIn("- 话题10", t1)            # 话题段硬上限 10 条
        H.run(self.d, self.out, "cur", call=fake)
        self.assertEqual(len(calls), 1)            # 第二次走缓存

    def test_short_window_skipped_for_substantive_one(self):
        write_session(self.d, "big", convo(40, self.now - 7200), self.now - 3000, summary="话题\n- 大窗口")
        write_session(self.d, "short", convo(4, self.now - 600), self.now - 500)
        prev, skipped = H.pick_previous(self.d, "cur")
        self.assertEqual(prev.id, "big")
        self.assertEqual([w.id for w in skipped], ["short"])

    def test_substantive_window_too_old_falls_back_to_short(self):
        write_session(self.d, "old", convo(40, self.now - 5 * 86400), self.now - 5 * 86400 + 4000, summary="话题\n- 旧")
        write_session(self.d, "short", convo(4, self.now - 600), self.now - 500)
        prev, _ = H.pick_previous(self.d, "cur")
        self.assertEqual(prev.id, "short")

    def test_side_channel_split_and_dedupe(self):
        turns = convo(40, self.now - 7200)
        write_session(self.d, "prev", turns, self.now - 3000)
        side = Path(self.tmp.name) / "side.jsonl"
        last_user = [t for t in turns if t[0] == "u"][-1]
        rows = [{"ts": self.now - 9000, "from": "me", "text": "更早的一条"},
                {"ts": last_user[1] + 2, "from": "me", "text": last_user[2]},          # 与原话重复
                {"ts": turns[-1][1] + 5, "from": "bot", "text": "晚于原话的一条"},
                {"ts": self.now + 3600, "from": "me", "text": "新窗口之后的不该出现"}]
        side.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        os.environ["ENGRAM_HANDOFF_SIDE_FILE"] = str(side)
        try:
            text = H.run(self.d, self.out, "cur", call=lambda s, u: None)
        finally:
            del os.environ["ENGRAM_HANDOFF_SIDE_FILE"]
        self.assertIn("更早的一条", text)
        self.assertIn("晚于原话的一条", text)
        self.assertNotIn("新窗口之后的不该出现", text)
        self.assertIn("已去掉与最后 10 轮重复的 1 条", text)
        self.assertLess(text.index("更早的一条"), text.index("一字未改"))
        self.assertGreater(text.index("晚于原话的一条"), text.index("一字未改"))

    def test_system_and_tool_messages_are_not_turns(self):
        p = self.d / "x.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in [
            {"type": "user", "timestamp": _iso(self.now), "message": {"content": "<system-reminder>x</system-reminder>"}},
            {"type": "user", "timestamp": _iso(self.now), "message": {"content": [{"type": "tool_result", "content": "ok"}]}},
            {"type": "assistant", "timestamp": _iso(self.now), "message": {"content": "NO_REPLY"}},
            {"type": "user", "timestamp": _iso(self.now + 1), "message": {"content": "真正的一句"}},
        ]), encoding="utf-8")
        w = H.read_window(p)
        self.assertEqual([t.text for t in w.turns], ["真正的一句"])

    def test_scrub_and_cap(self):
        os.environ["ENGRAM_HANDOFF_SCRUB_REGEX"] = "秘密"
        try:
            self.assertEqual(H.scrub("- 聊了秘密计划，定了时间"), "- （略），定了时间")
        finally:
            del os.environ["ENGRAM_HANDOFF_SCRUB_REGEX"]
        capped = H.cap_sections("悬着的事\n" + "\n".join(f"- {i}" for i in range(8)))
        self.assertEqual(capped.count("\n- "), 5)


if __name__ == "__main__":
    unittest.main()
