"""2026-10-28 清库配套补丁的回归测试。全部用临时库/临时文件，不碰 engram.db。"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from salience import COLD_WEIGHT, activation  # noqa: E402
from store.store import connect, promote, search  # noqa: E402

SPEC = importlib.util.spec_from_file_location("engram_hook_p", ROOT / "hooks_engram_recall.py")
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class 冷层降权(unittest.TestCase):
    def test_cold_is_scaled_not_filtered(self):
        with tempfile.TemporaryDirectory() as d:
            con = connect(Path(d) / "t.db")
            a = promote(con, subject="tester", fact_type="semantic", content="小测：星光湖那次看了蓝脚印（甲）",
                        importance=0.6, confidence=0.6, valid_from="2026-08-17")
            b = promote(con, subject="tester", fact_type="semantic", content="小测：星光湖那次捡了贝壳（乙）",
                        importance=0.6, confidence=0.6, valid_from="2026-08-17")
            base = {r["ext_id"]: r["score"] for r in search(con, "星光湖", limit=5)}
            con.execute("UPDATE facts SET layer='cold' WHERE id=?", (b,))
            con.commit()
            rows = search(con, "星光湖", limit=5)
            self.assertEqual(len(rows), 2, "冷层不应被排除")
            sc = {r["layer"]: r["score"] for r in rows}
            self.assertAlmostEqual(sc["cold"], base[rows[-1]["ext_id"]] * COLD_WEIGHT, places=3)
            self.assertEqual(rows[-1]["layer"], "cold")


class 无时间退到入库时间(unittest.TestCase):
    def test_null_valid_from_uses_created_at(self):
        with tempfile.TemporaryDirectory() as d:
            con = connect(Path(d) / "t.db")
            fid = promote(con, subject="claude", fact_type="episodic", content="我：这是一条没有时间的原话用来测试衰减",
                          importance=0.5, confidence=0.5)
            row = con.execute("""SELECT julianday('now') - julianday(COALESCE(valid_from, created_at)) AS age,
                                        julianday(valid_from) AS vf FROM facts WHERE id=?""", (fid,)).fetchone()
            self.assertIsNone(row["vf"])
            self.assertIsNotNone(row["age"], "valid_from 为空时年龄应退到 created_at")
            self.assertGreaterEqual(row["age"], -1e-3)   # created_at 按秒取整，可能比 now 晚不到 1 秒
            self.assertNotEqual(activation(importance=0.5, confidence=0.5, emotion=None, access_count=0,
                                           age_days=row["age"], user_pinned=False, fact_type="episodic"), None)


class 助理甲侧不直接晋升(unittest.TestCase):
    def test_claude_reply_kept_as_evidence_only(self):
        from extract import from_channel as fc
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            rec = {"message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "mcp__telegram__reply",
                 "input": {"text": "以后一律纯文本回复，不用 markdown，这是我跟小测说好的规矩。"}}]}}
            p.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            con = connect(Path(d) / "t.db")
            orig_connect, orig_argv = fc.connect, sys.argv
            fc.connect = lambda *a, **k: con
            try:
                sys.argv = ["from_channel.py", str(p)]
                fc.main()
            finally:
                fc.connect, sys.argv = orig_connect, orig_argv
            ev = con.execute("SELECT COUNT(*) FROM evidence WHERE source_actor='claude'").fetchone()[0]
            fx = con.execute("SELECT COUNT(*) FROM facts WHERE subject='claude'").fetchone()[0]
            self.assertEqual(ev, 1, "证据照存")
            self.assertEqual(fx, 0, "助理甲自己的回复不该直接成 fact")


class Hook收紧(unittest.TestCase):
    def test_strip_wrappers_removes_attachment_boilerplate(self):
        t = "(Images also saved locally: /home/tester/workspace/cc-connect/attachments/img_1_2.jpg)"
        self.assertEqual(hook.strip_wrappers(t), "")
        self.assertEqual(hook.strip_wrappers("(photo)"), "")
        self.assertEqual(hook.strip_wrappers("<task-notification>done</task-notification>"), "")
        self.assertIn("今天吃了炸鸡", hook.strip_wrappers("(photo)\n今天吃了炸鸡"))

    def test_boilerplate_terms_are_not_distinctive(self):
        t = "workspace processed notification photo attachment ubuntu"
        self.assertEqual(hook.distinctive_terms(t), [])
        self.assertIn("DeepSeek", hook.distinctive_terms("DeepSeek 又 402 了"))

    def test_single_strong_term_no_longer_enough(self):
        self.assertFalse(hook.keyword_relevant(["助理甲"], "助理甲：随便一句带人名的话"))
        self.assertTrue(hook.keyword_relevant(["助理甲", "记忆库"], "助理甲：记忆库清理"))
        self.assertTrue(hook.keyword_relevant(["助理甲", "小黄"], "助理甲：小黄是狗"))
        self.assertTrue(hook.keyword_relevant(["记忆库", "交接"], "记忆库交接页"))
        self.assertFalse(hook.keyword_relevant(["记忆库"], "记忆库"))

    def test_header_comment_has_no_stale_counts(self):
        src = (ROOT / "hooks_engram_recall.py").read_text(encoding="utf-8")
        self.assertNotIn("7942", src)


if __name__ == "__main__":
    unittest.main()
