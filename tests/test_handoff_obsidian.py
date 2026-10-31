"""2026-10-29：交接注入钩子 + Obsidian 导出/收集 + 夜间归纳的模型守卫。夹具全虚构。"""
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "maintenance"))
import hooks_handoff_inject as H  # noqa: E402
import obsidian_export as OX  # noqa: E402
import obsidian_collect as OC  # noqa: E402
import cluster_summarize as CS  # noqa: E402
import persona  # noqa: E402
from store.store import add_evidence, confirm, connect, promote  # noqa: E402

U = persona.USER

WIN = """---
session: 11111111-2222-3333-4444-555555555555
last_message: 2099-01-02 03:04
brief: 虚构
---
# 窗口 11111111

## 摘要

话题
- 聊了做蛋糕的计划
- 讨论了周末去海边

悬着的事
- 小红的生日礼物还没买

近况
- 最近睡得不太好

一句话概要
- 虚构一句话

## 最后 10 轮对话
我：这是原话，不该被注入
"""

TXT = """【跨窗口接力｜北京时间 2099-01-02 04:00】
来源会话 id：99999999-2222-3333-4444-555555555555
最后一条消息时间：2099/01/02 03:04:05（北京时间）

【上个窗口的摘要（模型生成）】
话题
- 虚构的话题甲
悬着的事
- 虚构的待办乙
近况
- 虚构的近况丙

【上个窗口最后 10 轮原话（一字未改）】
我：原话不该出现
"""

MANUAL = """【摘要】
这个窗口聊过什么
- 手写话题
悬着的事
- 手写待办
她的状态
- 手写状态

【最后 10 轮对话】
我：不该出现
"""


def mk_handoff(d: Path, windows=True, txt=True):
    if windows:
        (d / "windows").mkdir()
        (d / "windows" / "2099-01-02_11111111.md").write_text(WIN, encoding="utf-8")
        (d / "windows" / "index.md").write_text(
            "# 索引\n- [2099-01-02 03:04] 11111111 — 虚构 — 2099-01-02_11111111.md\n"
            "- [2098-12-31 03:04] 00000000 — 旧 — 2098-12-31_00000000.md\n", encoding="utf-8")
    if txt:
        (d / "handoff.txt").write_text(TXT, encoding="utf-8")


class HookParse(unittest.TestCase):
    def test_sections_from_window_md(self):
        s = H.parse_sections(WIN)
        self.assertEqual(s["summary"], ["聊了做蛋糕的计划", "讨论了周末去海边"])
        self.assertEqual(s["pending"], ["小红的生日礼物还没买"])
        self.assertEqual(s["state"], ["最近睡得不太好"])
        self.assertNotIn("原话", "".join(sum(s.values(), [])))

    def test_sections_from_manual_and_txt(self):
        s = H.parse_sections(MANUAL)
        self.assertEqual((s["summary"], s["pending"], s["state"]), (["手写话题"], ["手写待办"], ["手写状态"]))
        t = H.parse_sections(TXT)
        self.assertEqual((t["summary"], t["pending"], t["state"]), (["虚构的话题甲"], ["虚构的待办乙"], ["虚构的近况丙"]))

    def test_build_has_label_and_cap_keeps_pending(self):
        sec = {"summary": ["很长的话题" * 20] * 5, "pending": ["要紧的待办"], "state": ["状态"], "brief": []}
        out = H.build(sec, "2099-01-02", 200)
        self.assertLessEqual(len(out), 200)
        self.assertIn("不是她刚说的话", out)
        self.assertIn("要紧的待办", out)

    def test_empty_sections_build_nothing(self):
        self.assertEqual(H.build({"summary": [], "pending": [], "state": [], "brief": []}, "", 500), "")


class HookRun(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.d = Path(self.td.name)
        self.env = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_HANDOFF_INJECT": "1",
                    "ENGRAM_HANDOFF_DIR": str(self.d), "ENGRAM_HOME": self.td.name}

    def run_hook(self, stdin="{}", extra=None):
        with mock.patch.dict(os.environ, {**self.env, **(extra or {})}, clear=True):
            return H.run(stdin)

    def test_flag_off_injects_nothing(self):
        mk_handoff(self.d)
        self.assertEqual(self.run_hook(extra={"ENGRAM_HANDOFF_INJECT": "0"}), "")

    def test_uses_newest_in_index(self):
        mk_handoff(self.d)
        out = self.run_hook()
        self.assertIn("小红的生日礼物还没买", out)
        self.assertNotIn("原话", out)

    def test_falls_back_to_handoff_txt_when_windows_unreadable(self):
        mk_handoff(self.d, windows=False)
        (self.d / "windows").mkdir()
        (self.d / "windows" / "index.md").write_text("- [x] a — b — missing.md\n", encoding="utf-8")
        out = self.run_hook()
        self.assertIn("虚构的待办乙", out)

    def test_nothing_readable_is_silent(self):
        self.assertEqual(self.run_hook(), "")

    def test_compact_and_same_session_skipped(self):
        mk_handoff(self.d)
        self.assertEqual(self.run_hook(json.dumps({"source": "compact"})), "")
        self.assertEqual(self.run_hook(json.dumps({"session_id": "11111111-2222-3333-4444-555555555555"})), "")

    def test_too_old_skipped(self):
        mk_handoff(self.d, windows=False)
        (self.d / "handoff.txt").write_text(TXT.replace("2099/01/02", "2001/01/02"), encoding="utf-8")
        self.assertEqual(self.run_hook(), "")

    def test_secret_scrubbed(self):
        mk_handoff(self.d, windows=False)
        key = "sk-" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8Qr9"
        (self.d / "handoff.txt").write_text(TXT.replace("虚构的待办乙", "记住 key " + key), encoding="utf-8")
        out = self.run_hook()
        self.assertNotIn(key, out)

    def test_main_swallows_errors_and_prints_nothing_extra(self):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(H, "run", side_effect=ValueError("x")), \
             mock.patch.object(H, "_ROOT", self.td.name), \
             mock.patch("sys.stdin", io.StringIO("{}")), redirect_stdout(buf):
            self.assertEqual(H.main(), 0)
        self.assertEqual(buf.getvalue(), "")

    def test_flag_registered_default_off(self):
        import flags
        self.assertFalse(flags.DEFAULTS["HANDOFF_INJECT"])


class ObsidianBase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.d = Path(self.td.name)
        self.db = self.d / "t.db"
        self.con = connect(self.db)
        self.addCleanup(self.con.close)
        self.vault = self.d / "vault"
        self.ids = {}
        for i, (txt, typ) in enumerate([("喜欢红色的杯子", "preference"), ("周三去了动物园", "episodic"),
                                         ("养了一只叫团子的猫\n第二行", "semantic")]):
            eid = add_evidence(self.con, source_kind="chat", source_actor=U, source_locator=f"conv:t#msg:{i}",
                               content=txt, occurred_at=f"2099-01-0{i + 1}")
            fid = promote(self.con, subject=U, fact_type=typ, content=txt, evidence_ids=[eid], status="candidate",
                          valid_from=f"2099-01-0{i + 1}", importance=0.5, confidence=0.7)
            self.ids[i] = fid
        confirm(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (self.ids[0],)).fetchone()[0])
        self.con.commit()

    def export(self):
        with redirect_stdout(io.StringIO()):
            OX.main(["--vault", str(self.vault), "--db", str(self.db), "--handoff-dir", str(self.d / "ho")])

    def tick(self, fid):
        for p in (self.vault / "事实").glob("*.md"):
            t = p.read_text(encoding="utf-8")
            if f"^f{fid}" in t:
                lines = [(l.replace("- [ ]", "- [x]", 1) if l.endswith(f"^f{fid}") else l) for l in t.split("\n")]
                p.write_text("\n".join(lines), encoding="utf-8")
                return
        raise AssertionError("fact not in vault")


class Export(ObsidianBase):
    def test_export_is_read_only_and_shapes_lines(self):
        before = self.db.read_bytes()
        self.export()
        self.assertEqual(self.db.read_bytes(), before)
        body = "\n".join(p.read_text(encoding="utf-8") for p in (self.vault / "事实").glob("*.md"))
        self.assertIn(f"- 🔒 喜欢红色的杯子", body)                      # 已确认：没有复选框
        self.assertRegex(body, rf"- \[ \] 周三去了动物园 · 2099-01-02 \^f{self.ids[1]}")
        self.assertIn("养了一只叫团子的猫 ⏎ 第二行", body)               # 换行折成一行
        self.assertTrue((self.vault / "README.md").exists())

    def test_reexport_keeps_ticks(self):
        self.export()
        self.tick(self.ids[1])
        self.export()
        body = "\n".join(p.read_text(encoding="utf-8") for p in (self.vault / "事实").glob("*.md"))
        self.assertRegex(body, rf"- \[x\] 周三去了动物园.*\^f{self.ids[1]}")

    def test_unchanged_reexport_touches_no_file(self):
        self.export()
        snap = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (self.vault / "事实").glob("*.md")}
        self.export()
        self.assertEqual(snap, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in (self.vault / "事实").glob("*.md")})
        self.assertEqual(json.loads((self.vault / ".engram-export.json").read_text())["files_written"], 0)

    def test_new_fact_only_changes_newest_bucket_file(self):
        self.export()
        before = {p.name: p.read_bytes() for p in (self.vault / "事实").glob("*.md")}
        with mock.patch.object(OX, "BUCKET", 2):
            self.export()                                   # 换成小桶：旧的大桶文件应被清掉，新桶按 id 段分
            names = {p.name for p in (self.vault / "事实").glob("*.md")}
        self.assertNotEqual(names, set(before))
        self.assertTrue(all("_" in n for n in names))

    def test_handoff_mirror_skips_unreadable(self):
        ho = self.d / "ho"
        ho.mkdir()
        mk_handoff(ho)
        real = Path.read_text

        def fake(self_, *a, **k):
            if self_.name == "2099-01-02_11111111.md":
                raise PermissionError("nope")
            return real(self_, *a, **k)
        with mock.patch.object(Path, "read_text", fake):
            self.export()
        self.assertTrue((self.vault / "交接" / "handoff.md").exists())
        self.assertFalse((self.vault / "交接" / "windows" / "2099-01-02_11111111.md").exists())
        st = json.loads((self.vault / ".engram-export.json").read_text())
        self.assertEqual(st["handoff_unreadable"], 1)

    def test_missing_handoff_dir_is_fine(self):
        self.export()
        self.assertFalse((self.vault / "交接").exists())


class Collect(ObsidianBase):
    def run_collect(self, *a):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = OC.main(["--vault", str(self.vault), "--db", str(self.db),
                          "--archive-dir", str(self.d / "arch"), *a])
        return rc, buf.getvalue()

    def status(self, i):
        return self.con.execute("SELECT status FROM facts WHERE id=?", (self.ids[i],)).fetchone()[0]

    def test_dry_run_default_changes_nothing(self):
        self.export()
        self.tick(self.ids[1])
        rc, out = self.run_collect("--force")
        self.assertEqual(rc, 0)
        self.assertIn("dry-run", out)
        self.assertEqual(self.status(1), "candidate")

    def test_apply_archives_reversibly_and_never_confirms(self):
        self.export()
        self.tick(self.ids[1])
        rc, out = self.run_collect("--apply", "--force")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.status(1), "uncertain")
        row = self.con.execute("SELECT * FROM facts_archive WHERE fact_id=?", (self.ids[1],)).fetchone()
        self.assertEqual((row["action"], row["prev_status"], row["category"]), ("archived", "candidate", "obsidian_flag"))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE id=?", (self.ids[1],)).fetchone()[0], 1)
        self.assertTrue((self.d / "arch" / f"{row['batch_id']}-obsidian.jsonl").exists())
        rc, out = self.run_collect("--restore", row["batch_id"])
        self.assertEqual(self.status(1), "candidate")
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts_archive WHERE action='restored'").fetchone()[0], 1)

    def test_confirmed_refused_even_if_ticked(self):
        self.export()
        for p in (self.vault / "事实").glob("*.md"):
            t = p.read_text(encoding="utf-8")
            if f"^f{self.ids[0]}" in t:
                p.write_text(t.replace("- 🔒 喜欢", "- [x] 喜欢"), encoding="utf-8")
        rc, out = self.run_collect("--apply", "--force")
        self.assertIn("拒绝(confirmed/钉住) 1", out)
        self.assertEqual(self.status(0), "confirmed")

    def test_mass_tick_refused_without_force(self):
        self.export()
        self.tick(self.ids[1])
        self.tick(self.ids[2])
        rc, out = self.run_collect("--apply")
        self.assertEqual(rc, 2)
        self.assertEqual(self.status(1), "candidate")


class ClusterGuard(unittest.TestCase):
    def test_apply_refused_without_cluster_model(self):
        buf = io.StringIO()
        env = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_CLUSTER_SUMMARY": "1"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("store.store.connect", side_effect=AssertionError("must not connect")), redirect_stdout(buf):
            self.assertEqual(CS.main(["--apply"]), 2)
        self.assertIn("CLUSTER_MODEL", buf.getvalue())

    def test_dry_run_and_configured_model_pass_the_guard(self):
        env = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_CLUSTER_SUMMARY": "1", "CLUSTER_MODEL": "fake/flash"}
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(CS, "run", return_value={"calls": 0}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(CS.main(["--db", str(Path(td) / "x.db")]), 0)
            self.assertEqual(CS.main(["--db", str(Path(td) / "x.db"), "--apply"]), 0)
        self.assertEqual(run.call_count, 2)

    def test_cluster_model_is_first_choice_for_consolidate(self):
        import llm
        self.assertEqual(llm.PURPOSE_MODEL_VARS["consolidate"][0], "CLUSTER_MODEL")

    def test_nightly_script_caps_calls_at_20_by_default(self):
        t = (ROOT / "maintenance" / "nightly-cluster.sh").read_text(encoding="utf-8")
        self.assertIn("--max-calls", t)
        self.assertIn(":-20", t)


if __name__ == "__main__":
    unittest.main()
