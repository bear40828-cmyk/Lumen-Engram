"""收尾A6：日志轮转（>2MB 切）。临时目录里跑，不碰正式日志。"""
import gzip
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "maintenance" / "rotate_logs.sh"


def run(d, mx):
    return subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True,
                          env={**os.environ, "ENGRAM_DIR": str(d), "ENGRAM_LOG_MAX_BYTES": str(mx)})


class Rotate(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.d = Path(self.t.name)

    def test_syntax(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)

    def test_small_files_untouched(self):
        (self.d / "hook.log").write_text("x" * 100)
        r = run(self.d, 1000)
        self.assertEqual(r.returncode, 0)
        self.assertEqual((self.d / "hook.log").read_text(), "x" * 100)
        self.assertEqual(list((self.d / "archive" / "logs").glob("*")), [])

    def test_big_file_rotated_gzipped_and_recreated(self):
        body = ("2026-10-28 00:00:00 一行日志\n" * 200)
        (self.d / "hook.log").write_text(body)
        (self.d / "llm-route.log").write_text("short\n")
        r = run(self.d, 500)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("轮转 hook.log", r.stdout)
        self.assertTrue((self.d / "hook.log").exists())               # 同名文件还在（读日志的不会撞空）
        self.assertEqual((self.d / "hook.log").stat().st_size, 0)
        gz = list((self.d / "archive" / "logs").glob("hook.log.*.gz"))
        self.assertEqual(len(gz), 1)
        self.assertEqual(gzip.open(gz[0], "rt", encoding="utf-8").read(), body)    # 整份还在，没丢
        self.assertEqual(oct(gz[0].stat().st_mode & 0o777), "0o640")
        self.assertEqual((self.d / "llm-route.log").read_text(), "short\n")        # 没超的不动

    def test_writer_appending_by_name_still_works_after_rotation(self):
        (self.d / "hook.log").write_text("a" * 800)
        run(self.d, 500)
        with open(self.d / "hook.log", "a", encoding="utf-8") as fh:   # 钩子的写法：按名字追加
            fh.write("新的一行\n")
        self.assertEqual((self.d / "hook.log").read_text(), "新的一行\n")

    def test_missing_files_ok(self):
        self.assertEqual(run(self.d, 1).returncode, 0)

    def test_hook_and_llm_open_by_name_append(self):
        for f, needle in (("hooks_engram_recall.py", 'open(LOG, "a"'), ("llm.py", 'open(LOG_FILE, "a"')):
            self.assertIn(needle, (ROOT / f).read_text(encoding="utf-8"), f)    # 轮转安全的前提


if __name__ == "__main__":
    unittest.main()
