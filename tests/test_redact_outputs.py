"""密钥打码接在哪儿：入库（证据/事实）、送模型（llm/embedding）、出口（钩子/MCP/遥测）。密钥都是随机假值。"""
import importlib.util
import io
import json
import random
import string
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402
import redact  # noqa: E402
from common import sha256  # noqa: E402
from store.store import add_evidence, connect, promote  # noqa: E402

_r = random.Random(11)
B62 = string.ascii_letters + string.digits


def key():
    while True:
        s = "".join(_r.choice(B62) for _ in range(35))
        if any(c.isdigit() for c in s) and any(c.isupper() for c in s) and any(c.islower() for c in s):
            return "sk-" + s


SECRET = key()
TEXT = f"这是我的 key：{SECRET}，你先记着"


class Outputs(unittest.TestCase):
    def test_hook_log_redacts(self):
        spec = importlib.util.spec_from_file_location("hook_under_test", ROOT / "hooks_engram_recall.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        with tempfile.TemporaryDirectory() as d:
            hook.LOG = str(Path(d) / "hook.log")
            hook.log(f"不查 | {TEXT}")
            body = Path(hook.LOG).read_text(encoding="utf-8")
        self.assertNotIn(SECRET, body)
        self.assertIn("不查", body)

    def test_mcp_output_redacted(self):
        import mcp_server
        with mock.patch.object(mcp_server, "do_recall", lambda con, a: "找到：" + TEXT):
            r = mcp_server.handle("tools/call", {"name": "engram_recall", "arguments": {"text": "任意话题的一句话"}}, None)
        out = r["content"][0]["text"]
        self.assertNotIn(SECRET, out)
        self.assertIn("〔已打码:sk:", out)

    def test_telemetry_query_redacted(self):
        from telemetry import Run
        run = Run(None, query=TEXT, decision="skip")
        self.assertNotIn(SECRET, run.query)

    def test_hook_scrub_failure_injects_nothing(self):
        """出口扫描出问题：一个字都不注入（源码级断言：except 分支 return 0，且在 print 之前）。"""
        src = (ROOT / "hooks_engram_recall.py").read_text(encoding="utf-8")
        i = src.index("from redact import scrub")
        j = src.index("print(out_text)")
        self.assertLess(i, j)
        self.assertIn("return 0", src[i:j])
        self.assertNotIn('print("\\n".join(lines))', src)


if __name__ == "__main__":
    unittest.main()
