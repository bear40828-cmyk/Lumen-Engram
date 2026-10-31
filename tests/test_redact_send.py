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


class SendToModel(unittest.TestCase):
    def test_llm_post_body_has_no_secret(self):
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(req, timeout):
            seen["body"] = req.data.decode()
            seen["auth"] = req.get_header("Authorization")
            return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}}).encode())

        route = llm.Route("primary", "http://x/chat/completions", "route-key-123", "m")
        with mock.patch.object(llm, "_urlopen", fake):
            llm._post(route, [{"role": "system", "content": "只输出 JSON"},
                              {"role": "user", "content": TEXT}], temperature=0, max_tokens=10, json_mode=True, timeout=1)
        sent = json.loads(seen["body"])["messages"][1]["content"]
        self.assertNotIn(SECRET, seen["body"])
        self.assertNotIn(SECRET, sent)
        self.assertIn("〔已打码:sk:", sent)
        self.assertEqual(seen["auth"], "Bearer route-key-123")       # 自己的路由 key 不受影响

    def test_embed_text_has_no_secret(self):
        try:
            import numpy  # noqa: F401
        except ImportError:                       # 这台机器没装 numpy：塞个空壳，只验请求体
            import types
            sys.modules["numpy"] = types.ModuleType("numpy")
        import vectors
        seen = {}

        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake(req, timeout=None):
            seen["body"] = req.data.decode()
            return Resp(json.dumps({"result": {"data": [[0.1, 0.2]]}}).encode())

        with mock.patch.object(vectors, "_creds", lambda: ("t", "a")), mock.patch.object(vectors.urllib.request, "urlopen", fake):
            vectors.embed([TEXT])
        self.assertNotIn(SECRET, seen["body"])
        self.assertIn("〔已打码:sk:", json.loads(seen["body"])["text"][0])


if __name__ == "__main__":
    unittest.main()
