"""任务二·步骤1：共用模型调用 llm.py 的主/备用路由（全部 mock，不联网、不读真 key）"""
import io
import json
import os
import socket
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm  # noqa: E402

PRIMARY = {"DIGEST_API_KEY": "pk", "DIGEST_BASE_URL": "https://primary.test/v1/chat/completions",
           "DIGEST_MODEL": "pm", "RERANK_MODEL": "pm-rerank", "ENGRAM_ENV_FILE": "/nonexistent"}
FALLBACK = {"FALLBACK_API_KEY": "fk", "FALLBACK_BASE_URL": "https://api.fallback.test",
            "FALLBACK_MODEL": "fm"}


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def ok_resp(text="{\"x\":1}", usage=None):
    return Resp(json.dumps({"choices": [{"message": {"content": text}}],
                            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2}}).encode())


def http_err(code):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))


class FakeNet:
    """按顺序给每次请求一个结果（异常或响应），记下每次请求的 url/model/key。"""
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, req, timeout):
        body = json.loads(req.data)
        self.calls.append({"url": req.full_url, "model": body["model"],
                           "auth": req.headers.get("Authorization"), "body": body})
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class LLMRoutes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = mock.patch.object(llm, "LOG_FILE", Path(self.tmp.name) / "route.log")
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_chat(self, env, net, **kw):
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(llm, "_urlopen", net):
            return llm.chat_ex([{"role": "user", "content": "hi"}], **kw)

    def test_primary_ok_no_fallback_call(self):
        net = FakeNet(ok_resp())
        r = self.run_chat({**PRIMARY, **FALLBACK}, net, purpose="rerank")
        self.assertEqual((r.route, r.model, len(net.calls)), ("primary", "pm-rerank", 1))
        self.assertEqual(r.usage["prompt_tokens"], 3)
        self.assertTrue(net.calls[0]["body"]["enable_thinking"] is False)

    def test_fallback_on_retryable(self):
        for err in (http_err(402), http_err(401), http_err(503), http_err(429),
                    socket.timeout("t"), urllib.error.URLError("conn refused"),
                    ConnectionResetError()):
            with self.subTest(err=repr(err)):
                net = FakeNet(err, ok_resp("{\"y\":2}"))
                r = self.run_chat({**PRIMARY, **FALLBACK}, net)
                self.assertEqual(r.route, "fallback")
                self.assertEqual(r.text, "{\"y\":2}")
                self.assertEqual(net.calls[1]["url"], "https://api.fallback.test/chat/completions")
                self.assertEqual(net.calls[1]["model"], "fm")
                self.assertEqual(net.calls[1]["auth"], "Bearer fk")
                self.assertNotIn("enable_thinking", net.calls[1]["body"])   # 官方接口不带硅基专属参数

    def test_bad_body_falls_back(self):
        net = FakeNet(Resp(b'{"error":"x"}'), ok_resp())
        self.assertEqual(self.run_chat({**PRIMARY, **FALLBACK}, net).route, "fallback")

    def test_400_does_not_fallback(self):
        net = FakeNet(http_err(400))
        with self.assertRaises(urllib.error.HTTPError):
            self.run_chat({**PRIMARY, **FALLBACK}, net)
        self.assertEqual(len(net.calls), 1)

    def test_no_fallback_configured_behaves_as_before(self):
        net = FakeNet(http_err(402))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.run_chat(PRIMARY, net)
        self.assertEqual(cm.exception.code, 402)
        self.assertEqual(len(net.calls), 1)

    def test_both_fail_raises_once_each(self):
        net = FakeNet(http_err(402), http_err(500))
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.run_chat({**PRIMARY, **FALLBACK}, net)
        self.assertEqual(cm.exception.code, 500)
        self.assertEqual(len(net.calls), 2)          # 只重试一次，不来回弹

    def test_nothing_configured(self):
        with self.assertRaises(llm.LLMNotConfigured):
            self.run_chat({"ENGRAM_ENV_FILE": "/nonexistent"}, FakeNet())

    def test_only_fallback_configured_uses_it(self):
        net = FakeNet(ok_resp())
        r = self.run_chat({"ENGRAM_ENV_FILE": "/nonexistent", **FALLBACK}, net)
        self.assertEqual(r.route, "fallback")

    def test_env_file_is_read(self):
        f = Path(self.tmp.name) / "e.env"
        f.write_text('DIGEST_API_KEY="a"\nDIGEST_BASE_URL=https://p.test/c\nDIGEST_MODEL=m\n')
        net = FakeNet(ok_resp())
        r = self.run_chat({"ENGRAM_ENV_FILE": str(f)}, net, purpose="extract")
        self.assertEqual((r.route, net.calls[0]["url"], net.calls[0]["model"]),
                         ("primary", "https://p.test/c", "m"))

    def test_extract_model_override(self):
        net = FakeNet(ok_resp())
        self.run_chat({**PRIMARY, "EXTRACT_MODEL": "em"}, net, purpose="extract")
        self.assertEqual(net.calls[0]["model"], "em")

    def test_route_log_written_without_secrets(self):
        net = FakeNet(http_err(402), ok_resp())
        self.run_chat({**PRIMARY, **FALLBACK}, net, purpose="rerank")
        log = (Path(self.tmp.name) / "route.log").read_text()
        self.assertIn("rerank primary FAIL HTTP 402", log)
        self.assertIn("rerank fallback ok", log)
        self.assertNotIn("pk", log.replace("primary", "")); self.assertNotIn("fk", log)

    def test_chat_never_prints(self):
        net = FakeNet(http_err(402), ok_resp())
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf), mock.patch("sys.stderr", buf):
            self.run_chat({**PRIMARY, **FALLBACK}, net)
        self.assertEqual(buf.getvalue(), "")


class CallersUseSharedFunction(unittest.TestCase):
    """调用方都走 llm.chat，主路由 402 时能由备用路由拿到结果（rerank / newer / timeline）"""
    def test_rerank_judge_via_fallback(self):
        import rerank
        net = FakeNet(http_err(402), ok_resp('{"need": true, "scores": {"0": 3}}'))
        with mock.patch.dict(os.environ, {**PRIMARY, **FALLBACK}, clear=True), \
                mock.patch.object(llm, "_urlopen", net), \
                mock.patch.object(llm, "LOG_FILE", Path(tempfile.gettempdir()) / "t-route.log"):
            self.assertEqual(rerank.judge("q", ["a"]), {"need": True, "scores": [3]})

    def test_rerank_judge_unconfigured_returns_none(self):
        import rerank
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True):
            self.assertIsNone(rerank.judge("q", ["a"]))

    def test_timeline_unconfigured_message(self):
        from extract import timeline
        # 步骤4 起 timeline 默认冻结；这条测的是「开了之后没配模型」的老语义
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_TIMELINE": "1"}, clear=True):
            self.assertEqual(timeline.judge("n", "2026-01-01", [{"content": "o", "valid_from": ""}]),
                             (None, "没配模型"))

    def test_from_evidence_unconfigured_raises_runtimeerror(self):
        from extract import from_evidence
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True):
            with self.assertRaises(RuntimeError):
                from_evidence.call_model("x")

    def test_no_direct_urlopen_left_in_callers(self):
        root = Path(__file__).resolve().parent.parent
        for f in ("rerank.py", "newer.py", "corrections.py", "extract/timeline.py",
                  "extract/from_evidence.py"):
            self.assertNotIn("urlopen", (root / f).read_text(encoding="utf-8"), f)


if __name__ == "__main__":
    unittest.main()
