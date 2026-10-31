"""密钥打码：真阳性 + 假阳性 + 幂等 + 日志不含原文。密钥全是随机生成的假值，不是真的。"""
import os
import random
import string
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import redact as R  # noqa: E402

_r = random.Random(7)
B62 = string.ascii_letters + string.digits


def rnd(n, alphabet=B62):
    while True:
        s = "".join(_r.choice(alphabet) for _ in range(n))
        if any(c.isdigit() for c in s) and any(c.isupper() for c in s) and any(c.islower() for c in s):
            return s


SK = "sk-" + rnd(32)
GHPAT = "github_pat_" + rnd(40, B62 + "_")
GHP = "ghp_" + rnd(36)
SLACK = "xoxb-" + "-".join(rnd(n) for n in (12, 12, 24))
AKIA = "AKIA" + "".join(_r.choice(string.ascii_uppercase + string.digits) for _ in range(16))
TG = "".join(_r.choice(string.digits) for _ in range(9)) + ":" + rnd(35, B62 + "_-")
BARK = rnd(22)
BEARER = rnd(40, B62 + "._-")
LONG43 = rnd(43)


class TruePositives(unittest.TestCase):
    def check(self, kind, secret, template="她说：{}，你记一下"):
        out, c = R.redact(template.format(secret))
        self.assertNotIn(secret, out)
        self.assertEqual(c.get(kind), 1, (kind, c, out))
        self.assertIn(f"〔已打码:{kind}:{secret[:4]}…{secret[-2:]}〕", out)

    def test_sk(self):
        self.check("sk", SK)
        self.check("sk", "sk-ant-api03-" + rnd(40, B62 + "_-"))      # 长的、带连字符的

    def test_github(self):
        self.check("github", GHPAT)
        self.check("github", GHP)
        self.check("github", "gho_" + rnd(36))

    def test_slack(self):
        self.check("slack", SLACK)

    def test_aws(self):
        self.check("aws", AKIA)

    def test_tg(self):
        self.check("tg", TG)

    def test_bark_host(self):
        for url in (f"https://bark.example.com/{BARK}/标题/内容", f"https://api.day.app/{BARK}", f"http://bark.x.cc/{BARK}"):
            out, c = R.redact("推送走 " + url + " 就行")
            self.assertNotIn(BARK, out, url)
            self.assertEqual(c.get("bark"), 1, url)
            self.assertIn(url.split(BARK)[0], out)                   # 主机名保留

    def test_bark_path_segment(self):
        out, c = R.redact(f"调用 https://push.example.com/bark-server/{BARK}/hi 试试")
        self.assertNotIn(BARK, out)
        self.assertEqual(c.get("bark"), 1)

    def test_bearer(self):
        out, c = R.redact(f"Authorization: Bearer {BEARER}")
        self.assertNotIn(BEARER, out)
        self.assertIn("Bearer 〔已打码:bearer:", out)
        self.assertEqual(c, {"bearer": 1})

    def test_entropy_43_and_32(self):
        self.check("entropy", LONG43, "令牌在这儿：{}。你转给它")
        self.check("entropy", rnd(32), "绑定码 {} 10 分钟有效".replace("绑定码 ", "给你 "))

    def test_kv_hex_key(self):
        hexkey = "".join(_r.choice("0123456789abcdef") for _ in range(32))
        out, c = R.redact(f"api_key={hexkey}")
        self.assertNotIn(hexkey, out)
        self.assertEqual(c.get("kv"), 1)

    def test_kv_chinese(self):
        k = rnd(24)
        out, c = R.redact(f"密钥是 {k}，别外传")
        self.assertNotIn(k, out)

    def test_in_json_and_code_context(self):
        out, c = R.redact('{"api_key": "%s", "x": 1}' % SK)
        self.assertNotIn(SK, out)
        out, c = R.redact(f"export DEEPSEEK_API_KEY={SK}\nexport OTHER=1")
        self.assertNotIn(SK, out)
        self.assertEqual(sum(c.values()), 1, c)                      # 同一把只计一次

    def test_multiple_and_counts(self):
        out, c = R.redact(f"{SK} 还有 {GHP} 还有 {AKIA}")
        self.assertEqual(c, {"sk": 1, "github": 1, "aws": 1})
        for s in (SK, GHP, AKIA):
            self.assertNotIn(s, out)

    def test_idempotent(self):
        once, _ = R.redact(f"key {SK} and {TG} and https://bark.a.b/{BARK}")
        twice, c2 = R.redact(once)
        self.assertEqual(once, twice)
        self.assertEqual(c2, {})


class FalsePositives(unittest.TestCase):
    def same(self, s):
        out, c = R.redact(s)
        self.assertEqual(out, s, c)
        self.assertEqual(c, {})

    def test_hashes(self):
        h = "a" * 8 + "0123456789abcdef" * 3 + "ffffffff"
        self.same(f"content_sha256={h}")
        self.same(f"commit 4f82d94c5a7aa3d3a4d07e335337 fine")
        self.same("sha256:" + "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
        self.same("md5 d41d8cd98f00b204e9800998ecf8427e")
        self.same("/srv/archive/9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08/x.jsonl")

    def test_uuid_and_ids(self):
        self.same("session 3f2b8c1e-5a4d-4e7b-9c3a-1d2e3f4a5b6c 结束")
        self.same("fact_cdd39d3676a0 / ev_0123456789ab")

    def test_paths_urls(self):
        self.same("/home/tester/workspace/claude/memory-digest/.env 不能读")
        self.same("见 https://github.com/Finb/Bark 和 https://github.com/Finb/Bark/releases")
        self.same("https://example.com/a/b/c?x=1&y=2#top 打开看看")
        self.same("engram-pre-A4-20261027-193251.db 在 backup 里")
        self.same("archive/B20261028-GA1-fix.jsonl 和 B20261028-US2-review.json")

    def test_identifiers(self):
        self.same("test_known_defects_search_returns_fact_type_for_every_hit_row")
        self.same("distill_slice_dead_letter_retry_after_backoff_minutes_10_20_40")
        self.same("some-very-long-kebab-case-branch-name-for-the-ingest-slim-work-2026")
        self.same("def test_一句三事应当抽出至少三条候选(self): pass")

    def test_public_key_fileid_filename(self):
        pub = "AAAAC3NzaC1lZDI1NTE5AAAAI" + rnd(30)
        self.same(f"公钥 ssh-ed25519 {pub} grok@box")
        self.same('attachment_file_id="' + rnd(70, B62 + "_-") + '" 一张图')
        self.same("<channel>/home/u/.claude/channels/telegram/inbox/" + rnd(32, B62 + "-") + '.jpg">这啥啊')

    def test_url_ticket_still_redacted(self):
        t = rnd(43)
        out, c = R.redact(f"https://account.example.com/pass/qr/login?ticket={t}&dc=sgp")
        self.assertNotIn(t, out)
        self.assertEqual(c, {"entropy": 1})

    def test_words_numbers(self):
        self.same("2026-10-28T02:14:43+00:00 的 llm_usage 里 prompt_tokens=6423")
        self.same("她说今天喝了奶茶，还要去某大学办事情，手机号 13800138000 是假的")
        self.same("A" * 40)
        self.same("1234567890" * 4)
        self.same("token 数量不够，再试一次；密码太简单了")
        self.same("Bearer token 是旧的，作废了")
        self.same("sk-learn 和 disk-usage-monitoring-tool-config-v2 都不是密钥")
        self.same("看 task-management-system 的 skeleton-ui-component-library")

    def test_short_and_nonstr(self):
        self.assertEqual(R.redact(None), (None, {}))
        self.assertEqual(R.redact("短"), ("短", {}))
        self.assertEqual(R.redact(""), ("", {}))


class LogAndApi(unittest.TestCase):
    def test_log_has_counts_not_values(self):
        old = R.LOG_FILE
        with tempfile.TemporaryDirectory() as d:
            R.LOG_FILE = Path(d) / "redact.log"
            try:
                out = R.scrub(f"key {SK} 和 {GHP}", "unit")
                self.assertNotIn(SK, out)
                log = R.LOG_FILE.read_text(encoding="utf-8")
                self.assertIn("unit github=1 sk=1", log)
                self.assertNotIn(SK, log)
                self.assertNotIn(SK[3:12], log)
                R.scrub("没有密钥的话", "unit")
                self.assertEqual(len(R.LOG_FILE.read_text(encoding="utf-8").splitlines()), 1)
            finally:
                R.LOG_FILE = old

    def test_log_failure_is_silent(self):
        old = R.LOG_FILE
        R.LOG_FILE = Path("/proc/nonexistent/x.log")
        try:
            self.assertNotIn(SK, R.scrub("k " + SK))
        finally:
            R.LOG_FILE = old

    def test_no_off_switch(self):
        os.environ["ENGRAM_REDACT"] = "0"
        os.environ["REDACT"] = "off"
        self.assertNotIn(SK, R.redact("k " + SK)[0])
        self.assertFalse(hasattr(R, "enabled"))

    def test_speed(self):
        import time
        text = ("她今天说了很多话，" * 40 + SK + " ") * 20
        t = time.perf_counter()
        for _ in range(20):
            R.redact(text)
        self.assertLess((time.perf_counter() - t) / 20, 0.05)       # 一次 ~7KB 文本 < 50ms


if __name__ == "__main__":
    unittest.main()
