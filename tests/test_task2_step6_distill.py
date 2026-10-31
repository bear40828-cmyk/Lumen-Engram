"""任务二·步骤6：蒸馏「我：」原话（选样 / quote 核对 / 入库收起；模型调用全 mock）"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import distill_self as D  # noqa: E402
import llm  # noqa: E402
from store.store import add_evidence, confirm, connect, promote  # noqa: E402

LONG = "我把检索脚本改成了混合召回，还把备用模型的配置写进了配置文件，下一步要去跑一遍测试，看看报错日志里有没有新的问题，" \
       "然后再决定要不要把旧的分层路径删掉，现在的结论是先留着开关，等她点头。"


def mk(con, body, when="2026-10-01 10:00", **kw):
    ev = add_evidence(con, source_kind="chat", source_actor="claude", source_path="/tmp/x.jsonl",
                      source_locator=f"L{abs(hash(body)) % 10**6}", content=body,
                      verification_status="verified")
    return promote(con, subject="claude", fact_type="episodic", content=f"我（{when}）：{body}",
                   evidence_ids=[ev], valid_from="2026-10-01T02:00:00+00:00",
                   importance=0.5, confidence=0.7, **kw), ev


class Selection(unittest.TestCase):
    def test_category(self):
        self.assertIsNone(D.category("小测（2026-10-01 10:00）：" + LONG))          # 不是「我：」
        self.assertIsNone(D.category("我：（笑）"))                                  # K2
        self.assertIsNone(D.category("我：好的，收到"))                               # K3
        self.assertIsNone(D.category("我：<channel source=x>" + LONG))              # K0
        self.assertIsNone(D.category("我：哈哈哈今天好开心呀你真的好可爱呀😂😂😂😂😂" + "呀" * 10))  # K6
        self.assertEqual(D.category("我：" + LONG), "K7")                            # 含「要不要/下一步」
        self.assertEqual(D.category("我：" + "我把数据库的配置文件和部署脚本都改完了，服务器上的日志也清了一遍。" * 5), "K8")
        self.assertEqual(D.category("我：" + "昨天我们一起看了很久的电影，后来她睡着了，我就把灯关了。" * 2), "K9")

    def test_pick_is_deterministic_and_stratified(self):
        pool = [{"id": i, "cat": ("K7", "K8", "K9", "K9")[i % 4]} for i in range(400)]
        a, b = D.pick(pool, 50), D.pick(list(reversed(pool)), 50)
        self.assertEqual([p["id"] for p in a], [p["id"] for p in b])
        self.assertEqual(len(a), 50)
        self.assertEqual(sum(1 for p in a if p["cat"] == "K9") > sum(1 for p in a if p["cat"] == "K7"), True)

    def test_mask_hides_content(self):
        m = D.mask("我今天把Telegram的密码 12345 发给了小明")
        self.assertNotIn("Telegram", m)
        self.assertNotIn("12345", m)
        self.assertNotIn("小明", m)

    def test_plan_numbers(self):
        p = D.plan_numbers([{"content": "我：" + "字" * 100}] * 10)
        self.assertEqual(p["items"], 10)
        self.assertGreater(p["input_tokens"], 10 * 100 * 0.7)


class OnStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)

    def chat(self, payload):
        return mock.patch.object(llm, "chat_ex", return_value=llm.LLMResult(
            json.dumps(payload, ensure_ascii=False), "primary", "m", {"prompt_tokens": 500, "completion_tokens": 60}, ["primary:ok"]))

    def test_pool_excludes_archived_pinned_confirmed_and_llm(self):
        a, _ = mk(self.con, LONG)
        b, _ = mk(self.con, LONG + "（另一条）", user_pinned=True)
        c, _ = mk(self.con, LONG + "（第三条）")
        confirm(self.con, self.con.execute("SELECT ext_id FROM facts WHERE id=?", (c,)).fetchone()[0])
        self.con.commit()
        ids = {p["id"] for p in D.load_pool(self.con)}
        self.assertEqual(ids, {a})

    def test_distill_one_verbatim_quote_check(self):
        fid, _ = mk(self.con, LONG)
        row = D.load_pool(self.con)[0]
        good = {"subject": "claude", "type": "procedural", "content": "助理甲把旧分层路径留在开关后面，等她点头再决定删不删",
                "quote": "先留着开关，等她点头", "confidence": 0.9}
        bad = dict(good, quote="原话里根本没有这句话")
        with self.chat({"facts": [good, bad]}):
            r = D.distill_one(row)
        self.assertEqual(len(r["kept"]), 1)
        self.assertEqual(len(r["dropped"]), 1)
        self.assertEqual(r["usage"]["prompt_tokens"], 500)

    def test_apply_inserts_candidates_and_archives_sources(self):
        fid, ev = mk(self.con, LONG)
        keep, _ = mk(self.con, "我：别的话" + "。" * 3 + LONG[:60] * 2 + "要不要下一步")
        pinned, _ = mk(self.con, LONG + "（钉）", user_pinned=True)
        rows = D.load_pool(self.con)
        self.assertEqual({r["id"] for r in rows}, {fid, keep})
        good = {"subject": "claude", "type": "procedural", "content": "助理甲把旧分层路径留在开关后面",
                "quote": "先留着开关，等她点头", "confidence": 0.9}
        # 用真实 check 的结果结构
        src = next(r for r in rows if r["id"] == fid)
        ok, why = D.check(dict(good, src=0), [{"id": ev, "content": src["content"], "occurred_at": "2026-10-01T02:00:00+00:00"}])
        self.assertIsNotNone(ok, why)
        ok.pop("hit")
        results = [{"fact_id": fid, "cat": "K7", "route": "primary", "model": "m", "usage": {}, "dropped": [], "kept": [ok]}]
        before_conf = self.con.execute("SELECT COUNT(*) FROM facts WHERE status='confirmed'").fetchone()[0]
        stat = D.apply(self.con, rows, results, "BTEST-D1", Path(self.dir.name))
        self.assertEqual((stat["new_facts"], stat["archived"]), (1, 1))
        st = dict(self.con.execute("SELECT id, status FROM facts WHERE id IN (?,?,?)", (fid, keep, pinned)).fetchall())
        self.assertEqual(st, {fid: "uncertain", keep: "candidate", pinned: "candidate"})
        new = self.con.execute("""SELECT f.status, c.reason FROM facts f JOIN fact_class c ON c.fact_id=f.id
                                  WHERE c.reason=?""", (D.EXTRACTOR,)).fetchall()
        self.assertEqual([tuple(x) for x in new], [("candidate", D.EXTRACTOR)])
        arch = self.con.execute("SELECT fact_id, batch_id, action, prev_status FROM facts_archive").fetchall()
        self.assertEqual([tuple(x) for x in arch], [(fid, "BTEST-D1", "archived", "candidate")])
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE status='confirmed'").fetchone()[0], before_conf)
        # 原文导出了
        line = (Path(self.dir.name) / "BTEST-D1-distill.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(line[0])["fact"]["id"], fid)
        # 不物理删除
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts WHERE id=?", (fid,)).fetchone()[0], 1)

    def test_plan_mode_never_calls_model_or_writes(self):
        mk(self.con, LONG)
        before = self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        with mock.patch.object(D, "connect", return_value=self.con), \
                mock.patch.object(llm, "chat_ex", side_effect=AssertionError("plan 不该调模型")), \
                mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True), \
                mock.patch.object(sys, "argv", ["distill_self.py", "--plan"]):
            self.assertEqual(D.main(), 0)
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], before)

    def test_run_without_key_exits_2(self):
        mk(self.con, LONG)
        with mock.patch.object(D, "connect", return_value=self.con), \
                mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True), \
                mock.patch.object(sys, "argv", ["distill_self.py", "--run"]):
            self.assertEqual(D.main(), 2)



def setUpModule():
    # 这个文件测的是旧管线（值得记闸关）的行为：不管环境变量 ENGRAM_VALUE_GATE 开没开，都按闸关跑（闸开的行为在 test_value_gate*.py 里测）
    import os
    from unittest import mock
    p = mock.patch.dict(os.environ, {"ENGRAM_VALUE_GATE": "0", "ENGRAM_VALUE_GATE_UNION": "0"})
    p.start()
    unittest.addModuleCleanup(p.stop)


if __name__ == "__main__":
    unittest.main()
