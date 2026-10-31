"""2026-10-29：新事顶旧事 v4（关系分类 + 代码卡类型 + 可选第二遍复核）。虚构夹具，模型调用全部 mock。"""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract import timeline  # noqa: E402

ON = {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_TIMELINE": "1"}
OLDS = [{"content": "小林在用旧手机", "valid_from": "2026-09-01"},
        {"content": "小林周三去了牙科", "valid_from": "2026-09-03"},
        {"content": "小林打算周六去爬山", "valid_from": "2026-09-05"}]


def reply(*items, kind="state"):
    return json.dumps({"new_kind": kind, "items": list(items)}, ensure_ascii=False)


def item(i, rel, old_kind="state", q=""):
    return {"id": i, "old_kind": old_kind, "relation": rel, "question": q}


class SupersedeV4(unittest.TestCase):
    def judge(self, *texts, env=None):
        with mock.patch.dict(os.environ, {**ON, **(env or {})}, clear=True), \
                mock.patch.object(timeline, "chat", side_effect=list(texts)) as m:
            r = timeline.judge("小林换了新手机", "2026-09-10", OLDS)
        return r, m

    def test_updates_state_marks(self):
        (idx, why), _ = self.judge(reply(item(0, "updates", "state", "小林用什么手机")))
        self.assertEqual(idx, 0)
        self.assertTrue(why.startswith("小林用什么手机｜"))

    def test_enriches_duplicate_different_never_mark(self):
        for rel in ("enriches", "duplicate", "different"):
            (idx, _), _ = self.judge(reply(item(0, rel, "state", "x")))
            self.assertIsNone(idx, rel)

    def test_past_event_is_never_superseded(self):
        (idx, _), _ = self.judge(reply(item(1, "updates", "event", "小林哪天去牙科")))
        self.assertIsNone(idx)

    def test_plan_can_be_superseded_and_needs_question(self):
        (idx, _), _ = self.judge(reply(item(2, "updates", "plan", "周六去不去爬山")))
        self.assertEqual(idx, 2)
        (idx, _), _ = self.judge(reply(item(2, "updates", "plan", "")))
        self.assertIsNone(idx)

    def test_latest_of_several_wins(self):
        (idx, _), _ = self.judge(reply(item(0, "updates", "state", "q1"), item(2, "updates", "plan", "q2")))
        self.assertEqual(idx, 2)

    def test_bad_output_and_out_of_range_are_not_marks(self):
        for t in ("not json", reply(item(9, "updates", "state", "q")), reply({"id": "x", "relation": "updates"})):
            (idx, _), _ = self.judge(t)
            self.assertIsNone(idx)

    def test_confirm_pass_must_agree(self):
        env = {"ENGRAM_TIMELINE_CONFIRM": "1"}
        first = reply(item(0, "updates", "state", "小林用什么手机"))
        (idx, _), m = self.judge(first, reply(item(0, "enriches")), env=env)
        self.assertIsNone(idx)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(m.call_args.kwargs["purpose"], "timeline_confirm")
        (idx, _), _ = self.judge(first, reply(item(0, "updates", "state", "小林用什么手机")), env=env)
        self.assertEqual(idx, 0)

    def test_confirm_off_makes_one_call(self):
        _, m = self.judge(reply(item(0, "updates", "state", "q")))
        self.assertEqual(m.call_count, 1)

    def test_confirm_only_sees_the_flagged_old(self):
        env = {"ENGRAM_TIMELINE_CONFIRM": "1"}
        _, m = self.judge(reply(item(2, "updates", "plan", "q")), reply(item(0, "updates", "plan", "q")), env=env)
        prompt = m.call_args.args[0][1]["content"]
        self.assertIn("[0]（2026-09-05）小林打算周六去爬山", prompt)
        self.assertNotIn("[1]（", prompt)


class ConfirmFlag(unittest.TestCase):
    def test_registered_default_off_and_env_on(self):
        import flags
        self.assertIn("TIMELINE_CONFIRM", flags.DEFAULTS)
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent"}, clear=True):
            self.assertFalse(flags.on("TIMELINE_CONFIRM"))
            self.assertFalse(timeline._confirm_on())
        with mock.patch.dict(os.environ, {"ENGRAM_ENV_FILE": "/nonexistent", "ENGRAM_TIMELINE_CONFIRM": "1"}, clear=True):
            self.assertTrue(timeline._confirm_on())


class RoutePurpose(unittest.TestCase):
    def test_confirm_model_route(self):
        import llm
        env = {"DIGEST_API_KEY": "k", "DIGEST_BASE_URL": "http://x/v1", "EXTRACT_MODEL": "small",
               "TIMELINE_CONFIRM_MODEL": "big"}
        self.assertEqual(llm.routes_for("timeline_confirm", env)[0].model, "big")
        env.pop("TIMELINE_CONFIRM_MODEL")
        self.assertEqual(llm.routes_for("timeline_confirm", env)[0].model, "small")


if __name__ == "__main__":
    unittest.main()
