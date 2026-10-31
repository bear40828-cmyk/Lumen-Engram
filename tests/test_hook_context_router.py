"""2026-10-13 记忆库真实错召回的纯规则回归测试。"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "engram_hook", Path(os.environ.get("ENGRAM_HOOK_PATH") or ROOT / "hooks_engram_recall.py"))
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


class HookContextRouter(unittest.TestCase):
    def test_relay_joke_is_rejected_before_search(self):
        text = "小乙：以后你俩调情的时候我还要在旁边起哄呗？"
        self.assertEqual(hook.pre_recall_skip(text), "转述／玩笑，不翻旧账")

    def test_pizza_question_keeps_its_own_query(self):
        text = "可以点个披萨吗"
        query, used = hook.contextual_query({"prompt": text}, text)
        self.assertEqual(query, text)
        self.assertFalse(used)
        self.assertIsNone(hook.pre_recall_skip(text))

    def test_image_plus_done_uses_recent_dialogue(self):
        entries = [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text":
                    '<channel source="plugin:telegram:telegram">把测试工作台下面的测试信箱做好</channel>'}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text":
                    "工作台的下拉和测试信箱入口已经接上了。"}]},
            },
        ]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as fh:
            for row in entries:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            data = {
                "prompt": '<channel image_path="/tmp/workbench.jpg">做好了</channel>',
                "transcript_path": fh.name,
            }
            query, used = hook.contextual_query(data, "做好了")
        self.assertTrue(used)
        self.assertIn("测试信箱", query)
        self.assertTrue(query.endswith("做好了"))

    def test_camera_accessory_does_not_become_companion_robot(self):
        query = "给机器人装个小麦克风，当成你的耳朵，做成桌面摆设"
        wrong = "7/30 小测提议买陪伴机器人，带麦克风，后来改成手链挂饰"
        right = "小测想做一个带麦克风的桌面摆设，给助理甲当耳朵"
        self.assertFalse(hook.entity_compatible(query, wrong))
        self.assertTrue(hook.entity_compatible(query, right))

    def test_named_companion_robot_is_not_blocked(self):
        query = "陪伴机器人里面那个麦克风后来怎么说的"
        candidate = "7/30 小测提议买陪伴机器人，带麦克风"
        self.assertTrue(hook.entity_compatible(query, candidate))

    def test_choice_uses_directly_related_preference(self):
        query = "可以点个披萨吗"
        candidate = "小测吃披萨要配可乐，披萨要加芝士。"
        self.assertTrue(hook.deterministic_relevance(query, candidate))

    def test_done_reply_uses_project_entity_from_context(self):
        query = "把测试工作台下面的测试信箱做好\n工作台入口已经接好了\n做好了"
        candidate = "测试信箱是工作台底部的长期记忆入口。"
        self.assertTrue(hook.deterministic_relevance(query, candidate))


if __name__ == "__main__":
    unittest.main()
