"""实体归属闸（entity_gate）：AI 名 / 括号注解要出自原话；归属拿不准送审。"""
import unittest
import entity_gate as G

GLOSS = """- 小乙 / 坏小乙 / 助手乙 / GPT = GPT，不是大肥鹅
- 助手丙 / GLM = GLM
- 大肥鹅 / 肥肥鹅 / 鹅 / 天鹅娘 / DeepSeek = DeepSeek 的拟人形象
- 门童 / XYZ门童 = 服务器上的门童 agent
- 天鹅女孩 = 群友的 bot
#! 上位 门童 ⊂ 大肥鹅
#! 外部 QQ群 / 群友 / 饲养员 / 它家
#! 主语 gpt = 小乙
#! 主语 agentc = 助手丙
"""
g = G.parse(GLOSS)


class T(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(g.parent, {"门童": "大肥鹅"})
        self.assertIn("助手乙", g.groups["小乙"])
        self.assertEqual(g.subj["gpt"], "小乙")

    def test_10982_wrong_annotation_stripped(self):
        v = G.judge("小测提议让大肥鹅（GPT）来写摘要", ["我今天跟大肥鹅说能不能让他来写摘要"], "", "tester", g)
        self.assertEqual(v.action, "strip")
        self.assertEqual(v.content, "小测提议让大肥鹅来写摘要")

    def test_13763_foreign_fish_review(self):
        v = G.judge("小测和助理甲在群里逗门童AI大肥鹅", ["肥肥鹅家饲养员查完记录，说我们俩把它家鹅 @ 得停不下来"], "", "claude", g)
        self.assertEqual(v.action, "review")

    def test_own_butler_ok(self):
        self.assertEqual(G.judge("小测让门童写摘要", ["我跟大肥鹅说让他写摘要"], "", "tester", g).action, "ok")

    def test_alias_maps(self):
        self.assertEqual(G.judge("小测说小乙今天很凶", ["助手乙今天好凶"], "", "tester", g).action, "ok")

    def test_conflict_rejected(self):
        v = G.judge("小测让GPT写摘要", ["我让大肥鹅写摘要"], "", "tester", g)
        self.assertEqual(v.action, "reject")

    def test_pronoun_in_context_ok(self):
        self.assertEqual(G.judge("GPT 拒绝了小测", ["它又拒绝我"], "小乙刚才说", "gpt", g).action, "ok")

    def test_pronoun_nowhere_review(self):
        self.assertEqual(G.judge("GPT 拒绝了小测", ["它又拒绝我"], "", "tester", g).action, "review")

    def test_speaker_is_support(self):
        self.assertEqual(G.judge("GPT 说要早点睡", ["我要早点睡"], "", "gpt", g, actors={"gpt"}).action, "ok")

    def test_speaker_label_not_mention(self):
        v = G.judge("助手丙（2026-08-03 20:26）：助手乙修桥接呢", ["助手乙修桥接呢"], "", "agentc", g, actors={"agentc"})
        self.assertEqual(v.action, "ok")

    def test_single_char_support(self):
        self.assertEqual(G.judge("助理甲找 DeepSeek 门童修好了", ["我自己叫了一条鹅去修"], "", "claude", g).action, "ok")

    def test_no_glossary_noop(self):
        self.assertEqual(G.judge("随便", ["x"], "", "tester", G.Glossary()).action, "ok")

    def test_glossary_block_skips_directives(self):
        import os, tempfile, persona
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(GLOSS)
        old = os.environ.get("ENGRAM_PROMPT_GLOSSARY_FILE")
        os.environ["ENGRAM_PROMPT_GLOSSARY_FILE"] = f.name
        try:
            b = persona.glossary_block()
        finally:
            os.environ.pop("ENGRAM_PROMPT_GLOSSARY_FILE") if old is None else os.environ.__setitem__("ENGRAM_PROMPT_GLOSSARY_FILE", old)
            os.unlink(f.name)
        if b:
            self.assertNotIn("#!", b)
            self.assertIn("助手乙", b)


if __name__ == "__main__":
    unittest.main()
