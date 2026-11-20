import tests  # noqa: F401
import unittest
import hooks_engram_recall as H

GPU = "助理甲（2026-09-01）：确认显卡是RTX 4060 Laptop，显存8G"


class OwnerCheck(unittest.TestCase):
    def test_other_owner_blocks(self):
        for q in ("不知道Grok bot的电脑有没有显卡，有的话在他电脑上跑", "Grok's GPU 够不够", "你的显卡是什么", "他的电脑能跑吗"):
            self.assertFalse(H.owner_compatible(q, GPU), q)

    def test_self_owner_passes(self):
        for q in ("我的好像是4060的，你去翻一下聊天记录", "我的显卡是什么来着", "显卡显存多少"):
            self.assertTrue(H.owner_compatible(q, GPU), q)

    def test_candidate_names_owner_passes(self):
        self.assertTrue(H.owner_compatible("Grok bot的电脑有没有显卡", "Grok bot 的机器没有显卡，只有 CPU"))

    def test_unrelated_noun_passes(self):
        self.assertTrue(H.owner_compatible("你的意思是要吃饭吗", GPU))
        self.assertTrue(H.owner_compatible("你的电脑", "她喜欢吃椒盐排骨"))


if __name__ == "__main__":
    unittest.main()
