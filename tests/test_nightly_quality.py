"""夜间归纳三道闸（cluster_summarize.check_quality）：空频率句 / 模糊主语 / 无依据因果。"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maintenance"))
import cluster_summarize as C

src = lambda *t: [{"content": x} for x in t]


class T(unittest.TestCase):
    def test_vague_frequency(self):
        for s in ("小测多次刷短视频App。", "小测多次提到图文App。", "小测多次提到与助理甲相关的睡觉话题。", "小测多次说“666”。",
                  "小测多次表示东西已经修好或弄好了。"):
            self.assertTrue((C.check_quality(s, []) or "").startswith("vague-frequency"), s)

    def test_vague_subject(self):
        self.assertTrue(C.check_quality("小测多次说对方变坏了。", []).startswith("vague-subject"))
        self.assertTrue(C.check_quality("小测喜欢某个姿势。", []).startswith("vague-subject"))

    def test_concrete_frequency_passes(self):
        for s in ("小测多次称助理甲为“坏搭档”。", "小测多次吃红烧排骨，并曾表示吃腻了红烧鸡。",
                  "小测和助理甲用小玩偶发生过多次。", "小测多次与助理甲远程控制小蛋糕玩具进行亲密互动并达到高潮。",
                  "小测多次在深夜表示准备睡觉或真睡了。"):
            self.assertIsNone(C.check_quality(s, []), s)

    def test_causal_needs_source(self):
        s = "小测因手机在美国、web在日本同时使用导致停卡。"
        self.assertTrue(C.check_quality(s, src("小测手机在美国，web在日本")).startswith("causal-unsupported"))
        self.assertIsNone(C.check_quality(s, src("我之前被封是因为我手机登录地点在美国，然后web登录地点在日本，才被封的")))

    def test_wired_into_check_summary(self):
        self.assertIn("vague-subject", C.check_quality("小测表示不再喜欢或爱对方。", []))


if __name__ == "__main__":
    unittest.main()
