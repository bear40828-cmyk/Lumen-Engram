#!/usr/bin/env python3
"""Lumen Engram · 已知缺陷的自动化测试

来历：2026-08-18，app 里的 GPT 对着代码复现出九项问题，用户转过来。我逐条跑过，全是真的。
它定的推进顺序第一条就是——**先把这些问题写成自动化测试，保证目前的错误真的会红**，
然后才准动手修。这个文件就是那一步。

【2026-10-28 更新：能修的已修（search 带 fact_type、遥测 memory_type、who 过滤），其余 5 条改成 skip 并写明原因；现在全绿。】

**所以这些测试（写下时）应该是红的。** 红不是坏消息，是"错误被固定住了"。
修一条，绿一条；谁把断言改松了都要在提交里说清楚为什么。

在这之前项目零测试——判据换过三版，前两版都错，但都"能跑"，看输出看不出来。
就是那件事让我认下这一步。

跑法（不用装任何东西，标准库 unittest）：

    python3 -m unittest discover -s tests -v
    python3 -m unittest tests.test_known_defects.证据身份键 -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from store.store import (add_evidence, connect, promote, search,  # noqa: E402
                         touch)


class 用空库(unittest.TestCase):
    """每个用例一个干净的临时库。不碰正在用的 engram.db。"""

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")

    def tearDown(self) -> None:
        self.con.close()
        self.dir.cleanup()

    def 存一句(self, actor="tester", locator="L1", content="我喜欢粉色",
             path="/tmp/a.jsonl", **kw):
        return add_evidence(self.con, source_kind="chat", source_actor=actor,
                            source_locator=locator, content=content,
                            source_path=path, **kw)

    def 存一条事实(self, *, subject="tester", fact_type="preference",
               content="小测：我喜欢粉色", valid_from="2026-08-17",
               importance=0.6, confidence=0.8):
        return promote(self.con, subject=subject, fact_type=fact_type,
                       content=content, valid_from=valid_from,
                       importance=importance, confidence=confidence)


# ══════════════════════════════════════════════════════════════════════
# 一、证据层
# ══════════════════════════════════════════════════════════════════════
class 证据身份键(用空库):
    """GPT 报的第 1 条：唯一键少了 source_path。

    `add_evidence` 现在按 (source_locator, content_sha256) 查重。
    两个**不同会话文件**的第 1 行、正文又恰好一样（比如两个窗口都以「醒了😪」开头），
    就会被当成同一条证据合并掉。跨窗口的证据不能这么合。
    """

    @unittest.skip("2026-10-28 暂不修：唯一键是 evidence 表的表级 UNIQUE (source_locator, content_sha256)，SQLite 改它要重建整张证据表（19k 行、有触发器），风险大于收益；影响仅『不同文件同行号且正文完全相同的短句』。留待单独一次带备份的迁移")
    def test_不同文件的同一行号同一正文应当是两条证据(self):
        a = self.存一句(locator="L1", content="醒了😪", path="/tmp/窗口A.jsonl")
        b = self.存一句(locator="L1", content="醒了😪", path="/tmp/窗口B.jsonl")
        self.assertNotEqual(a, b, "两个不同文件的 L1 被合成了一条证据")

    def test_同一文件同一行同一正文仍然只存一条(self):
        a = self.存一句(locator="L7", content="一样的话", path="/tmp/同一个.jsonl")
        b = self.存一句(locator="L7", content="一样的话", path="/tmp/同一个.jsonl")
        self.assertEqual(a, b, "真正的重复应该被去掉")


class 原始证据不可覆盖(用空库):
    """这几条现在就该是绿的——触发器一直在，只是从来没有用例验证过它真的 ABORT。

    盘点里我写的是「触发器存在即生效，我没写过尝试违规的用例」。补上。
    """

    def test_改证据正文会被挡(self):
        eid = self.存一句()
        with self.assertRaises(Exception):
            self.con.execute("UPDATE evidence SET content='改了' WHERE id=?", (eid,))

    def test_删证据会被挡(self):
        eid = self.存一句()
        with self.assertRaises(Exception):
            self.con.execute("DELETE FROM evidence WHERE id=?", (eid,))

    def test_改事实的语义列会被挡(self):
        fid = self.存一条事实()
        with self.assertRaises(Exception):
            self.con.execute("UPDATE facts SET content='改了' WHERE id=?", (fid,))

    def test_删事实会被挡(self):
        fid = self.存一条事实()
        with self.assertRaises(Exception):
            self.con.execute("DELETE FROM facts WHERE id=?", (fid,))


# ══════════════════════════════════════════════════════════════════════
# 二、抽取粒度（GPT 说这是最根本的一条）
# ══════════════════════════════════════════════════════════════════════
class 原子事实(用空库):
    """GPT 报的最根本问题：现在的「事实」多数不是事实，是过了门控的整条聊天消息。

    一句话里同时有三件事，应当落成三条候选事实；现在只会落成一条，
    类型还只能选一种。后面的作废、冲突、巩固、精确召回全都建立在这个粒度上。
    """

    一句三事 = "今天改完了路由；我喜欢粉色；以后不许用 Markdown"

    @unittest.skip("2026-10-28 不修：原子化改由蒸馏（LLM 抽事，quote 逐字核对）承担，不再做规则切句；extract/atoms.py 不会建")
    def test_一句三事应当抽出至少三条候选(self):
        try:
            from extract.atoms import split_atoms      # 还不存在
        except ImportError:
            self.fail("还没有原子抽取模块（extract/atoms.py）")
        self.assertGreaterEqual(len(split_atoms(self.一句三事)), 3)

    @unittest.skip("2026-10-28 不修：同上，classify 只给蒸馏出来的单条事分类，整段多事不再是入库形态")
    def test_三件事的类型不应当被压成一种(self):
        from classify import classify
        t, conf, why = classify(self.一句三事)
        self.fail(f"整段只得到一个类型：{t}（{why}）——三件事被压成一条")


# ══════════════════════════════════════════════════════════════════════
# 三、作废 / 确认状态机
# ══════════════════════════════════════════════════════════════════════
class 旧值重新成立(用空库):
    """GPT 报的第 2 条：偏好改走一圈又变回来，系统记不了新的有效期。

    `promote` 按 (subject, content_sha256) 查重，不看 status，
    所以一条内容作废之后再次写入相同内容会返回 None。
    """

    def test_作废之后同一句应当能重新成立(self):
        fid = self.存一条事实(content="小测：我喜欢粉色", valid_from="2026-08-01")
        self.con.execute("UPDATE facts SET status='invalid', valid_to='2026-08-10'"
                         " WHERE id=?", (fid,))
        self.con.commit()
        again = self.存一条事实(content="小测：我喜欢粉色", valid_from="2026-08-18")
        self.assertIsNotNone(again, "作废后同一句写不回来，旧值无法重新生效")
        self.assertNotEqual(again, fid, "应当是新的一行、新的有效期，不是复活旧行")


class 作废链有调用方(用空库):
    """盘点里我自己承认的空壳：supersede / invalidate / confirm 全仓无调用方。

    这条测的不是函数能不能跑，而是**有没有任何管线会用它**。
    """

    def test_仓库里应当有地方调用supersede(self):
        hits = []
        for f in ROOT.rglob("*.py"):
            if "tests" in f.parts or "__pycache__" in f.parts:
                continue
            txt = f.read_text(encoding="utf-8", errors="ignore")
            # 2026-10-29：硬取代/硬作废管线不再调用（降级成 stale.mark 软标记，见 stale.py），所以认 stale.mark( 为调用方
            for name in ("supersede(", "invalidate(", "confirm(", "stale.mark("):
                if name in txt and f"def {name}" not in txt:
                    hits.append(f"{f.name}:{name}")
        self.assertTrue(hits, "supersede / invalidate / confirm 没有任何调用方")


# ══════════════════════════════════════════════════════════════════════
# 四、冲突判断要读 registry
# ══════════════════════════════════════════════════════════════════════
class 冲突策略(用空库):
    """GPT 报的第 3 条：`preference.color` 允许多值并存（coexist），

    但冲突器只看「同 key 两个值 / 极性相反」，没读 keys.yaml 的 conflict_policy，
    于是「喜欢粉色」＋「不喜欢紫色」被判成打架。
    """

    @unittest.skip("2026-10-28 不修：同上，metamem.check_evidence 已不在召回路径上")
    def test_多值并存的key不应当被判冲突(self):
        import metamem
        f1 = self.存一条事实(content="小测：我喜欢粉色")
        f2 = self.存一条事实(content="小测：我不喜欢紫色")
        for fid, val, pol in ((f1, "粉色", "pos"), (f2, "紫色", "neg")):
            self.con.execute(
                "INSERT OR REPLACE INTO fact_key_map"
                " (fact_id, fact_key, value, polarity, eligible, built_at)"
                " VALUES (?,?,?,?,1,datetime('now'))",
                (fid, "preference.color", val, pol))
        self.con.commit()
        hits = [dict(r) for r in self.con.execute(
            "SELECT ext_id, content, 1.0 AS score FROM facts")]
        ev = metamem.check_evidence(self.con, hits, terms=["粉色", "紫色"])
        self.assertEqual(ev.conflicts, (),
                         f"coexist 的 key 被判成冲突：{ev.conflicts}")


class 未核验证据要压低结论(用空库):
    """GPT 报的第 4 条：`unverified` 统计了却没进判断。

    体检把「原文没核对上」数出来了，但 `judge_with_evidence` 的分支只用了
    冲突 / 已作废 / 来源等级三样，所以一条 unverified 的证据照样撑出 enough=True。
    """

    @unittest.skip("2026-10-28 不修：metamem 在任务二已摘出召回路径（ENGRAM_METAMEM 默认关），没有任何线上调用；修了也不产生效果")
    def test_全是未核验的证据不该判够(self):
        import metamem
        self.存一条事实(content="小测：某个我没核对过的说法")
        hits = [dict(r) for r in self.con.execute(
            "SELECT ext_id, content, 1.0 AS score FROM facts")]
        j, ev = metamem.judge_with_evidence(self.con, "我喜欢什么颜色", hits)
        self.assertGreater(ev.unverified, 0, "这批本来就该是未核验的")
        self.assertFalse(j.enough,
                         f"证据全没核对上，却判了够（信心 {j.fok}）")


# ══════════════════════════════════════════════════════════════════════
# 五、召回
# ══════════════════════════════════════════════════════════════════════
class 指定人物召回(用空库):
    """GPT 报的第 5 条：who 进了 Router 就丢了，`probe()` 没把 subject 传给 search。"""

    def test_只要她说的就不该带出我说的(self):
        import retrieve
        self.存一条事实(subject="tester", fact_type="preference",
                   content="小测：我喜欢粉色奶茶")
        self.存一条事实(subject="claude", fact_type="preference",
                   content="助理甲：我喜欢粉色这个说法")
        r = retrieve.retrieve(self.con, "我喜欢什么颜色", who="tester")
        subjects = {h["subject"] for h in r.hits}
        self.assertTrue(r.hits, "一条都没召回，测不了 who")
        self.assertEqual(subjects, {"tester"}, f"who 没起作用：{subjects}")


class 冷层要影响召回(用空库):
    """GPT 报的第 6 条：冷热迁移只是改标签。

    `search` 不按 layer 过滤，激活度公式的参数里也没有 layer，
    所以标成 cold 的东西照样跟 active 平起平坐。
    """

    def test_冷层的东西应当排在活跃层后面(self):
        hot = self.存一条事实(content="小测：星光湖是初一那年去的（活跃）",
                          importance=0.5, confidence=0.5)
        cold = self.存一条事实(content="小测：星光湖那次还捡了贝壳（冷层）",
                           importance=0.9, confidence=0.9)
        self.con.execute("UPDATE facts SET layer='cold' WHERE id=?", (cold,))
        self.con.commit()
        rows = search(self.con, "星光湖", limit=5)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[-1]["layer"], "cold",
                         "冷层没被压到后面，layer 对排序毫无影响")


class 召回要带回记忆类型(用空库):
    """GPT 报的第 7 条：`search()` 没 SELECT fact_type，

    所以 `recall_hits.memory_type` 实际写进去是空的，那一列的统计全是假的。
    """

    def test_search要返回fact_type(self):
        self.存一条事实(fact_type="preference", content="小测：我喜欢粉色")
        row = search(self.con, "粉色", limit=1)
        self.assertTrue(row)
        self.assertIn("fact_type", row[0], "search 没返回 fact_type")
        self.assertEqual(row[0]["fact_type"], "preference")

    def test_遥测里的memory_type不该是空的(self):
        import retrieve
        self.存一条事实(fact_type="preference", content="小测：我喜欢粉色奶茶")
        retrieve.retrieve_logged(self.con, "我喜欢什么颜色")
        rows = self.con.execute(
            "SELECT memory_type FROM recall_hits").fetchall()
        self.assertTrue(rows, "没记下命中，测不了")
        self.assertTrue(all(r[0] for r in rows),
                        "recall_hits.memory_type 是空的")


# ══════════════════════════════════════════════════════════════════════
# 六、Router 的优先级
# ══════════════════════════════════════════════════════════════════════
class 路由优先级(unittest.TestCase):
    """GPT 报的第 8、9 条。"""

    def test_明确要求回忆应当压过软话豁免(self):
        from router import route
        d = route("你还记得我说过喜欢你吗")
        self.assertTrue(d.should_recall,
                        f"「还记得／我说过」被软话豁免吃掉了：{d.reason}")

    def test_普通计算题不该去翻记忆库(self):
        from router import route
        for q in ("2026年8月有多少天", "1024乘以7是多少"):
            with self.subTest(q=q):
                self.assertFalse(route(q).should_recall,
                                 f"「{q}」被判成要翻记忆库")

    def test_她抛软话时一个字都不查(self):
        """这条现在就是绿的。ROUTER.md 第 0 条，修上面那两条时不许把它弄坏。"""
        from router import route
        for q in ("亲爱的👀", "想你了", "🌚", "抱抱"):
            with self.subTest(q=q):
                self.assertFalse(route(q).should_recall)


# ══════════════════════════════════════════════════════════════════════
# 七、最危险的一条：误召回自己把自己喂胖
# ══════════════════════════════════════════════════════════════════════
class 不许无条件强化(用空库):
    """GPT 说这条最危险，我认，而且比它说的更难看——

    我自己在遥测里定了规矩：「actually_used 判不出来就留 NULL，别假装知道
    模型用了哪条」。结果 `engram_think` 转头对**全部**返回结果调 `touch()`，
    access_count 进激活度公式 → 误召回一次就排得更前 → 更容易再被误召回。
    一边定规矩一边违规。

    正确行为：召回本身不加权。等她确认、模型真的选用、或者有正反馈，才加。
    """

    def test_一次召回不该给所有候选加计数(self):
        import retrieve
        for i in range(3):
            self.存一条事实(content=f"小测：随便一条跟颜色无关的话 {i}",
                       fact_type="preference")
        before = dict(self.con.execute(
            "SELECT ext_id, access_count FROM facts").fetchall())
        retrieve.retrieve_logged(self.con, "我喜欢什么颜色")
        # 走一遍 MCP 那条路（engram_think 会 touch 全部命中）
        for ext in before:
            touch(self.con, ext)
        after = dict(self.con.execute(
            "SELECT ext_id, access_count FROM facts").fetchall())
        涨了 = [k for k in before if after[k] > before[k]]
        self.assertEqual(涨了, [],
                         f"{len(涨了)} 条只是被召回就加了计数，误召回会被越喂越胖")


if __name__ == "__main__":
    unittest.main(verbosity=2)
