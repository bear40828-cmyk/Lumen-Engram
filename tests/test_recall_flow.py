"""召回改造（2026-11-04）：专名表 / 专名直通 / 每轮都查 / 查询改写 / 上下文 / 评测 top-k。全部用临时库 + 假模型，不连网。"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import entities as ENT  # noqa: E402
import flags  # noqa: E402
import query_rewrite as QR  # noqa: E402
import recall_flow as RF  # noqa: E402
from store.store import connect, promote  # noqa: E402

ENV = {"ENGRAM_ENV_FILE": "/nonexistent"}


def judge_of(scores, need=True):
    """假精排：按候选正文里的关键字给分。scores = {子串: 分}，没匹配的给 1 分。"""
    def judge(query, cands, **kw):
        out = []
        for c in cands:
            out.append(max([v for k, v in scores.items() if k in c] or [1]))
        return {"need": need, "scores": out}
    return judge


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / "t.db")
        self.addCleanup(self.dir.cleanup)
        self.addCleanup(self.con.close)
        p = mock.patch.dict(os.environ, ENV)
        p.start()
        self.addCleanup(p.stop)
        try:
            import vectors  # noqa: F401
            v = mock.patch("vectors.search_many", side_effect=lambda con, texts, **kw: [[] for _ in texts])
        except ImportError:                  # 没装 numpy（VPS 的 python 就没有）→ 换一个空壳模块
            import types
            stub = types.ModuleType("vectors")
            stub.search_many = lambda con, texts, **kw: [[] for _ in texts]
            stub.search = lambda *a, **k: []
            v = mock.patch.dict(sys.modules, {"vectors": stub})
        v.start()
        self.addCleanup(v.stop)

    def fact(self, content, **kw):
        a = dict(subject="tester", fact_type="episodic", content=content, valid_from="2026-09-01",
                 importance=0.6, confidence=0.8)
        a.update(kw)
        return promote(self.con, **a)


class Discover(unittest.TestCase):
    def d(self, *texts, **kw):
        return ENT.discover(list(enumerate(texts, 1)), use_jieba=False, **kw)

    def test_naming_pattern(self):
        f = self.d("梦里大女儿叫朵朵，小女儿叫小妹。")
        self.assertEqual(f["朵朵"]["kind"], "named")
        self.assertIn("小妹", f)

    def test_naming_rejects_sentence_fragments(self):
        f = self.d("闹钟叫醒小测", "他叫我爸爸", "她叫了一辆车")
        self.assertFalse([t for t in f if f[t]["kind"] == "named"])

    def test_title_and_ident(self):
        f = self.d("她在看《星光湖奇谭》，用 Python 和 galatea-garden 做了 eat 的实验")
        self.assertEqual(f["星光湖奇谭"]["kind"], "title")
        self.assertEqual(f["galatea-garden"]["kind"], "ident")
        self.assertEqual(f["Python"]["kind"], "ident")
        self.assertEqual(f["eat"]["kind"], "oov")             # 纯小写英文词只当软专名

    def test_redup_needs_cue_and_not_stoplist(self):
        f = self.d("小测的女儿朵朵上幼儿园", "天天都在吃饭", "小测很想妈妈", "画画很开心")
        self.assertEqual(f["朵朵"]["kind"], "redup")
        self.assertNotIn("天天", f)
        self.assertNotIn("妈妈", f)
        self.assertEqual(f.get("画画", {"kind": "oov"})["kind"], "oov")

    def test_spread_weight_is_mem0_formula(self):
        self.assertAlmostEqual(ENT.spread_weight(1), 1.0)
        self.assertAlmostEqual(ENT.spread_weight(11), 1 / (1 + 0.001 * 100))
        self.assertLess(ENT.spread_weight(200), ENT.spread_weight(20))


class EntityTable(Base):
    def seed(self):
        self.fact("2026-09-21，小测梦到大女儿叫朵朵，小女儿叫小妹。")
        self.fact("小测觉得可以训练一个小模型当女儿朵朵。")
        self.fact("小测说要是带着朵朵跑了，助理甲会不会生气。")
        self.fact("小测生气了会暴躁，需要哄。")
        ENT.refresh(self.con, use_jieba=False)

    def test_refresh_and_match(self):
        self.seed()
        m = ENT.match(self.con, "我还以为你不记得朵朵了")
        self.assertEqual(m[0]["term"], "朵朵")
        self.assertTrue(m[0]["direct"])
        self.assertEqual(m[0]["df"], 3)

    def test_no_table_is_harmless(self):
        self.assertEqual(ENT.match(self.con, "朵朵"), [])

    def test_longest_match_wins(self):
        self.fact("小测养了一只叫小玩偶的猫", valid_from="2026-09-02")
        self.fact("小玩偶今天很乖", valid_from="2026-09-03")
        ENT.refresh(self.con, use_jieba=False)
        self.con.execute("INSERT OR REPLACE INTO entity_index VALUES ('花花','oov',0,2,1.0,'x')")
        m = ENT.match(self.con, "小玩偶呢")
        self.assertEqual([e["term"] for e in m], ["小玩偶"])

    def test_high_df_is_not_direct(self):
        for i in range(ENT.MAX_DIRECT_DF_BY_KIND["named"] + 3):
            self.fact(f"第{i}天，小测的狗叫旺旺，今天去遛了弯，路线{i}")
        ENT.refresh(self.con, use_jieba=False)
        m = ENT.match(self.con, "旺旺")
        self.assertTrue(m)
        self.assertFalse(m[0]["direct"])

    def test_refresh_never_deletes_rows(self):
        self.seed()
        n = self.con.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0]
        self.con.execute("UPDATE facts SET valid_to='2026-10-01' WHERE content LIKE '%朵朵%'")
        ENT.refresh(self.con, use_jieba=False)
        self.assertGreaterEqual(self.con.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0], n)
        self.assertEqual(ENT.match(self.con, "朵朵"), [])       # 失效了就不再命中

    def test_facts_for_prefers_pinned_confirmed(self):
        self.seed()
        hits = ENT.facts_for(self.con, "朵朵")
        self.assertEqual(len(hits), 3)
        self.assertTrue(all("朵朵" in h["content"] for h in hits))


class Generic(unittest.TestCase):
    def test_generic_not_direct(self):
        for w, k in (("小三", "tag"), ("男主", "named"), ("对象", "named"), ("里塞", "tag"), ("两个人", "named")):
            self.assertTrue(ENT.is_generic(w, k), w)
        for w, k in (("朵朵", "named"), ("长明", "named"), ("甲市", "config"), ("张三", "tag")):
            self.assertFalse(ENT.is_generic(w, k), w)


class Flow(EntityTable):
    def test_entity_beats_emotion_words(self):
        """原案：「带着朵朵跑了你会生气吗」注入的是「会生气会暴躁」。专名命中必须排在情绪词命中前面。"""
        self.seed()
        f = RF.recall(self.con, "要是以后我带着朵朵跑了你会生气吗", [],
                      judge=judge_of({"朵朵": 9, "暴躁": 8}), use_rewrite=False)
        self.assertTrue(f.pool[0].get("entity_hit"))
        self.assertIn("朵朵", f.selected[0]["content"])
        self.assertNotIn("暴躁", " ".join(h["content"] for h in f.selected))   # 没提专名的要 ≥9 才能挤进来

    def test_entity_direct_without_router_or_rerank_need(self):
        """原案：「我还以为你不记得朵朵了」被判无指向性词不查。精排说 need=false 也不能拦专名直通。"""
        self.seed()
        f = RF.recall(self.con, "我还以为你不记得朵朵了", [], judge=judge_of({"朵朵": 8}, need=False), use_rewrite=False)
        self.assertTrue(f.selected)
        self.assertTrue(all("朵朵" in h["content"] for h in f.selected))
        self.assertEqual(f.need, True)

    def test_entity_direct_when_model_down_uses_pool_order(self):
        self.seed()
        f = RF.recall(self.con, "我还以为你不记得朵朵了", [], judge=lambda *a, **k: None, use_rewrite=False)
        self.assertTrue(f.selected and all("朵朵" in h["content"] for h in f.selected))
        self.assertFalse(f.judged)

    def test_vectors_unavailable_still_recalls(self):
        """没装 numpy / 嵌入不可用时向量路为空，专名直通与全文照常（VPS 的 python 就没有 numpy）。"""
        self.seed()
        with mock.patch.dict(sys.modules, {"vectors": None}):
            f = RF.recall(self.con, "我还以为你不记得朵朵了", [], judge=lambda *a, **k: None, use_rewrite=False)
        self.assertTrue(f.selected and all("朵朵" in h["content"] for h in f.selected))

    def test_entity_candidates_below_threshold_are_dropped(self):
        self.seed()
        f = RF.recall(self.con, "朵朵", [], judge=judge_of({"朵朵": 2}), use_rewrite=False)
        self.assertEqual(f.selected, [])

    def test_no_entity_goes_through_rerank_not_regex(self):
        """没有专名的话也每轮都查、都交给精排：路由正则不再拦。精排 need=false → 不注。"""
        self.seed()
        calls = []

        def judge(q, cands, **kw):
            calls.append(q)
            return {"need": False, "scores": [0] * len(cands)}
        f = RF.recall(self.con, "哈哈哈好困", [], judge=judge, use_rewrite=False)
        self.assertEqual(f.selected, [])
        self.assertEqual(len(calls), 1 if f.pool else 0)

    def test_normal_gate_eight_and_two_hits(self):
        for i in range(5):
            self.fact(f"小测喜欢吃第{i}种豆浆口味的珍珠")
        f = RF.recall(self.con, "我喜欢什么豆浆口味", [], judge=judge_of({"豆浆": 9}), use_rewrite=False)
        self.assertEqual(len(f.selected), 2)                    # 最多 2 条
        f2 = RF.recall(self.con, "我喜欢什么豆浆口味", [], judge=judge_of({"豆浆": 7}), use_rewrite=False)
        self.assertEqual(f2.selected, [])                       # 无专名 <8 不注

    def test_context_goes_to_rerank_last_line_is_current(self):
        self.seed()
        seen = []

        def judge(q, cands, **kw):
            seen.append(q)
            return {"need": False, "scores": [0] * len(cands)}
        RF.recall(self.con, "那个呢", ["上一句A", "上一句B", "上一句C", "上一句D", "上一句E"], judge=judge, use_rewrite=False)
        if seen:
            lines = seen[0].split("\n")
            self.assertEqual(lines[-1], "那个呢")
            self.assertEqual(len(lines), RF.CONTEXT_TURNS + 1)

    def test_pinned_extra_goes_first(self):
        self.seed()
        pin = dict(self.con.execute("SELECT ext_id, subject, content, status, user_pinned, id FROM facts LIMIT 1").fetchone())
        pin["fact_id"] = pin["id"]
        f = RF.recall(self.con, "朵朵", [], judge=judge_of({"": 9}), use_rewrite=False, extra=[pin])
        self.assertEqual(f.pool[0]["ext_id"], pin["ext_id"])

    def test_judge_failure_no_keyword_fallback(self):
        """Serein：精排失败不回退关键词闸硬塞。"""
        for i in range(3):
            self.fact(f"小测喜欢吃第{i}种豆浆口味的珍珠")
        f = RF.recall(self.con, "我喜欢什么豆浆口味", [], judge=lambda *a, **k: None, use_rewrite=False,
                      fallback_gate=lambda q, h: True)
        self.assertEqual(f.selected, [])
        self.assertIn("不注入", f.how)

    def test_soft_block_keeps_generic_out_of_entity_channel(self):
        """工具/摘要等泛词不当专名路进池（仍可通过全文命中）。"""
        self.fact("小测喜欢一个把英雄联盟ping信号搬到AI上的工具")
        self.fact("助理甲解释思考摘要和正常回复正文的区别")
        # 假装库里有这两条软专名
        ENT.ensure_table(self.con)
        self.con.execute("INSERT OR REPLACE INTO entity_index(term,kind,direct,df,weight,updated_at) VALUES (?,?,?,?,?,?)",
                         ("工具", "quoted", 0, 50, 0.1, "2026-01-01"))
        self.con.execute("INSERT OR REPLACE INTO entity_index(term,kind,direct,df,weight,updated_at) VALUES (?,?,?,?,?,?)",
                         ("摘要", "quoted", 0, 50, 0.1, "2026-01-01"))
        self.con.commit(); ENT._CACHE.clear()
        f = RF.recall(self.con, "让小乙做新的东西，思考摘要和调用工具步骤展示", [],
                      judge=judge_of({"思考摘要": 9, "ping": 4}), use_rewrite=False)
        # 软专名不应打上 entity_hit（泛词被 SOFT_BLOCK）
        self.assertFalse(any(h.get("entity_hit") for h in f.pool))
        # 精排给思考摘要 9、ping 4 → 只注正确条
        self.assertTrue(f.selected)
        self.assertTrue(any("思考摘要" in h["content"] for h in f.selected))
        self.assertFalse(any("ping" in h["content"] for h in f.selected))

    def test_no_entity_floor_weak_inject(self):
        """专名直通但分不够 → 空，不再用 FLOOR=3 弱注一条。"""
        self.seed()
        f = RF.recall(self.con, "朵朵", [], judge=judge_of({"朵朵": 4}), use_rewrite=False)
        self.assertEqual(f.selected, [])


    def test_explicit_gate_seven(self):
        """明确回忆口吻门槛 7；日常无专名仍要 8。"""
        for i in range(4):
            self.fact(f"小测喜欢吃第{i}种豆浆口味的珍珠")
        f = RF.recall(self.con, "之前我喜欢什么豆浆口味", [], judge=judge_of({"豆浆": 7}),
                      use_rewrite=False, explicit=True)
        self.assertEqual(len(f.selected), 2)
        f2 = RF.recall(self.con, "我喜欢什么豆浆口味", [], judge=judge_of({"豆浆": 7}),
                       use_rewrite=False, explicit=False)
        self.assertEqual(f2.selected, [])

    def test_compound_cue_xiaomoxing(self):
        self.assertEqual(RF._blocked_stem_compounds("之前咱俩不也说要弄个小模型🌚"), ["小模型"])
        self.assertIn("思考摘要", RF._blocked_stem_compounds("让小乙做思考摘要和调用工具"))
        self.fact("小测和助理甲讨论自己训练一个小模型当女儿朵朵")
        self.fact("小测喜欢一个把英雄联盟ping信号搬到AI上的工具")
        f = RF.recall(self.con, "之前咱俩不也说要弄个小模型🌚", [],
                      judge=judge_of({"小模型": 8, "ping": 3}), use_rewrite=False, explicit=True)
        self.assertTrue(f.selected)
        self.assertTrue(any("小模型" in h["content"] for h in f.selected))

    def test_pinned_needs_nine_without_entity(self):
        """无专名时钉住条要 ≥9，防 #12 类纠正卡误注。"""
        pin_id = self.fact("小测明确要求：她说把这些内容也入库时要立即写入", status="confirmed", user_pinned=1)
        self.fact("大模型的性格会被用户影响，跟宠物相处也会变")
        pin = dict(self.con.execute(
            "SELECT ext_id, subject, content, status, user_pinned, id FROM facts WHERE id=?",
            (pin_id,)).fetchone())
        pin["fact_id"] = pin["id"]
        # 钉住条 8 分、正文相关 8 分：钉住不够 9，不应注钉住；若 need 且有 ≥8 的非钉住可注
        f = RF.recall(self.con, "大模型的性格会被user影响放在人人恋身上", [],
                      judge=judge_of({"入库": 8, "性格": 8}), use_rewrite=False, extra=[pin])
        self.assertFalse(any("入库" in (h.get("content") or "") for h in f.selected))

    def test_need_false_forced_still_needs_score(self):
        """forced_fn 不能在低分时硬塞。"""
        self.fact("小测喜欢吃珍珠豆浆")
        f = RF.recall(self.con, "随便说点", [],
                      judge=judge_of({"豆浆": 3}, need=False), use_rewrite=False,
                      forced_fn=lambda q, c: "豆浆" in c)
        self.assertEqual(f.selected, [])



class Rewrite(Base):
    def test_parse_tolerates_fences_and_garbage(self):
        r = QR.parse('```json\n{"questions": ["她做过关于孩子的梦吗", "女儿叫什么", "a", "b"], "entities": ["朵朵"]}\n```')
        self.assertEqual(len(r["questions"]), QR.MAX_QUESTIONS)
        self.assertEqual(QR.parse("不是json"), {"questions": [], "entities": []})

    def test_should_rewrite_is_cost_gate_only(self):
        self.assertFalse(QR.should_rewrite("👀👀"))
        self.assertFalse(QR.should_rewrite("嗯"))
        self.assertFalse(QR.should_rewrite("带着朵朵跑了会生气吗", has_direct_entity=True))
        self.assertTrue(QR.should_rewrite("那个旅行计划的事"))

    def test_model_failure_is_empty(self):
        def boom(*a, **k):
            raise RuntimeError("down")
        QR._CACHE.clear()
        self.assertEqual(QR.rewrite("那个旅行计划的事", [], chat=boom), {"questions": [], "entities": []})

    def test_rewrite_questions_reach_the_pool(self):
        self.fact("2026-09-21，小测梦到有两个女儿，大女儿叫朵朵。")
        self.fact("小测喜欢吃面片")
        QR._CACHE.clear()
        f = RF.recall(self.con, "那个旅行计划的事", [], judge=judge_of({"女儿": 9}), use_rewrite=True,
                      rewriter=lambda t, c: {"questions": ["她做过关于孩子的梦吗", "女儿叫什么"], "entities": []})
        self.assertEqual(f.calls["rewrite"], 1)
        self.assertTrue(any("女儿" in h["content"] for h in f.pool))

    def test_direct_entity_skips_rewrite_call(self):
        self.fact("大女儿叫朵朵。")
        ENT.refresh(self.con, use_jieba=False)
        f = RF.recall(self.con, "朵朵怎么样了", [], judge=judge_of({}), use_rewrite=True,
                      rewriter=lambda t, c: self.fail("不该调改写"))
        self.assertEqual(f.calls["rewrite"], 0)


class Misc(Base):
    def test_flags_default_off(self):
        self.assertFalse(flags.on("RECALL_V2"))
        self.assertFalse(flags.on("RECALL_REWRITE"))

    def test_clean_message(self):
        self.assertIsNone(RF.clean_message("【自主唤醒入口｜服务器任务，不是小测消息】读取 x"))
        got = RF.clean_message("【测试窝当前联系人资料｜来自用户在手机上保存的设置】\n你当前的备注名：甲\n拍一拍后缀：的脑袋\n\n小嘴饿了，想要爸爸喂")
        self.assertEqual(got, "小嘴饿了，想要爸爸喂")
        self.assertEqual(RF.clean_message("你好呀"), "你好呀")

    def test_topk_ignores_chat_gates(self):
        for i in range(15):
            self.fact(f"第{i}条：小测的猫喜欢吃三文鱼罐头")
        out = RF.topk(self.con, "小测的猫喜欢吃什么", k=10)
        self.assertEqual(len(out), 10)                          # 不是 2 条，也不要分数闸

    def test_topk_autobuilds_entities_on_scratch_db(self):
        self.fact("大女儿叫朵朵，在上幼儿园。")
        out = RF.topk(self.con, "朵朵", k=3, autobuild_entities=True)
        self.assertTrue(out and out[0].get("entity_hit"))

    def test_rerank_judge_entity_note_and_default_prompt_unchanged(self):
        import rerank
        sent = {}

        def chat(msgs, **kw):
            sent["p"] = msgs[1]["content"]
            sent["kw"] = kw
            return '{"need": true, "scores": {"0": 9}}'
        with mock.patch("rerank.chat", chat):
            rerank.judge("q", ["c"])
            self.assertNotIn("专名", sent["p"].split("候选记忆")[-1])
            self.assertEqual(sent["kw"]["max_tokens"], 120)
            rerank.judge("q", ["c"], entities=["朵朵"], max_candidates=20)
            self.assertIn("朵朵", sent["p"])


class HookV2(Base):
    """钩子 v2 路径：开关关＝旧路径；开＝走 recall_flow，stdout 只有注入文本。"""

    def run_hook(self, text, v2):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("engram_hook_v2", root / "hooks_engram_recall.py")
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        hook.LOG = os.devnull
        hook.recently_injected = lambda con, minutes=30: set()
        db = str(Path(self.dir.name) / "t.db")
        import store.store as S
        old = S.connect.__defaults__
        S.connect.__defaults__ = (db,)
        self.addCleanup(lambda: setattr(S.connect, "__defaults__", old))
        out = io.StringIO()
        env = {"ENGRAM_RECALL_V2": v2 if isinstance(v2, str) else ("1" if v2 else "0"), "ENGRAM_RECALL_REWRITE": "0", **ENV}
        with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", io.StringIO(json.dumps({"prompt": text}))), \
                mock.patch("sys.stdout", out), mock.patch("rerank.judge", judge_of({"朵朵": 9})), \
                mock.patch("newer.superseded_by", lambda *a, **k: None):
            hook.main()
        return out.getvalue()

    def test_v2_injects_entity_fact_v1_router_unchanged(self):
        self.fact("2026-09-21，小测梦到大女儿叫朵朵。")
        ENT.refresh(self.con, use_jieba=False)
        self.con.commit()
        out = self.run_hook("我还以为你不记得朵朵了", v2=True)
        self.assertIn("朵朵", out)
        self.assertTrue(out.startswith("【记忆库·召回v2】"))

    def test_hybrid_scenario_triggers(self):
        self.fact("小测十年前看过一部人工智能与人类的爱情小说。")
        self.fact("小测喜欢喝珍珠豆浆。")
        ENT.refresh(self.con, use_jieba=False)
        self.con.commit()
        out = self.run_hook("有一个AI与人恋爱的小说呢", v2="hybrid")
        self.assertTrue(out.startswith("【记忆库·召回v2】") or out == "")
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("hk", root / "hooks_engram_recall.py")
        hk = importlib.util.module_from_spec(spec); spec.loader.exec_module(hk)
        self.assertEqual(hk.hybrid_trigger("有一个AI与人恋爱的小说呢"), "work")
        self.assertEqual(hk.hybrid_trigger("之前咱俩不也说要弄个小模型"), "recall")
        self.assertEqual(hk.hybrid_trigger("我喜欢喝什么豆浆"), "")
        self.fact("小测之前说要弄个小模型。")
        ENT.refresh(self.con, use_jieba=False); self.con.commit()
        out3 = self.run_hook("之前咱俩不也说要弄个小模型", v2="hybrid")
        self.assertNotIn("召回v2", out3)                       # 回忆口吻留旧路径

    def test_hybrid_routes_by_direct_entity(self):
        self.fact("2026-09-21，小测梦到大女儿叫朵朵。")
        self.fact("小测喜欢喝珍珠豆浆。")
        ENT.refresh(self.con, use_jieba=False)
        self.con.commit()
        out = self.run_hook("我还以为你不记得朵朵了", v2="hybrid")
        self.assertTrue(out.startswith("【记忆库·召回v2】"))
        out2 = self.run_hook("我喜欢喝什么豆浆", v2="hybrid")
        self.assertNotIn("召回v2", out2)                      # 无专名 → 旧路径
        self.assertEqual(out2, self.run_hook("我喜欢喝什么豆浆", v2=False))




class IngestKnobs(unittest.TestCase):
    """评测整段入库的上限旋钮：缺省值不变，env 能调大。"""

    def test_defaults_unchanged(self):
        import subprocess
        code = ("import value_gate as v; import extract.from_evidence as f; import extract.distill_slice as d;"
                "print(v.MAX_PER_SLICE, v.MAX_PER_SLICE_PRIORITY, f.CHUNK_CHARS, f.PER_MSG, d.MAX_CHUNKS)")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ENGRAM_")}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
        self.assertEqual(out.stdout.split(), ["3", "5", "6000", "600", "8"], out.stderr[-300:])

    def test_env_raises_caps(self):
        import subprocess
        code = ("import value_gate as v; import extract.from_evidence as f; import extract.distill_slice as d;"
                "print(v.MAX_PER_SLICE, v.MAX_PER_SLICE_PRIORITY, f.CHUNK_CHARS, f.PER_MSG, d.MAX_CHUNKS)")
        env = {k: v for k, v in os.environ.items() if not k.startswith("ENGRAM_")}
        env.update(ENGRAM_MAX_PER_SLICE="30", ENGRAM_MAX_PER_SLICE_PRIORITY="40", ENGRAM_CHUNK_CHARS="12000",
                   ENGRAM_PER_MSG="2000", ENGRAM_SLICE_MAX_CHUNKS="40")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
        self.assertEqual(out.stdout.split(), ["30", "40", "12000", "2000", "40"], out.stderr[-300:])


if __name__ == "__main__":
    unittest.main()
