from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from store.store import add_evidence, connect, promote, search, touch

class 用空库(unittest.TestCase):

    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.con = connect(Path(self.dir.name) / 't.db')

    def tearDown(self) -> None:
        self.con.close()
        self.dir.cleanup()

    def 存一句(self, actor='user', locator='L1', content='我喜欢红色', path='/tmp/a.jsonl', **kw):
        return add_evidence(self.con, source_kind='chat', source_actor=actor, source_locator=locator, content=content, source_path=path, **kw)

    def 存一条事实(self, *, subject='user', fact_type='preference', content='用户：我喜欢红色', valid_from='2026-08-17', importance=0.6, confidence=0.8):
        return promote(self.con, subject=subject, fact_type=fact_type, content=content, valid_from=valid_from, importance=importance, confidence=confidence)

class 证据身份键(用空库):

    def test_不同文件的同一行号同一正文应当是两条证据(self):
        a = self.存一句(locator='L1', content='测试句', path='/tmp/窗口A.jsonl')
        b = self.存一句(locator='L1', content='测试句', path='/tmp/窗口B.jsonl')
        self.assertNotEqual(a, b, '两个不同文件的 L1 被合成了一条证据')

    def test_同一文件同一行同一正文仍然只存一条(self):
        a = self.存一句(locator='L7', content='一样的话', path='/tmp/同一个.jsonl')
        b = self.存一句(locator='L7', content='一样的话', path='/tmp/同一个.jsonl')
        self.assertEqual(a, b, '真正的重复应该被去掉')

class 原始证据不可覆盖(用空库):

    def test_改证据正文会被挡(self):
        eid = self.存一句()
        with self.assertRaises(Exception):
            self.con.execute("UPDATE evidence SET content='改了' WHERE id=?", (eid,))

    def test_删证据会被挡(self):
        eid = self.存一句()
        with self.assertRaises(Exception):
            self.con.execute('DELETE FROM evidence WHERE id=?', (eid,))

    def test_改事实的语义列会被挡(self):
        fid = self.存一条事实()
        with self.assertRaises(Exception):
            self.con.execute("UPDATE facts SET content='改了' WHERE id=?", (fid,))

    def test_删事实会被挡(self):
        fid = self.存一条事实()
        with self.assertRaises(Exception):
            self.con.execute('DELETE FROM facts WHERE id=?', (fid,))

class 原子事实(用空库):
    一句三事 = '今天改完了路由；我喜欢红色；以后不许用 Markdown'

    def test_一句三事应当抽出至少三条候选(self):
        try:
            from extract.atoms import split_atoms
        except ImportError:
            self.fail('还没有原子抽取模块（extract/atoms.py）')
        self.assertGreaterEqual(len(split_atoms(self.一句三事)), 3)

    def test_三件事的类型不应当被压成一种(self):
        from classify import classify
        t, conf, why = classify(self.一句三事)
        self.fail(f'整段只得到一个类型：{t}（{why}）——三件事被压成一条')

class 旧值重新成立(用空库):

    def test_作废之后同一句应当能重新成立(self):
        fid = self.存一条事实(content='用户：我喜欢红色', valid_from='2026-08-01')
        self.con.execute("UPDATE facts SET status='invalid', valid_to='2026-08-10' WHERE id=?", (fid,))
        self.con.commit()
        again = self.存一条事实(content='用户：我喜欢红色', valid_from='2026-08-18')
        self.assertIsNotNone(again, '作废后同一句写不回来，旧值无法重新生效')
        self.assertNotEqual(again, fid, '应当是新的一行、新的有效期，不是复活旧行')

class 作废链有调用方(用空库):

    def test_仓库里应当有地方调用supersede(self):
        hits = []
        for f in ROOT.rglob('*.py'):
            if 'tests' in f.parts or '__pycache__' in f.parts:
                continue
            txt = f.read_text(encoding='utf-8', errors='ignore')
            for name in ('supersede(', 'invalidate(', 'confirm('):
                if name in txt and f'def {name}' not in txt:
                    hits.append(f'{f.name}:{name}')
        self.assertTrue(hits, 'supersede / invalidate / confirm 没有任何调用方')

class 冲突策略(用空库):

    def test_多值并存的key不应当被判冲突(self):
        import metamem
        f1 = self.存一条事实(content='用户：我喜欢红色')
        f2 = self.存一条事实(content='用户：我不喜欢绿色')
        for fid, val, pol in ((f1, '红色', 'pos'), (f2, '绿色', 'neg')):
            self.con.execute("INSERT OR REPLACE INTO fact_key_map (fact_id, fact_key, value, polarity, eligible, built_at) VALUES (?,?,?,?,1,datetime('now'))", (fid, 'preference.color', val, pol))
        self.con.commit()
        hits = [dict(r) for r in self.con.execute('SELECT ext_id, content, 1.0 AS score FROM facts')]
        ev = metamem.check_evidence(self.con, hits, terms=['红色', '绿色'])
        self.assertEqual(ev.conflicts, (), f'coexist 的 key 被判成冲突：{ev.conflicts}')

class 未核验证据要压低结论(用空库):

    def test_全是未核验的证据不该判够(self):
        import metamem
        self.存一条事实(content='用户：某个我没核对过的说法')
        hits = [dict(r) for r in self.con.execute('SELECT ext_id, content, 1.0 AS score FROM facts')]
        j, ev = metamem.judge_with_evidence(self.con, '我喜欢什么颜色', hits)
        self.assertGreater(ev.unverified, 0, '这批本来就该是未核验的')
        self.assertFalse(j.enough, f'证据全没核对上，却判了够（信心 {j.fok}）')

class 指定人物召回(用空库):

    def test_只要用户说的就不该带出我说的(self):
        import retrieve
        self.存一条事实(subject='user', fact_type='preference', content='用户：我喜欢红色奶茶')
        self.存一条事实(subject='assistant', fact_type='preference', content='助手：我喜欢红色这个说法')
        r = retrieve.retrieve(self.con, '我喜欢什么颜色', who='user')
        subjects = {h['subject'] for h in r.hits}
        self.assertTrue(r.hits, '一条都没召回，测不了 who')
        self.assertEqual(subjects, {'user'}, f'who 没起作用：{subjects}')

class 冷层要影响召回(用空库):

    def test_冷层的东西应当排在活跃层后面(self):
        hot = self.存一条事实(content='用户：第一次去海边是初一那年（活跃）', importance=0.5, confidence=0.5)
        cold = self.存一条事实(content='用户：海边那次还捡了贝壳（冷层）', importance=0.9, confidence=0.9)
        self.con.execute("UPDATE facts SET layer='cold' WHERE id=?", (cold,))
        self.con.commit()
        rows = search(self.con, '海边', limit=5)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[-1]['layer'], 'cold', '冷层没被压到后面，layer 对排序毫无影响')

class 召回要带回记忆类型(用空库):

    def test_search要返回fact_type(self):
        self.存一条事实(fact_type='preference', content='用户：我喜欢红色')
        row = search(self.con, '红色', limit=1)
        self.assertTrue(row)
        self.assertIn('fact_type', row[0], 'search 没返回 fact_type')
        self.assertEqual(row[0]['fact_type'], 'preference')

    def test_遥测里的memory_type不该是空的(self):
        import retrieve
        self.存一条事实(fact_type='preference', content='用户：我喜欢红色奶茶')
        retrieve.retrieve_logged(self.con, '我喜欢什么颜色')
        rows = self.con.execute('SELECT memory_type FROM recall_hits').fetchall()
        self.assertTrue(rows, '没记下命中，测不了')
        self.assertTrue(all((r[0] for r in rows)), 'recall_hits.memory_type 是空的')

class 路由优先级(unittest.TestCase):

    def test_明确要求回忆应当压过软话豁免(self):
        from router import route
        d = route('你还记得我说过喜欢你吗')
        self.assertTrue(d.should_recall, f'「还记得／我说过」被软话豁免吃掉了：{d.reason}')

    def test_普通计算题不该去翻记忆库(self):
        from router import route
        for q in ('2026年8月有多少天', '1024乘以7是多少'):
            with self.subTest(q=q):
                self.assertFalse(route(q).should_recall, f'「{q}」被判成要翻记忆库')

    def test_用户抛软话时一个字都不查(self):
        from router import route
        for q in ('在吗', '你好', '好的', '抱抱'):
            with self.subTest(q=q):
                self.assertFalse(route(q).should_recall)

class 不许无条件强化(用空库):

    def test_一次召回不该给所有候选加计数(self):
        import retrieve
        for i in range(3):
            self.存一条事实(content=f'用户：随便一条跟颜色无关的话 {i}', fact_type='preference')
        before = dict(self.con.execute('SELECT ext_id, access_count FROM facts').fetchall())
        retrieve.retrieve_logged(self.con, '我喜欢什么颜色')
        for ext in before:
            touch(self.con, ext)
        after = dict(self.con.execute('SELECT ext_id, access_count FROM facts').fetchall())
        涨了 = [k for k in before if after[k] > before[k]]
        self.assertEqual(涨了, [], f'{len(涨了)} 条只是被召回就加了计数，误召回会被越喂越胖')
if __name__ == '__main__':
    unittest.main(verbosity=2)
