"""回收站（maintenance/trash.py）：删除进回收站、撤回、15 天到期归档（不删主库行）。"""
import datetime as dt, gzip, json, sqlite3, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "maintenance"))
from store.store import connect, promote
import trash


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.con = connect(Path(self.d) / "t.db")
        self.con.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
        for extra in ("telemetry.sql", "cards.sql", "derived.sql", "working.sql"):
            p = ROOT / extra
            if p.exists():
                try:
                    self.con.executescript(p.read_text(encoding="utf-8"))
                except sqlite3.Error:
                    pass
        if not self.con.execute("SELECT 1 FROM sqlite_master WHERE name='facts_archive'").fetchone():
            self.skipTest("schema 里没有 facts_archive")
        self.ids = [promote(self.con, subject="tester", fact_type="semantic", content=f"测试事实{i}") for i in range(3)]
        self.con.commit()
        trash.ensure(self.con)

    def st(self, i):
        return tuple(self.con.execute("SELECT status, valid_to IS NULL, layer FROM facts WHERE id=?", (i,)).fetchone())

    def test_delete_restore(self):
        b = trash.delete(self.con, self.ids[:2], reason="废话", approved_by="t")
        self.assertEqual(self.st(self.ids[0]), ("invalid", 0, "active"))
        L = trash.listing(self.con)
        self.assertEqual(L[0]["in_bin"], 2)
        self.assertEqual(trash.restore(self.con, b, ids=[self.ids[0]]), 1)
        self.assertEqual(self.st(self.ids[0]), ("candidate", 1, "active"))
        self.assertEqual(trash.listing(self.con)[0]["in_bin"], 1)

    def test_confirmed_refused(self):
        self.con.execute("UPDATE facts SET status='confirmed' WHERE id=?", (self.ids[2],))
        with self.assertRaises(SystemExit):
            trash.delete(self.con, [self.ids[2]], reason="x", approved_by="t")

    def test_expire_keeps_rows(self):
        b = trash.delete(self.con, self.ids[:2], reason="废话", approved_by="t")
        n0 = self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        early = trash.expire(self.con, at=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=14)).isoformat(), out_dir=Path(self.d), apply=True)
        self.assertEqual(early, [])
        done = trash.expire(self.con, at=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=16)).isoformat(), out_dir=Path(self.d), apply=True)
        self.assertEqual(done[0]["facts"], 2)
        recs = [json.loads(l) for l in gzip.open(done[0]["path"], "rt", encoding="utf-8")]
        self.assertEqual({r["fact"]["id"] for r in recs}, set(self.ids[:2]))
        self.assertEqual(self.con.execute("SELECT COUNT(*) FROM facts").fetchone()[0], n0)     # 主库不删
        self.assertEqual(self.st(self.ids[0])[2], "cold")
        self.assertEqual(trash.listing(self.con)[0]["state"], "expired")
        with self.assertRaises(SystemExit):
            trash.restore(self.con, b)
        self.assertEqual(trash.restore(self.con, b, force=True), 2)
        with self.assertRaises(sqlite3.DatabaseError):
            self.con.execute("DELETE FROM trash_batches")

    def test_register_existing(self):
        from store.store import _bump
        i = self.ids[0]
        self.con.execute("""INSERT INTO facts_archive (fact_id,batch_id,category,reason,prev_status,prev_layer,prev_version,action,archived_at)
                            VALUES (?,?,?,?,?,?,?,'archived','2026-10-30T16:18:42-07:00')""", (i, "obsidian-x", "obsidian_flag", "勾", "candidate", "active", 1))
        _bump(self.con, i, 1, status="uncertain")
        self.assertEqual(trash.register(self.con, "obsidian-x", kind="obsidian_flag", opened_at=None, approved_by="t"), 1)
        b = self.con.execute("SELECT expires_at FROM trash_batches").fetchone()[0]
        self.assertTrue(b.startswith("2026-11-14"))


if __name__ == "__main__":
    unittest.main()
