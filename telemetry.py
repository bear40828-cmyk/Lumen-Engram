from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from versions import CLASSIFIER_VERSION, ROUTER_VERSION, SALIENCE_VERSION, config_version, git_commit
CHARS_PER_TOKEN = 1.6

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def ensure(con) -> None:
    con.executescript((ROOT / 'telemetry.sql').read_text(encoding='utf-8'))

class Run:

    def __init__(self, con, *, query: str, decision: str, reason: str='', terms: list[str] | None=None, mode: str='mixed'):
        self.con, self.query, self.decision, self.reason = (con, query, decision, reason)
        self.terms = terms or []
        self.mode = mode
        self.hits: list[dict] = []
        self.candidate_count = 0
        self.context_chars = 0
        self._t0 = time.perf_counter_ns()
        self.run_id: int | None = None

    def add_hit(self, *, fact_id: int, rank: int, final_score: float, salience_score: float | None=None, lexical_score: float | None=None, layer: str | None=None, memory_type: str | None=None, snippet_chars: int=0, selected: bool=False) -> None:
        self.hits.append(dict(fact_id=fact_id, rank=rank, final_score=final_score, salience_score=salience_score, lexical_score=lexical_score, layer=layer, memory_type=memory_type, snippet_chars=snippet_chars, selected=int(bool(selected))))
        if selected:
            self.context_chars += snippet_chars

    def __enter__(self):
        ensure(self.con)
        return self

    def __exit__(self, *exc) -> None:
        latency = round((time.perf_counter_ns() - self._t0) / 1000000.0, 3)
        cur = self.con.execute('INSERT INTO recall_runs (created_at, query_text, query_hash, router_decision,\n                 router_reason, extracted_terms, retrieval_mode, candidate_count,\n                 returned_count, latency_ms, config_version, git_commit, router_version,\n                 salience_version, classifier_version, total_context_chars,\n                 total_context_tokens_est)\n               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (now(), self.query, hashlib.sha256((self.query or '').encode()).hexdigest()[:16], self.decision, self.reason, json.dumps(self.terms, ensure_ascii=False), self.mode, self.candidate_count or len(self.hits), len(self.hits), latency, config_version(), git_commit(), ROUTER_VERSION, SALIENCE_VERSION, CLASSIFIER_VERSION, self.context_chars, int(self.context_chars / CHARS_PER_TOKEN)))
        self.run_id = cur.lastrowid
        for h in self.hits:
            self.con.execute('INSERT OR IGNORE INTO recall_hits (run_id, fact_id, rank, lexical_score,\n                     salience_score, final_score, layer_before, memory_type, snippet_chars,\n                     selected_for_context, actually_used)\n                   VALUES (?,?,?,?,?,?,?,?,?,?,NULL)', (self.run_id, h['fact_id'], h['rank'], h['lexical_score'], h['salience_score'], h['final_score'], h['layer'], h['memory_type'], h['snippet_chars'], h['selected']))
        self.con.commit()

def feedback(con, run_id: int, feedback_type: str, *, fact_id: int | None=None, note: str | None=None) -> None:
    ensure(con)
    con.execute('INSERT INTO recall_feedback (run_id, fact_id, feedback_type, note, created_at) VALUES (?,?,?,?,?)', (run_id, fact_id, feedback_type, note, now()))
    con.commit()

def report(con, runs: int=0) -> None:
    ensure(con)
    total = con.execute('SELECT COUNT(*) FROM recall_runs').fetchone()[0]
    if not total:
        print('还没有召回记录。跑几次 recall.py --auto 再来看。')
        return
    skipped = con.execute("SELECT COUNT(*) FROM recall_runs WHERE router_decision='skip'").fetchone()[0]
    print(f'召回 {total} 次，其中 Router 判不查 {skipped} 次（{skipped / total:.0%}）')
    row = con.execute("SELECT AVG(returned_count), AVG(latency_ms),\n                                AVG(total_context_chars), MAX(total_context_chars)\n                         FROM recall_runs WHERE router_decision='recall'").fetchone()
    if row and row[0]:
        print(f'查的那些：平均返回 {row[0]:.1f} 条，平均 {row[1]:.3f} ms，平均进上下文 {row[2]:.0f} 字（最多 {row[3]}）')
    print('\n按检索方式：')
    for mode, n, c in con.execute("SELECT retrieval_mode, COUNT(*), AVG(total_context_chars) FROM recall_runs\n               WHERE router_decision='recall' GROUP BY retrieval_mode ORDER BY 2 DESC"):
        print(f'  {mode or '?':<8} {n:>4} 次，平均 {c or 0:.0f} 字')
    print('\n按记忆类型（命中过几次 / 有几次真进了上下文）：')
    for t, n, sel in con.execute('SELECT memory_type, COUNT(*), SUM(selected_for_context) FROM recall_hits\n               GROUP BY memory_type ORDER BY 2 DESC'):
        print(f'  {t or '?':<11} 命中 {n:>4}   进上下文 {sel or 0:>4}')
    cold = con.execute("SELECT COUNT(*) FROM recall_hits WHERE layer_before='cold'").fetchone()[0]
    print(f'\n冷层被命中 {cold} 次 ← 这个数大就说明衰减太狠')
    fb = con.execute('SELECT feedback_type, COUNT(*) FROM recall_feedback GROUP BY feedback_type').fetchall()
    print('反馈：' + ('，'.join((f'{t} {n}' for t, n in fb)) if fb else '还没有'))
    if runs:
        print(f'\n最近 {runs} 次：')
        for r in con.execute('SELECT id, created_at, router_decision, router_reason, extracted_terms,\n                          returned_count, total_context_chars FROM recall_runs\n                   ORDER BY id DESC LIMIT ?', (runs,)):
            terms = '，'.join(json.loads(r[4] or '[]')) or '—'
            print(f'  #{r[0]} {r[1][11:19]} {r[2]:<7} 词[{terms}] 返回{r[5]} 字{r[6]}  {r[3][:28]}')

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=0)
    args = ap.parse_args()
    from store.store import connect
    report(connect(), args.runs)
    return 0
if __name__ == '__main__':
    sys.exit(main())
