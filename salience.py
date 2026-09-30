from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import half_life
HALF_LIFE_DAYS = 45.0
COLD_THRESHOLD = 0.55
WARM_THRESHOLD = 0.75

def decay(age_days: float | None, fact_type: str | None=None) -> float:
    if age_days is None or age_days < 0:
        return 0.5
    return math.pow(0.5, age_days / half_life(fact_type))

def activation(*, importance: float | None, confidence: float | None, emotion: float | None, access_count: int, age_days: float | None, user_pinned: bool, fact_type: str | None=None, context_hit: float=0.0) -> float:
    if user_pinned:
        return 10.0
    base = importance if importance is not None else 0.5
    fresh = decay(age_days, fact_type)
    repeat = min(access_count, 8) * 0.06
    conf = (confidence if confidence is not None else 0.5) * 0.4
    emo = (emotion if emotion is not None else 0.0) * 0.15
    return round(base * (0.4 + 0.6 * fresh) + repeat + conf + emo + context_hit * 0.8, 4)

def register(con) -> None:
    con.create_function('engram_decay', 2, decay)

    def _act(importance, confidence, emotion, access_count, age_days, pinned, fact_type=None):
        return activation(importance=importance, confidence=confidence, emotion=emotion, access_count=access_count or 0, age_days=age_days, user_pinned=bool(pinned), fact_type=fact_type)
    con.create_function('engram_activation', 6, _act)
    con.create_function('engram_activation', 7, _act)

def migrate_layers(con, *, dry_run: bool=False) -> dict:
    rows = con.execute("\n        SELECT id, ext_id, layer, user_pinned,\n               engram_activation(importance, confidence, emotion, access_count,\n                                 julianday('now') - julianday(valid_from), user_pinned,\n                                 fact_type) AS act\n        FROM facts\n        WHERE status IN ('candidate','confirmed') AND user_pinned = 0\n    ").fetchall()
    to_cold = [r for r in rows if r['layer'] == 'active' and r['act'] < COLD_THRESHOLD]
    to_warm = [r for r in rows if r['layer'] == 'cold' and r['act'] >= WARM_THRESHOLD]
    if not dry_run:
        for r in to_cold:
            con.execute("UPDATE facts SET layer='cold', updated_at=datetime('now'), version=version+1 WHERE id=?", (r['id'],))
        for r in to_warm:
            con.execute("UPDATE facts SET layer='active', updated_at=datetime('now'), version=version+1 WHERE id=?", (r['id'],))
        con.commit()
    return {'看了': len(rows), '搬去冷层': len(to_cold), '搬回活跃': len(to_warm)}
