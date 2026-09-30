from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from salience import COLD_THRESHOLD, HALF_LIFE_DAYS, WARM_THRESHOLD, migrate_layers
from store.store import connect

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真的搬，不加就是干看')
    args = ap.parse_args()
    con = connect()
    print(f'半衰期 {HALF_LIFE_DAYS:.0f} 天｜低于 {COLD_THRESHOLD} 搬冷层｜高于 {WARM_THRESHOLD} 搬回活跃')
    before = dict(con.execute('SELECT layer, COUNT(*) FROM facts GROUP BY layer').fetchall())
    result = migrate_layers(con, dry_run=not args.apply)
    print(('搬了：' if args.apply else '会搬（没动库）：') + '，'.join((f'{k} {v}' for k, v in result.items())))
    if args.apply:
        after = dict(con.execute('SELECT layer, COUNT(*) FROM facts GROUP BY layer').fetchall())
        print(f'分层：{before} → {after}')
        print('提醒：冷层不是删除，搜索照样能捞到，只是排在后面。')
    return 0
if __name__ == '__main__':
    sys.exit(main())
