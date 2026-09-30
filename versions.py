from __future__ import annotations
import subprocess
from functools import lru_cache
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CLASSIFIER_VERSION = 'classify-0.2'
ROUTER_VERSION = 'router-0.3'
METAMEM_VERSION = 'metamem-0.2'
RETRIEVE_VERSION = 'retrieve-0.2'
SALIENCE_VERSION = 'salience-0.2'
REGISTRY_VERSION = 'keys-0.2'
CARDS_VERSION = 'cards-0.3'
GATE_VERSION = 'gate-0.2'

@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'

def config_version() -> str:
    return '+'.join((GATE_VERSION, CLASSIFIER_VERSION, SALIENCE_VERSION, REGISTRY_VERSION, CARDS_VERSION, ROUTER_VERSION, METAMEM_VERSION, RETRIEVE_VERSION))
if __name__ == '__main__':
    print('git      ', git_commit())
    print('config   ', config_version())
    for name in ('CLASSIFIER', 'ROUTER', 'SALIENCE', 'REGISTRY', 'CARDS', 'GATE', 'METAMEM', 'RETRIEVE'):
        print(f'{name.lower():<10}', globals()[f'{name}_VERSION'])
