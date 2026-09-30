from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / 'keys.yaml'

@dataclass(frozen=True)
class Key:
    name: str
    cardinality: str
    value_type: str
    conflict_policy: str
    patterns: tuple[re.Pattern, ...]
    value_pattern: re.Pattern | None
    requires_self: bool = False
    digit_boundary: bool = False
    owner: str | None = None
    consolidatable: bool = True

    @property
    def single(self) -> bool:
        return self.cardinality == 'single'

    @property
    def supersedes(self) -> bool:
        return self.conflict_policy == 'supersede'

@lru_cache(maxsize=1)
def registry() -> tuple[Key, ...]:
    raw = yaml.safe_load(REGISTRY.read_text(encoding='utf-8')) or {}
    keys = []
    for name, spec in raw.items():
        keys.append(Key(name=name, cardinality=spec.get('cardinality', 'multi'), value_type=spec.get('value_type', 'string'), conflict_policy=spec.get('conflict_policy', 'coexist'), patterns=tuple((re.compile(p) for p in spec.get('patterns', []))), value_pattern=re.compile(spec['value_pattern']) if spec.get('value_pattern') else None, requires_self=bool(spec.get('requires_self')), digit_boundary=bool(spec.get('digit_boundary')), owner=spec.get('owner'), consolidatable=bool(spec.get('consolidatable', True))))
    return tuple(keys)

def by_name(name: str) -> Key | None:
    return next((k for k in registry() if k.name == name), None)
RE_NEG = re.compile('(不|没|别|莫|讨厌|受不了|不再)')
SELF = ('我', '咱', '俺')
OTHER = ('用户', '他', '你', '室友', '组员', '别人', '人家', '用户', '助手', '助手B', '助手C')
RE_SWEAR = re.compile('(他妈|他吗|用户妈|妈的)')

def _polarity(body: str, pos: int) -> str:
    left = body[max(0, pos - 10):pos]
    return 'neg' if RE_NEG.search(left) else 'pos'

def _is_self(body: str, pos: int) -> bool:
    left = RE_SWEAR.sub('', body[max(0, pos - 12):pos])
    best, who = (-1, None)
    for kind, words in (('other', OTHER), ('self', SELF)):
        for w in words:
            i = left.rfind(w)
            if i > best:
                best, who = (i, kind)
    return who == 'self'

def match(content: str) -> tuple[str | None, str | None, str, bool]:
    body = content.split('：', 1)[-1]
    for key in registry():
        if not any((p.search(body) for p in key.patterns)):
            continue
        value, pol, eligible = (None, 'pos', True)
        if key.value_pattern and (m := key.value_pattern.search(body)):
            value = m.group(0)
            pol = _polarity(body, m.start())
            if key.requires_self and (not _is_self(body, m.start())):
                eligible = False
            if key.digit_boundary and value.isdigit():
                before = body[m.start() - 1] if m.start() > 0 else ''
                after = body[m.end()] if m.end() < len(body) else ''
                if before.isdigit() or after.isdigit():
                    eligible = False
        return (key.name, value, pol, eligible)
    return (None, None, 'pos', False)

def main() -> int:
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            k, v, pol, ok = match(text)
            spec = by_name(k) if k else None
            tail = f'{k}  值={v}  极性={pol}  可巩固={('是' if ok else '否')}' if spec else '没命中任何 key（留 NULL，不造）'
            print(f'{text[:38]:<40}{tail}')
        return 0
    print(f'registry：{len(registry())} 个 key（{REGISTRY.name}）\n')
    for k in registry():
        print(f'  {k.name:<28}{k.cardinality:<8}{k.conflict_policy:<11}{k.value_type}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
