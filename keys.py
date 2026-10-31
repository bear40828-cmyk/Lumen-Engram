#!/usr/bin/env python3
"""Lumen Engram · fact_key registry 的加载与匹配

GPT 第二轮的硬规矩，全部照办：
  - extractor **只能从 registry 里选 key**，选不到留 NULL，绝不临场发明
  - key 里不编 subject（subject 已经单独一列）
  - 每个 key 从第一天就带 cardinality / conflict_policy，
    因为 fact_key 的真正价值是回答「新事实该覆盖旧的还是并存」

    python3 keys.py                 # 看这份表
    python3 keys.py "我真喜欢粉色"   # 看某句话命中哪个 key、抠出什么值
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

import persona

ROOT = Path(__file__).resolve().parent


def _registry_path() -> Path:
    """注册表文件：ENGRAM_KEYS_FILE > keys.local.yaml（不进仓库的真实部署表）> keys.yaml（通用示例）。"""
    p = persona.get("ENGRAM_KEYS_FILE")
    if p:
        return Path(p)
    local = ROOT / "keys.local.yaml"
    return local if local.exists() else ROOT / "keys.yaml"


REGISTRY = _registry_path()


@dataclass(frozen=True)
class Key:
    name: str
    cardinality: str
    value_type: str
    conflict_policy: str
    patterns: tuple[re.Pattern, ...]
    value_pattern: re.Pattern | None
    requires_self: bool = False      # 这个槽位必须是「本人的」才算数
    digit_boundary: bool = False     # 纯数字值要卡边界，别从长串数字里抠
    owner: str | None = None         # 这条槽位说的是谁的事（跟「谁说的」分开）
    consolidatable: bool = True      # 允许自动巩固成语义事实

    @property
    def single(self) -> bool:
        return self.cardinality == "single"

    @property
    def supersedes(self) -> bool:
        return self.conflict_policy == "supersede"


@lru_cache(maxsize=1)
def registry() -> tuple[Key, ...]:
    raw = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    keys = []
    for name, spec in raw.items():
        keys.append(Key(
            name=name,
            cardinality=spec.get("cardinality", "multi"),
            value_type=spec.get("value_type", "string"),
            conflict_policy=spec.get("conflict_policy", "coexist"),
            patterns=tuple(re.compile(p) for p in spec.get("patterns", [])),
            value_pattern=(re.compile(spec["value_pattern"])
                           if spec.get("value_pattern") else None),
            requires_self=bool(spec.get("requires_self")),
            digit_boundary=bool(spec.get("digit_boundary")),
            owner=(persona.USER if spec.get("owner") == "user" else spec.get("owner")),
            consolidatable=bool(spec.get("consolidatable", True)),
        ))
    return tuple(keys)


def by_name(name: str) -> Key | None:
    return next((k for k in registry() if k.name == name), None)


RE_NEG = re.compile(r"(不|没|别|莫|讨厌|受不了|不再)")
SELF = ("我", "咱", "俺")
OTHER = ("她", "他", "你", "同事", "组员", "别人", "人家", persona.N_USER, persona.N_CLAUDE, persona.N_GPT, persona.N_C)
RE_SWEAR = re.compile(r"(他妈|他吗|她妈|妈的)")


def _polarity(body: str, pos: int) -> str:
    """值前面一小段里有否定词就算 neg。「我不喜欢紫色」不能跟「我喜欢紫色」并成一组。"""
    left = body[max(0, pos - 10):pos]
    return "neg" if RE_NEG.search(left) else "pos"


def _is_self(body: str, pos: int) -> bool:
    """值前面最近的人称是不是「我」。「你这么喜欢紫色」说的是对方，不算她的偏好。"""
    left = RE_SWEAR.sub("", body[max(0, pos - 12):pos])
    best, who = -1, None
    for kind, words in (("other", OTHER), ("self", SELF)):
        for w in words:
            i = left.rfind(w)
            if i > best:
                best, who = i, kind
    return who == "self"


def match(content: str) -> tuple[str | None, str | None, str, bool]:
    """返回 (fact_key, value, polarity, eligible)。

    eligible=False 表示「命中了槽位但不该拿来自动巩固」——
    比如 requires_self 的 key 说的其实是别人，或者数字值是从长串里抠出来的。
    命中不了就返回 (None, None, "pos", False)：不许造 key。
    """
    body = content.split("：", 1)[-1]
    for key in registry():
        if not any(p.search(body) for p in key.patterns):
            continue
        value, pol, eligible = None, "pos", True
        if key.value_pattern and (m := key.value_pattern.search(body)):
            value = m.group(0)
            pol = _polarity(body, m.start())
            if key.requires_self and not _is_self(body, m.start()):
                eligible = False
            if key.digit_boundary and value.isdigit():
                before = body[m.start() - 1] if m.start() > 0 else ""
                after = body[m.end()] if m.end() < len(body) else ""
                if before.isdigit() or after.isdigit():
                    eligible = False
        return key.name, value, pol, eligible
    return None, None, "pos", False


def main() -> int:
    if len(sys.argv) > 1:
        for text in sys.argv[1:]:
            k, v, pol, ok = match(text)
            spec = by_name(k) if k else None
            tail = (f"{k}  值={v}  极性={pol}  可巩固={'是' if ok else '否'}"
                    if spec else "没命中任何 key（留 NULL，不造）")
            print(f"{text[:38]:<40}{tail}")
        return 0

    print(f"registry：{len(registry())} 个 key（{REGISTRY.name}）\n")
    for k in registry():
        print(f"  {k.name:<28}{k.cardinality:<8}{k.conflict_policy:<11}{k.value_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
