from __future__ import annotations
import hashlib
import json

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def bigrams(text: str) -> str:
    s = ''.join((ch for ch in text if not ch.isspace()))
    return ' '.join((s[i:i + 2] for i in range(len(s) - 1))) or s

def gramset(text: str) -> set[str]:
    return set(bigrams(text).split())

def message_text(rec: dict) -> tuple[str | None, str]:
    msg = rec.get('message')
    if not isinstance(msg, dict):
        return (None, '')
    role = msg.get('role')
    if role not in ('user', 'assistant'):
        return (None, '')
    content = msg.get('content')
    if isinstance(content, str):
        return (role, content)
    if not isinstance(content, list):
        return (None, '')
    parts = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        if blk.get('type') == 'text':
            parts.append(blk.get('text', ''))
        elif blk.get('type') == 'tool_use':
            inp = blk.get('input')
            if isinstance(inp, dict) and isinstance(inp.get('text'), str):
                parts.append(inp['text'])
    return (role, '\n'.join((p for p in parts if p)))

def read_line(path: str, line_no: int) -> tuple[str | None, str] | None:
    try:
        with open(path, encoding='utf-8', errors='ignore') as fh:
            for idx, raw in enumerate(fh, start=1):
                if idx < line_no:
                    continue
                if idx > line_no:
                    break
                try:
                    return message_text(json.loads(raw))
                except json.JSONDecodeError:
                    return (None, '')
    except OSError:
        return None
    return None
