#!/usr/bin/env python3
"""把信箱（聊天应用的消息日志）聊天直接入库：原话本身就是证据，不经摘要层。
2026-10-18：10/10～10/17 信箱消息走 Monitor 旁门进会话，摘要层没抽到，信箱内容整段缺失。
只收 cc-connect 接通前（ts < CUTOFF）的真人对话；之后的走正常管线。可重复跑，同一条不会重复入。"""
from __future__ import annotations
import datetime as dt, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import persona
from common import sha256
from classify import classify
from gate.gate import judge
from keys import match as match_key
import unified
from store.store import add_evidence, connect, promote, record_class, record_key, record_tier
if not persona.MAILBOX_LOG:
    sys.exit('没配 ENGRAM_MAILBOX_LOG（信箱 messages.jsonl 的路径），无事可做')
LOG = Path(persona.MAILBOX_LOG)
CUTOFF = 1000000004          # 2026-10-17 17:51 北京，信箱接进 cc-connect
CN = dt.timezone(dt.timedelta(hours=8))
SKIP = re.compile(r'^(\[\[' + re.escape(persona.ATTACH_TAG) + r'-|\{"approve"|【自主唤醒|【已确认】|【已跳过】)')
con = connect()
done = {r[0] for r in con.execute("select source_locator from evidence where source_path=?", (str(LOG),))}
tally = {"promoted": 0, "dup": 0, "gate_no": 0, "skip": 0, "evidence": 0}
for line in LOG.read_text().splitlines():
    if not line.strip(): continue
    m = json.loads(line)
    loc = f"id{m['id']}"
    text = (m.get('text') or '').strip()
    if m['ts'] >= CUTOFF or m.get('imported') or loc in done or len(text) < 2 or SKIP.match(text):
        tally["skip"] += 1; continue
    t = dt.datetime.fromtimestamp(m['ts'], CN)
    # 说话人统一（任务三）：认得出才有名字，认不出（不是她、也不是明确写着的 AI）跳过计数，不再默认算助手
    import speakers
    actor = speakers.mailbox_actor(m)
    if actor is None:
        tally["unknown_speaker"] = tally.get("unknown_speaker", 0) + 1
        continue
    speaker = {persona.USER: persona.N_USER, 'claude': persona.N_CLAUDE, 'gpt': persona.N_GPT, persona.AGENT_C: persona.N_C}[actor]
    ev = add_evidence(con, source_kind="chat", source_actor=actor, source_path=str(LOG), source_locator=loc,
                      content=text, occurred_at=t.astimezone(dt.timezone.utc).isoformat(), verification_status="verified")
    record_tier(con, ev, "primary", f"{LOG}:{loc}"); tally["evidence"] += 1
    content = f"{speaker}（{t:%Y-%m-%d %H:%M}）：{text}"
    # 防回声审计（task2-08）：subject = actor（这条消息的说话人本人）；unified.decide 与 store.promote 两道兜底。
    v = unified.decide(con, subject=actor, content=content, evidence_rows=[{"verification_status": "verified"}])
    if not v.promote: tally["gate_no"] += 1; continue
    fid = promote(con, subject=v.subject, fact_type=v.fact_type, content=content, evidence_ids=[ev], valid_from=t.astimezone(dt.timezone.utc).isoformat(),
                  importance=v.importance, confidence=v.confidence)
    if fid is not None:
        record_class(con, fid, v.fact_type, v.type_conf, v.type_why)
        if v.fact_key: record_key(con, fid, v.fact_key, v.value, v.polarity, v.eligible)
    tally["dup" if fid is None else "promoted"] += 1
con.commit(); print(tally)
