from __future__ import annotations
import os
import re
_DOMAIN = "".join("|" + re.escape(t.strip()) for t in os.environ.get("ENGRAM_DOMAIN_TERMS", "").split(",") if t.strip())
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import sha256
SLOTS: dict[str, re.Pattern] = {'数值': re.compile('\\d{1,4}\\s*[年月日号点分岁%元块条页分钟小时天周]|\\d{3,}|[一二三四五六七八九十百千]+[年月日号点岁块条]'), '专名': re.compile(_DOMAIN.lstrip('|') or r'(?!x)x'), '人物': re.compile('用户|助手|室友|我哥|我妈|我爹|我姐|同学|老师|导员'), '偏好': re.compile('喜欢|讨厌|不爱|最爱|爱吃|想吃|受不了|怕|不敢|习惯|偏好|口味|最想'), '变化': re.compile('已经|做完|写完|跑通|上线|改成|换成|删了|买了|考出|拿到|收到|通了|定了|结束|开始|搬|重启|部署|修好|坏了|丢了|没了|回来了|到了|走了|睡了|起了'), '规矩': re.compile('不许|别再|以后要|以后别|规矩|说好|承诺|答应|不能|必须|一律|默认|优先'), '纠错': re.compile('记错|说错|搞错|归错|错了|我认|收回|更正|纠正|幻觉|不是我|不该')}
CHATTER = re.compile('^(嗯+|哦+|噢+|啊+|哈+|呵+|嘿+|行|好|好的|好呀|收到|在|在呢|是|对|不|没|草|靠|1|笑死|绷不住|可爱|乖|抱抱|亲亲|想你|爱你|晚安|早安|早|睡了|去洗澡|洗澡去|没什么|没事|随便|都行|不知道|谁知道|真的吗|是吧|对吧|好吧|算了)[\\W\\s]*$')
NO_TEXT = re.compile('^[\\W\\s_]*$', re.UNICODE)

@dataclass
class Verdict:
    promote: bool
    reason: str
    code: str = ''
    importance: float | None = None
    confidence: float | None = None
    slots: tuple[str, ...] = ()

def find_slots(text: str) -> tuple[str, ...]:
    return tuple((name for name, pat in SLOTS.items() if pat.search(text)))

def judge(con, *, subject: str, content: str, evidence_rows: list[dict], user_pinned: bool=False) -> Verdict:
    body = content.split('：', 1)[-1].strip()
    if user_pinned:
        return Verdict(True, 'user_pinned：用户要求留的，门控不参与', 'pinned', 1.0, 1.0)
    if NO_TEXT.match(body):
        return Verdict(False, '只有表情或标点', 'no_text')
    if not evidence_rows:
        return Verdict(False, '没有可核对的证据，留在证据层', 'no_evidence')
    verified = [e for e in evidence_rows if e.get('verification_status') == 'verified']
    if not verified:
        worst = '/'.join(sorted({e.get('verification_status') or '?' for e in evidence_rows}))
        return Verdict(False, f'证据未通过核对（{worst}）', 'evidence_unverified')
    slots = find_slots(body)
    if CHATTER.match(body) and (not slots):
        return Verdict(False, '寒暄或情绪应答，没有事实槽位', 'chatter_no_slot')
    if not slots:
        return Verdict(False, f'找不到事实槽位（{len(body)} 字）', 'no_slot', slots=slots)
    if slots == ('人物',) and len(body) < 12:
        return Verdict(False, f'只提到人名、没别的事实（{len(body)} 字）', 'short_person_only', slots=slots)
    if con.execute('SELECT 1 FROM facts WHERE subject=? AND content_sha256=?', (subject, sha256(content))).fetchone():
        return Verdict(False, '库里已有一模一样的', 'duplicate', slots=slots)
    confidence = min(1.0, 0.6 + 0.1 * len(verified))
    importance = min(0.95, 0.3 + 0.12 * len(slots) + (0.1 if len(body) >= 30 else 0.0))
    return Verdict(True, '有 ' + '＋'.join(slots), 'has_slots', round(importance, 2), round(confidence, 2), slots)
