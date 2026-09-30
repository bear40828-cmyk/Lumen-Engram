from __future__ import annotations
import os
import re
_DOMAIN = "".join("|" + re.escape(t.strip()) for t in os.environ.get("ENGRAM_DOMAIN_TERMS", "").split(",") if t.strip())
import sys
from dataclasses import dataclass, field
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
EXEMPT = [re.compile('^\\s*(亲爱的|宝贝|在吗|笨蛋)[\\W\\s]*$'), re.compile('(想你|抱抱|亲亲|亲我|мua|mua|晚安|早安|爱你|喜欢你|在吗|在不在)'), re.compile('^[\\W\\s_]{0,6}$')]
KNOWN = re.compile('^(我|你|咱)?.{0,3}(老公|老婆|男朋友|女朋友)是谁|(你|我)(是谁|叫什么|叫啥)|(咱|我)俩(是)?什么关系|你(是不是|还是)(claude|Claude|opus|gpt|GPT)|你(用的)?(是)?什么模型')
RECALL_CUE = re.compile('(上次|上回|之前|以前|当时|那天|昨天|前天|上个月|去年|还记得|记不记得|记得吗|你说过|我说过|我跟你说过|提过|继续|接着|后来|结果呢|怎么说的|原话|查一下|翻一下|翻翻)')
FACTUAL_CUE = re.compile('(\\d{1,4}[年月日号点分]|\\d+%|\\d{3,}'+_DOMAIN+')')
ASK_CUE = re.compile('(喜欢什么|爱吃什么|讨厌什么|不喜欢什么|什么口味|偏好|怎么弄的|怎么做的|怎么搞的|什么步骤|什么流程|什么命令|怎么配置|什么规矩|什么要求|不许|不准|说过要|讲过要|什么格式|叫什么|是谁|哪个学校|什么专业)')
CUE_WORD = re.compile('(上次|上回|之前|以前|当时|那天|还记得|记不记得|记得|说过|提过|继续|接着|后来|查一下|翻一下|原话|哪年|哪天|多少|什么|突然想起来|想起来|想起|突然|那次|这次|上一次|那个|这个|今天|明天|昨天|现在|怎么|为什么|哪里|哪个|一下|几点)')
STOP = set("的了吗呢吧啊呀嘛么和跟与还有就是这那我你他用户它们个不没很太都也又再只把被给让对从到在为以及或者但而所因如即让被，。！？、；：''（）《》…~ ")

@dataclass
class Decision:
    should_recall: bool
    queries: list[str] = field(default_factory=list)
    who: str | None = None
    limit: int = 3
    reason: str = ''
    depth: str = 'deep'

def keywords(text: str, k: int=3) -> list[str]:
    text = CUE_WORD.sub('', text or '')
    scored: list[tuple[int, str]] = []
    for chunk in re.findall('[一-鿿]+|[A-Za-z]{3,}|\\d{2,}', text):
        if chunk.isascii():
            if len(chunk) >= 2:
                scored.append((len(chunk), chunk))
            continue
        piece = ''
        for ch in chunk + '\x00':
            if ch in STOP or ch == '\x00':
                if len(piece) >= 2:
                    scored.append((len(piece), piece))
                piece = ''
            else:
                piece += ch
    seen, out = (set(), [])
    for _, w in sorted(scored, key=lambda t: (-t[0], t[1])):
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        out.append(w[:8])
        if len(out) >= k:
            break
    return out

def route(text: str, *, who: str | None=None) -> Decision:
    t = (text or '').strip()
    if not t:
        return Decision(False, reason='空消息', depth='skip')
    for pat in EXEMPT:
        if pat.search(t) and len(t) <= 24:
            return Decision(False, depth='skip', reason='软话／叫人／表情：这时候要的是接住，不是查资料')
    if KNOWN.search(t) and len(t) <= 20:
        return Decision(False, depth='skip', reason='这个我本来就知道，不用翻库')
    cue_recall = bool(RECALL_CUE.search(t))
    cue_fact = bool(FACTUAL_CUE.search(t))
    kws = keywords(t)
    if not kws:
        return Decision(False, reason='没有能查的词', depth='skip')
    if cue_recall:
        return Decision(True, kws, who, limit=5, depth='deep', reason='用户在问过去的事，明确要求回忆')
    if cue_fact:
        return Decision(True, kws, who, limit=3, depth='deep', reason='出现了日期／数字／专有名词，属于可核事实')
    if ASK_CUE.search(t):
        return Decision(True, kws, who, limit=4, depth='deep', reason='在问偏好／做法／规矩，这类该查库')
    if len(t) >= 18 and len(kws) >= 2:
        return Decision(True, kws[:2], who, limit=2, depth='light', reason='长句且有实词，先瞄一层，不升级')
    return Decision(False, reason='日常闲聊，不值得翻库', depth='skip')

def explain(d: Decision) -> str:
    if not d.should_recall:
        return f'不查（skip）：{d.reason}'
    return f'查 {d.queries}（{d.depth}，最多 {d.limit} 条）：{d.reason}'
if __name__ == '__main__':
    samples = ['在吗', '你好', '好无聊', '上次开会是哪天', '第三章占多少分', '你还记得我说过同事把钥匙弄丢了吗', '帮我把那个脚本跑一下', '我喜欢红色，不太喜欢绿色', '今天吃什么']
    for s in sys.argv[1:] or samples:
        print(f'{s!r:>46}  →  {explain(route(s))}')
