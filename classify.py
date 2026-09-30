from __future__ import annotations
import re
RE_CLAUSE = re.compile('[。！？；;!?\\n]+|，(?=\\S)|,(?=\\S)|(?<=\\S)(?:但是|但|不过|然后|而且|另外)')
RE_REPORT = re.compile('(听说|据说|他说|用户说|别人说|人家说|转述|原话是|好像|似乎|传闻)')
RE_PREF = re.compile('(喜欢|不喜欢|讨厌|最爱|爱吃|爱喝|想吃|想喝|受不了|不爱|爱看|爱玩)')
RE_PREF_NOUN = re.compile('(口味|偏好|最想要|吃不了|玩不了)')
FIRST = ('我', '咱', '俺')
SECOND = ('你',)
THIRD = ('用户', '他', '它', '室友', '组员', '别人', '人家', '同学', '老师', '妈', '哥', '嫂', '侄女', '导员', '主播', '网友', '那位', '用户', '助手', '助手B', '助手C')
RE_PROC = re.compile('(一律|必须|不许|别再|以后要|以后别|规矩|说好|约定|默认|优先|步骤|流程|先.{0,6}再|怎么(做|弄|起|跑|装|改)|命令|脚本|重启|部署|挂载|记得|注意|千万|不能忘|下次|(之前|前)先|先(查|问|确认|摸|看)|别拿|不要拿)')
RE_SEM = re.compile('(老家|籍贯|本科|专业|毕业|生日|属于|叫做|名字是|是我的|我的.{0,4}是|住在|在.{0,6}读|考(的是|了)?[^。]{0,6}(学校|大学|专业课)|\\d+\\s*%|满分|分值|构成|由.{0,8}组成|一共.{0,6}(科|门|条|人)|考.{0,3}科|性别|年龄|多大|几岁)')
RE_EPI = re.compile('(今天|昨天|刚才|刚刚|方才|早上|中午|下午|晚上|凌晨|\\d{1,2}[:：]\\d{2}|这次|那次|当时|然后|结果|一会儿|等会)')
CONF_FLOOR = 0.55
PRIORITY = {'preference': 3, 'semantic': 2, 'procedural': 1, 'episodic': 0}

def clauses(text: str) -> list[str]:
    return [c.strip() for c in RE_CLAUSE.split(text) if c and c.strip()]
RE_SWEAR = re.compile('(他妈|他吗|用户妈|妈的|尼玛)')

def experiencer(clause: str, verb_pos: int) -> str:
    left = RE_SWEAR.sub('', clause[max(0, verb_pos - 8):verb_pos])
    best_pos, best_kind = (-1, 'unknown')
    for kind, words in (('other', THIRD), ('self', FIRST), ('second', SECOND)):
        for w in words:
            pos = left.rfind(w)
            if pos > best_pos:
                best_pos, best_kind = (pos, kind)
    return best_kind

def classify_clause(clause: str) -> tuple[str, float]:
    if RE_REPORT.search(clause):
        return ('episodic', 0.7)
    m = RE_PREF.search(clause)
    if m:
        who = experiencer(clause, m.start())
        if who == 'self':
            return ('preference', 0.9)
        if who == 'second':
            return ('preference', 0.6)
        if who == 'other':
            return ('episodic', 0.8)
        return ('preference', 0.5)
    if RE_PREF_NOUN.search(clause):
        return ('preference', 0.7)
    proc = bool(RE_PROC.search(clause))
    sem = bool(RE_SEM.search(clause))
    epi = bool(RE_EPI.search(clause))
    if proc and (not sem):
        return ('procedural', 0.75)
    if sem and (not epi):
        return ('semantic', 0.8)
    if sem and epi:
        return ('semantic', 0.6)
    if proc:
        return ('procedural', 0.6)
    return ('episodic', 0.6)

def classify(content: str, subject: str='user') -> tuple[str, float, str]:
    body = content.split('：', 1)[-1].strip()
    if not body:
        return ('episodic', 0.5, '空正文')
    votes = [classify_clause(c) for c in clauses(body)] or [('episodic', 0.5)]
    non_epi = [v for v in votes if v[0] != 'episodic']
    if not non_epi:
        return ('episodic', max((v[1] for v in votes)), f'{len(votes)} 个分句都判事件')
    kind, raw = max(non_epi, key=lambda v: (v[1], PRIORITY[v[0]]))
    share = sum((1 for v in votes if v[0] == kind)) / len(votes)
    conf = round(min(0.98, raw * (0.75 + 0.25 * share)), 2)
    if conf < CONF_FLOOR:
        return ('episodic', conf, f'想判 {kind}，置信度只有 {conf}，按规矩退回事件')
    return (kind, conf, f'{len(votes)} 分句中 {kind} 占 {share:.0%}')
RECENCY_HALF_LIFE = {'preference': 365.0, 'semantic': 240.0, 'procedural': 120.0, 'episodic': 45.0}

def half_life(fact_type: str | None) -> float:
    return RECENCY_HALF_LIFE.get(fact_type or 'episodic', 45.0)
if __name__ == '__main__':
    samples = [('用户：我比较喜欢红色，不太喜欢绿色', 'user'), ('助手B：他昨天在会上反驳了你的方案', 'agent_b'), ('用户：我同事习惯把快递堆在工位旁边', 'user'), ('用户：听说用户喜欢红色', 'user'), ('用户：项目下个季度结束', 'user'), ('助手：这门课考三部分，每部分占比相同', 'assistant'), ('用户：回复一律用纯文本', 'user'), ('用户：今天出门忘带钥匙又折回去了', 'user')]
    for text, subj in samples:
        kind, conf, why = classify(text, subj)
        print(f'{kind:<11}{conf:>5.2f}{half_life(kind):>6.0f}天  {text[:34]:<36}{why}')
