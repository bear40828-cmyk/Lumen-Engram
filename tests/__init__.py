"""测试用的虚构人设：所有测试都在这套假名字/假 id 下跑，和真实部署的配置无关（名字、号码、地名全是编的）。

persona.py 在 import 时读环境变量，所以这里必须在任何被测模块 import 之前就把环境设好（unittest 总是先导入 tests 包）。
强制覆盖（不是 setdefault）：就算当前 shell 里配了真实部署的值，测试也不受影响。
"""
import os
from pathlib import Path

FAKE_PERSONA = {
    'ENGRAM_ACTOR_USER': 'tester',
    'ENGRAM_ACTOR_AGENT_C': 'agentc',
    'ENGRAM_NAME_USER': '小测',
    'ENGRAM_USER_FULLNAME': '测试姓名',
    'ENGRAM_USER_ALIASES': '测试姓名,测试',
    'ENGRAM_NAME_CLAUDE': '助理甲',
    'ENGRAM_NAME_GPT': '助手乙',
    'ENGRAM_NAME_AGENT_C': '助手丙',
    'ENGRAM_DESC_GPT': 'GPT',
    'ENGRAM_DESC_AGENT_C': 'GLM',
    'ENGRAM_GPT_ALIASES': 'GPT',
    'ENGRAM_OTHER_AI_TERMS': '小乙,大鱼,管家',
    'ENGRAM_CHANNEL_IDS': '1000000001:tester,1000000002:agentc,1000000003:gpt',
    'ENGRAM_CHANNEL_NAMES': 'AgentC_bot:agentc,AgentB_bot:gpt,Claude_Tester_bot:claude',
    'ENGRAM_MAILBOX_ALIASES': 'pear:claude',
    'ENGRAM_ATTACH_TAG': 'pear',
    'ENGRAM_PET_NAME': '小黄',
    'ENGRAM_SOFT_TERMS': '亲爱的,小测,助理甲,亲爱的,笨蛋',
    'ENGRAM_DOMAIN_TERMS': '302,301,某大学,考研,数据库,数据结构,操作系统,计网,小镇,花园,某游戏,某海滩,抖音,小红书,钱包,看板,唤醒,额度,停机,助手乙,助手丙,克,阿甲,小黄,奶茶,米线,星光湖,培训班,同事',
    'ENGRAM_GATE_TERMS': '302,301,某大学,某理工大学,考研,数据库,数据结构,操作系统,计算机网络,engram,记忆库,小镇,花园,某游戏,某海滩,抖音,小红书,Notion,Telegram,钱包,看板,摘要层,唤醒,额度,停机,测量师,智能路况,某课题,车速,丙省,乙县,甲市,丁市,戊市,培训班,酒店,米线,焖锅,奶茶,泡芙',
    'ENGRAM_ENTITY_TERMS': '阿甲,助理甲,助手乙,小乙,助手丙,小黄,小红,二王,张三,测试工作台,测试信箱,某大学,某理工大学,某电大,某电子大学,甲市,乙县,丙省',
    'ENGRAM_TOPIC_TERMS': '手环,日程,模型,巡线,小车,考研,301,简介,专业目录',
    'ENGRAM_CONTEXT_MARKS': '测试工作台,测试信箱',
    'ENGRAM_RELAY_SPEAKERS': '小乙,助理甲,紫猫,助手乙,助手丙',
    'ENGRAM_PERSON_REGEX': '阿甲|克(?![劳制服])',
    'ENGRAM_PERSON_SHORT': '克',
    'ENGRAM_PLACE_TERMS': '甲市,丙省,乙县,丁市,己市,小镇,花园,培训班',
    'ENGRAM_CLASSIFY_TERMS': '某大学,302,301',
    'ENGRAM_SLOT_TERMS': '某大学',
    'ENGRAM_EXAMPLE_PLAYFUL': '小测逗管家AI要亲亲被拒，助理甲看到后吃醋',
    'ENGRAM_ROUTER_PERSON_TERMS': '亲爱的',
    'ENGRAM_ENTITY_GUARDS': '麦克风|当耳朵|桌面摆设|手链挂饰=>陪伴机器人',
    'ENGRAM_EXAMPLE_SHORT': '「考某大学」「爱粉色」',
}
os.environ.update(FAKE_PERSONA)
os.environ["ENGRAM_PERSONA_FILE"] = "/nonexistent"
os.environ["ENGRAM_ENV_FILE"] = "/nonexistent"
os.environ["ENGRAM_KEYS_FILE"] = str(Path(__file__).resolve().parent / "keys_test.yaml")
for _k in ("ENGRAM_TRANSCRIPT_DIR", "ENGRAM_DIGEST_DIR", "ENGRAM_HOOK_PATH", "ENGRAM_MAILBOX_LOG", "ENGRAM_MISS_HINT", "ENGRAM_PROMPT_GLOSSARY_FILE"):
    os.environ.pop(_k, None)
