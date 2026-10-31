"""Lumen Engram · 事实时间线：新事顶掉旧事（Graphiti 的 valid_at / invalid_at 那块）

抽取器每天加新事，同一件事的说法会变：「chess 在做」→「chess 不做了」、
「宠物没人喂」→「宠物托给别人照看」。不处理的话新旧两条都活着，召回时自己打架。

做法：新事入库前，在**同一个 subject、还活着、也是抽取器产出**的事里找几条沾边的，
交给模型判「新的这条是不是让某条旧的不再成立」。是 → 调用方走 stale.mark（旧文是 store.supersede，已停用）：
（2026-10-29 降级：只在旁表记一条「可能过时」软标记，见 stale.py；不封旧事、不改状态。）

从严：只有「状态变了 / 说法被推翻 / 同一个问题有了新答案」才算。
补充、细化、同一话题的另一件事，都不算——那是两条都活着。
判不出来、调用失败，一律当不顶替，照常新增。宁可两条并存，不许误杀。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from router import keywords  # noqa: E402
from llm import chat, LLMNotConfigured  # noqa: E402

import persona  # noqa: E402
ENV_FILE = persona.env_file()
TIMEOUT_S = 30
MAX_OLD = 6      # candidates 每次最多给模型几条旧事

PROMPT = persona.fill("""记忆库里有同一个人的几条旧记录，现在来了一条新记录。逐条判断「新记录」和每条「旧记录」的关系。

第一步：给新记录定类型（new_kind）：
- state 当前状态/所在/在用/关系/身份/习惯  - opinion 偏好/看法/决定/立场
- plan 打算/计划/待办  - event 某时某刻发生过的一件事（做了、说了、去了、完成了）

第二步：对每条旧记录给 old_kind（同上四类）和 relation，只能是下面之一，按这个顺序优先判：
1. enriches  新记录是补充、细化、进展、后续、更具体、范围缩小，旧记录依然成立（做了一步又做了一步、项目有了新进展、同一件事说得更细）
2. duplicate 新旧说的是同一件事，只是换了说法
3. different 同一话题下的另一件事，或回答的是不同的问题
4. updates   新记录明确让旧记录**不再成立**：两者不可能同时为真（现在这个时间点），回答的是同一个问题，新的取代旧的。旧记录必须是「当前状态/偏好/看法/决定/计划」。
   旧记录是计划/打算/待办时：新记录说明它被取消了、改了、没做成、被别的安排取代 → updates；只是「开始做了/做了一步/做完了」 → enriches

硬规则：
- 过去发生过的事件（event）永远不会因为后来的事而变假。「周三去了牙科」不会被「周五去了牙科」顶掉。旧记录是 event 就不能是 updates。
- 不要因为「更新」「同一批人/物」「时间更晚」就判 updates。只有新记录**自己明说或明显蕴含**旧状态已结束/改了/说错了才行。
- 新记录是在旧事上又做了点什么（继续、推进、完成其中一步、遇到问题），旧事仍成立 → enriches。
- 旧记录是一整串清单，新记录只改其中一项 → enriches。
- 拿不准，一律 enriches 或 different，不要 updates。

例子（虚构）：
  旧「小林在用旧手机」 新「小林换了新手机」 → 同一个问题「小林用什么手机」，旧状态已结束 → updates
  旧「小林周三要去牙科复诊」 新「小林周三下午三点去牙科复诊」 → 更具体 → enriches
  旧「小林周三去了牙科」 新「小林周五去了牙科」 → 两件事（event） → different
  旧「小林正在写小说第三章」 新「小林把第三章写完了，开始写第四章」 → 进展 → enriches
  旧「小林最后一天上课是9月23日」 新「小林最后一天上课是9月24日」 → 同一个问题换了答案 → updates
  旧「小林打算自己做 App」 新「小林决定不做 App 了，改用现成的」 → 计划被推翻 → updates
  旧「小林打算周六去爬山」 新「小林周六在家睡了一天，没去爬山」 → 计划没成，被取代 → updates

新记录（{new_when}）：{new}

旧记录：
{olds}

只输出 JSON：{{"new_kind":"state|opinion|plan|event","items":[{{"id":编号,"old_kind":"state|opinion|plan|event","relation":"enriches|duplicate|different|updates","question":"若 updates：新旧共同回答的那个问题，否则空"}}]}}""")


from llm import load_env as _env  # noqa: E402,F401  共用配置读取（见 llm.py）


def _bg2(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def near_duplicate(a: str, b: str) -> bool:
    """两句话几乎是一回事（字面重合 ≥ 0.7）且数字（含汉字数字）完全一样：算重复，不是「新事推翻旧事」。
    借鉴 Graphiti 的 dedupe 在 contradiction 之前：重复的先剔掉，不进矛盾判断；
    数字/日期不同的绝不算重复（「9月23日」和「9月24日」是一个问题换了答案）。"""
    ga, gb = _bg2(a), _bg2(b)
    if not ga or not gb:
        return False
    num = r"\d+|[一二三四五六七八九十零两]+"
    return len(ga & gb) / len(ga | gb) >= 0.7 and set(re.findall(num, a)) == set(re.findall(num, b))


def candidates(con, subject: str, content: str, before: str | None) -> list[dict]:
    """同 subject、活着、抽取器产出、发生在新事之前的，按关键词找几条沾边的。"""
    # 2026-10-28（任务二·步骤4）冻结：9/23 试过入库时自动封旧事，判得不准就停了（--no-timeline），
    # 新旧冲突交给召回时的 newer.py。入口默认关；ENGRAM_TIMELINE=1 恢复（顶替还要走 store.supersede，
    # 所以同时需要 ENGRAM_INVALIDATE_CHAIN=1）。代码原样保留。
    import flags
    if not flags.on("TIMELINE"):
        return []
    terms = [t for t in keywords(content, k=4) if len(t) >= 2]
    seen, out = set(), []
    for t in terms:
        # 一律加引号当短语（和 store.search 一样）：带 - . : AND NOT 之类字符的词不会被当成 FTS 语法而整词丢掉
        table = "facts_gram_fts" if len(t) == 2 else "facts_fts"
        match = '"%s"' % t.replace('"', '""')
        sql = f"""SELECT f.id, f.ext_id, f.content, f.valid_from FROM {table} x
                  JOIN facts f ON f.id = x.rowid
                  WHERE {table} MATCH ? AND f.subject = ?
                    AND f.status = 'candidate' AND f.user_pinned = 0 AND f.valid_to IS NULL   -- 只顶 candidate、没钉住的
                    AND f.id IN (SELECT fact_id FROM fact_class WHERE reason LIKE 'extract-llm-%')"""
        args: list = [match, subject]
        if before:
            # 旧事必须比新事**早至少一个日历日**。2026-10-29 实测：同一天里的两句话判「后一句推翻前一句」
            # 12 条里只有 2 条对；≥1 天之后的 6 条里 3 条对。借鉴 Graphiti：先用时间窗把「同时」的排除，
            # 再让模型判矛盾（它的 invalid_at/expired_at 也是代码按时间先后定的，不让模型管时间）。
            # 没有 valid_from 的旧事排不出先后，也不进。
            sql += " AND f.valid_from IS NOT NULL AND date(f.valid_from) < date(?)"
            args.append(before)
        sql += " ORDER BY f.valid_from DESC LIMIT 4"
        try:
            rows = con.execute(sql, args).fetchall()
        except Exception:                                            # noqa: BLE001
            continue            # FTS 语法不认的词（带引号、符号）直接跳过
        for r in rows:
            if r["id"] not in seen:
                seen.add(r["id"])
                if near_duplicate(content, r["content"]):
                    continue
                out.append(dict(r))
    return out[:MAX_OLD]


def _confirm_on() -> bool:
    """第二遍复核开关：ENGRAM_TIMELINE_CONFIRM=1（flags 登记过）。"""
    import flags
    return flags.on("TIMELINE_CONFIRM")


KINDS_OK = ("state", "opinion", "plan")      # 旧事只有这三类才可能被顶（过去发生过的事件永远不会变假，BeliefMem 的 D2）


def _classify(new: str, new_when: str, olds: list[dict], *, purpose: str = "extract") -> dict:
    text = chat([
        {"role": "system", "content": "你只输出合法 JSON。"},
        {"role": "user", "content": PROMPT.format(
            new=new, new_when=(new_when or "")[:10],
            olds="\n".join(f"[{i}]（{(o.get('valid_from') or '')[:10]}）{o['content']}"
                           for i, o in enumerate(olds)))},
    ], purpose=purpose, max_tokens=400, timeout=TIMEOUT_S)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def _updates(got: dict, olds: list[dict]) -> list[tuple[int, str, str]]:
    """从分类结果里挑出「真顶替」：relation=updates、旧事不是事件、说得出共同的问题。返回 [(下标, 问题, old_kind)]。"""
    out = []
    for it in got.get("items") or []:
        try:
            i = int(it.get("id"))
        except (TypeError, ValueError):
            continue
        q = str(it.get("question") or "").strip()
        if it.get("relation") != "updates" or not 0 <= i < len(olds):
            continue
        if str(it.get("old_kind")) not in KINDS_OK or not q:
            continue
        out.append((i, q, str(it.get("old_kind"))))
    return out


def judge(new: str, new_when: str, olds: list[dict]) -> tuple[int | None, str]:
    """返回 (被顶掉的那条在 olds 里的下标 或 None, 理由)。任何失败都当不顶替。

    2026-10-29 重写（见 TALK 的「新事顶旧事」调研）：不再问「有没有顶掉」，而是给每条旧事分类
    {enriches 补充/进展, duplicate, different, updates}（BeliefMem 的 EXP/DUP/CON/REV 优先级 + Graphiti
    「进展不是矛盾」），再由代码卡：只认 updates、旧事必须是 状态/看法/计划（不能是已发生的事件）、必须说得出共同问题。
    第二遍（级联，ENGRAM_TIMELINE_CONFIRM=1）：命中的这一对再交给 TIMELINE_CONFIRM_MODEL
    （没配就退回 EXTRACT_MODEL）用同一套标准重判，两遍都说 updates 才算。"""
    import flags
    if not flags.on("TIMELINE"):
        return (None, "时间线判断已冻结")
    if not olds:
        return None, "没有沾边的旧事"
    try:
        hits = _updates(_classify(new, new_when, olds), olds)
        if not hits:
            return None, "没有哪条旧事被顶替"
        # 多条命中取最近的一条（只标一条，保持「单目标」）
        i, q, kind = max(hits, key=lambda h: olds[h[0]].get("valid_from") or "")
        if _confirm_on():
            again = _updates(_classify(new, new_when, [olds[i]], purpose="timeline_confirm"), [olds[i]])
            if not again:
                return None, "第二遍复核不认，不顶替"
            q = again[0][1] or q
        return i, f"{q}｜{kind}被新说法顶替"
    except LLMNotConfigured:
        return None, "没配模型"
    except Exception as e:                                           # noqa: BLE001
        return None, f"调用失败：{e}"
