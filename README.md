# Lumen Engram

面向长期对话 AI 的记忆系统：让 AI 在多轮、跨会话的聊天里记住用户说过的事，并且**记得有据可查**。

> 设计文档：[docs/framework-design.md](docs/framework-design.md)（四层记忆、认知路由、遗忘与强化、冲突修正、与传统方案对比）
>
> 该文档描述的是**目标架构**；当前已实现的范围以本文末尾「状态」一节为准。

核心问题不是「存下来」，而是：

- 记住的每一条，能不能追回到原话的哪个文件、哪一行？
- 模型自己复述过的话，会不会被当成用户说的，越传越真？
- 用户随口一句撒娇闲聊，要不要去翻记忆库？

## 三条原则

1. 模型评分只影响排序与冷热，**永不决定记录是否存在**。
2. **来源字段本身不是证据**：回原文重算哈希，对不上一律降为 `uncertain`。
3. 遗忘是「不再容易被想起」，不是删除历史。

## 两层结构

这里的「两层」是**数据存储结构**（证据 / 事实）；设计文档里的「四层」（工作 / 情景 / 语义 / 程序）是**认知记忆分类**，落在事实层的 `fact_type` 上，两者不冲突。

```
evidence（原始证据，只追加，不改不删）
    │  每条带 source_path + 行号 + content_sha256 + 核对状态
    ↓  gate：决定「晋不晋升」，不是「写不写库」
facts（长期记忆，语义字段不可改写，只能作废）
    │  valid_from / valid_to / supersedes_id 串成时间线
    └─ stances：认 / 不认 / 悬置，只追加，改主意留痕
```

## 防「回声共识」

用户说一次某件事，AI 复述一次，另一个 AI 再复述一次——如果三条都算独立支撑，
系统会把 AI 的回声巩固成「稳定事实」。

本项目给证据加了 `evidence_tier`（primary / secondary / derived）和 `origin_root` 聚类，
巩固时**只数一手、且说话人就是当事人**的证据。巩固条件：≥3 条独立支撑、跨 ≥2 天、方向一致。

## 召回链路

```
用户消息
  → router    该不该查（软话、寒暄、常识直接跳过）、查什么词、读几条
  → retrieve  按任务类型决定先读哪层（情景 / 语义 / 偏好 / 程序），信心够了就停
  → metamem   元记忆：判断「答上了没有」、检出互相矛盾的记录
  → rerank    可选：LLM 精排，≥7 分才注入上下文
  → hook      以「线索，非结论」的形式注入对话
```

## 目录

| 路径 | 作用 |
|---|---|
| `schema.sql` | 证据层 / 事实层 / 表态表 |
| `store/store.py` | 幂等入证、晋升去重、作废链、钉住、乐观锁、检索 |
| `gate/gate.py` | 晋升门控：按事实槽位（数值、专名、人物、偏好、变化、规矩、纠错）判断 |
| `provenance/verify.py` | 按行号回原文重算 sha256，对不上降级 |
| `extract/` | 从会话记录 / 日摘要抽取证据；`timeline.py` 判断新事实是否顶替旧事实 |
| `classify.py` | 记忆类型分类（情景 / 语义 / 偏好 / 程序），各自不同半衰期 |
| `keys.py` + `keys.yaml` | fact_key 注册表：抽取器只能选不能造 key，带基数与冲突策略 |
| `consolidate.py` | 巩固：多条情景记忆抽象成语义记忆，不覆盖原始记录 |
| `salience.py` / `maintain.py` | 动态显著性：时间衰减、冷热分层（冷层仍可检索） |
| `router.py` / `retrieve.py` / `metamem.py` | 召回路由、分层检索、元记忆 |
| `cards.py` | 薄发现层：40 字卡片 + 命中片段，先看卡片再决定展开哪条全文 |
| `working.py` | 工作记忆：当前目标、在办事项、情绪 |
| `corrections.py` | 纠正层：从对话里挖出「用户纠正过 AI 的事实」，单独高优先级召回 |
| `rerank.py` | LLM 精排 |
| `telemetry.py` | 召回遥测（只追加，与事实层解耦） |
| `eval_recall.py` | 召回评测：原题 / 换说法 / 反面卷（日常闲聊不应误注入） |
| `mcp_server.py` | stdio MCP 服务，把召回接进对话客户端 |
| `hooks_engram_recall.py` | 对话前钩子：自动判断并注入相关记忆 |

## 运行

```sh
cp .env.example .env              # 填 LLM 接口（抽取、精排用；不填则只走规则）
export ENGRAM_ENV_FILE=.env ENGRAM_HOME=$PWD
sqlite3 engram.db < schema.sql
python3 recall.py --stats
python3 recall.py --auto "上次去海边是哪年的事"
python3 router.py                 # 看路由对样例句的判断
python3 -m unittest discover -s tests -v
```

## 实际在跑的部分

以下部件已实现并在一个长期运行的 Agent 上日常使用：

| 部件 | 在做什么 | 代码 |
| --- | --- | --- |
| 召回钩子 | 每轮对话前按当前消息检索、门控后注入上下文 | `recall.py` `retrieve.py` |
| 证据溯源 | 每条记忆绑定原始消息的位置和哈希，可回查原话 | `schema.sql` 的 `evidence` 表、`ingest.py` |
| 抽取入库 | 后台从聊天记录抽取事实写入存储 | `extract/` `ingest.py` |
| 巩固与防回声 | 合并重复记忆，AI 自己说过的话不能单独成为事实 | `consolidate.py` |
| 工作记忆与立场 | 会话内短期记忆、长期立场，都能通过工具读写 | `working.py` `metamem.py` |
| MCP 工具 | 对话模型主动查找、钉住、记录立场 | `mcp_server.py` |

还没落地的：认知路由（按场合切换记忆策略）、原子事实拆分、向量检索、情绪维度。

## Known Limitations

`tests/test_known_defects.py` 用回归用例持续跟踪 13 项已知边界问题（如原子事实拆分、冷层排序、纯计算题误触发召回），修复一项即转为通过。

## 状态

个人项目，持续迭代中。已完成：证据溯源、门控、防回声巩固、分层召回、纠正层、召回评测。
未完成：原子事实拆分、向量检索、情绪维度。
