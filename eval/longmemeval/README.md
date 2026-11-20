# LongMemEval 评测

在 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 上对比 Lumen Engram 的召回与纯关键词 BM25 对照组。

每道题新建一个临时库，题目自带的历史会话逐轮写入证据层，用户每一轮直接升为候选事实（晋升门按中文对话写，英文原话进不了事实层，所以这里绕过它）。因此本评测只测路由、召回和排序，不测抽取与晋升。两组都取前 k 条（默认 10）带日期的片段，交给同一个答题模型作答，再由同一个判分模型对照标准答案判对错。

## 环境

- Python 3.10+，SQLite 需带 FTS5（官方 CPython 自带）
- 在仓库根目录执行：

```sh
python3 -m venv .venv
.venv/bin/pip install -r eval/longmemeval/requirements.txt
```

## 数据

数据集不随仓库提供，从官方来源下载（见 LongMemEval 仓库 README 的数据说明，托管在 Hugging Face）：

```sh
mkdir -p data
wget -P data https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

2026-10-03 那次结果（见根目录 README「评测」）用的是清洗前的 `longmemeval_s.json`（`xiaowu0162/longmemeval`）。两版题目基本一致，对比时请注明用的是哪一版。

## 召回模式

| `--mode` | 对应线上配置 | 说明 |
|---|---|---|
| `legacy` | `ENGRAM_RECALL_V2` 不设（默认） | 规则路由 + 分层检索（`retrieve.py`） |
| `hybrid` | `ENGRAM_RECALL_V2=hybrid` 的新路径部分 | 专名直通 + FTS + 向量 + RRF 候选池（`recall_flow.topk`）；`--rewrite` 开查询改写（会调模型） |
| `1` | `ENGRAM_RECALL_V2=1` | 候选池（≤40）经交叉编码器重排后取前 k |
| `bm25` | 对照组 | SQLite FTS5，问句拆词后 OR 查询，按 bm25 排序 |

`--arms` 指定同时跑哪几组（逗号分隔），默认「`--mode` + `bm25`」。评测取前 k 条交给答题模型，不经聊天注入时的最多 2 条与最低分阈值。

## 模型与环境变量

| 用途 | 环境变量 |
|---|---|
| 答题与判分（OpenAI 兼容接口） | `EVAL_BASE_URL`、`EVAL_API_KEY`、`EVAL_MODEL`（未设时退回 `DIGEST_BASE_URL` / `DIGEST_API_KEY` / `DIGEST_MODEL`） |
| 交叉编码器，`--xrerank api`（默认） | `XRERANK_URL`（Cohere/Jina 格式的 `/v1/rerank`）、`XRERANK_API_KEY`、`XRERANK_MODEL`（缺省 `BAAI/bge-reranker-v2-m3`）；未设 URL 时用 `DIGEST_BASE_URL` 同域名的 `/rerank` |
| 交叉编码器，`--xrerank local` | 无需 key；`pip install sentence-transformers`，首次运行从 Hugging Face 下载 `BAAI/bge-reranker-v2-m3` |
| 向量（`--vectors`，可选） | `CF_API_TOKEN`、`CF_ACCOUNT_ID`（Cloudflare Workers AI，模型见 `vectors.py` 的 `MODEL`，目前是 `@cf/baai/bge-m3`；也可写进仓库根目录的 `.env.jev`）；`--vectors` 时 meta 里记 `vector_model`，每道题的 Lumen 组 meta 里记实际建成的向量条数 `n_vectors`（0 条说明凭证没配或嵌入失败） |
| 查询改写（`--rewrite`，可选） | 走仓库的 `llm.py`：`DIGEST_*` + `REWRITE_MODEL` |

`--mock` 不调用任何模型和网络：答题取最后一条片段，判分做子串匹配，交叉编码器换成词重合打分。只用来检查流程能跑通，分数没有意义。

答题模型的输出在判分前会过一遍 `strip_thinking`：去掉 `<think>...</think>` 整段思考，以及孤立的 `</think>` 之前的内容，只留最终答案再判分（推理模型常在思考里同时提到新值和旧值，不剥干净判分会被带偏）。

新旧顶替（2026-11-21）：同一件事的新旧两条，新的排前面。
- 精排输入 `xrerank.best_window`：取含查询词最密的片段窗口，而不是前 300 字——关键的新值常埋在一段话的后半段，截前 300 字会把它截掉（实测：Fitbit 9 个月那条被截后精排分 0.0000，改后 0.9914）。
- 池内分组 `recall_flow.superseded_map`：含查询关键词的句子共享 ≥2 个池内低频实词的，算同一件事；组内最新的保留，旧的分数 ×0.5 并标 `superseded_by`（只排后、不删不封；答题时旧片段前会加「【较早版本，可能已被更新】」）。
- 精排排序 `recall_flow.order_by_xscore`：交叉编码器分差在 0.1 以内的视作并列，日期新的排前面——新旧两条的分差基本是噪声，不该由它定先后。

## 运行

先冒烟（不花额度）：

```sh
.venv/bin/python eval/longmemeval/run.py --data data/longmemeval_s_cleaned.json \
  -n 2 --arms legacy,hybrid,1,bm25 --mock --out smoke.json
```

正式跑（示例：knowledge-update 30 题，新路径 + BM25）：

```sh
export EVAL_BASE_URL=https://api.example.com/v1 EVAL_API_KEY=... EVAL_MODEL=...
export XRERANK_URL=https://api.example.com/v1/rerank XRERANK_API_KEY=...
.venv/bin/python eval/longmemeval/run.py --data data/longmemeval_s_cleaned.json \
  --types knowledge-update -n 30 --seed 7 --mode 1 --out results.json
```

题型可选 `single-session-user`、`single-session-assistant`、`single-session-preference`、`multi-session`、`knowledge-update`、`temporal-reasoning`，逗号分隔；`--types` 留空为全部题型，`-n 0` 为全部题目。抽题方式：按 `question_id` 排序后用 `--seed` 做 `random.sample`，同一份数据、同一种子抽到的题相同。

每道题两组各调用 2 次模型（答题 + 判分），`--mode 1` 另加 1 次 rerank。30 题双组约 120 次对话调用。

## 输出

`--out` 指定的 JSON，每做完一题覆盖写一次：

```json
{
  "meta": {"data": "...", "types": "...", "n": 30, "seed": 7, "k": 10, "arms": ["1", "bm25"],
           "vectors": false, "rewrite": false, "xrerank": "api", "xrerank_model": "BAAI/bge-reranker-v2-m3",
           "answer_judge_model": "...", "git_commit": "<sha>[-dirty]", "started_at": "...", "question_ids": ["..."]},
  "summary": {"1": {"correct": 0, "total": 0, "by_type": {"knowledge-update": "0/0"}}, "bm25": {"...": "..."}},
  "results": [{"id": "...", "type": "...", "q": "...", "gold": "...",
               "1": {"pred": "...", "ok": true, "n_hits": 10, "meta": {"n_pool": 40, "top_xscores": [0.9]}, "secs": 4.2,
                     "hits": [["2023-05-10", "片段内容"], "..."]},
               "bm25": {"pred": "...", "ok": false, "n_hits": 10, "meta": {"query": "..."}, "secs": 3.1}}]
}
```

每组结果里还存了 `hits`（该组交给答题模型的前 k 条片段，`[日期，内容]`），用来事后查某道题答错时正确的新值在第几条、旧值在第几条。`meta` 里 `vector_model` 记开向量时用的嵌入模型（没开记 `null`）。

`git_commit` 带 `-dirty` 表示运行时工作区有未提交改动。复现时请保留 `meta` 原样一起报告。
