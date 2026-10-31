-- Lumen Engram · 召回遥测（append-only，跟事实层彻底解耦）
-- GPT 第二轮定的结构：一次召回和它返回的 N 条候选是一对多，所以拆成两张表，
-- 反馈再单独一张。它是**系统行为记录**，不是记忆事实——绝不写进 facts。
--
-- 有了这个才回答得了：45 天是不是太短、cold 里有多少其实常被用、
-- rank 8 的东西是不是她真正想要的、哪种类型最容易误召回。

CREATE TABLE IF NOT EXISTS recall_runs (
  id                     INTEGER PRIMARY KEY,
  created_at             TEXT NOT NULL,
  query_text             TEXT,
  query_hash             TEXT,
  router_decision        TEXT,          -- recall / skip
  router_reason          TEXT,
  extracted_terms        TEXT,          -- JSON 数组
  retrieval_mode         TEXT,          -- direct / trigram / bigram / card / mixed
  candidate_count        INTEGER,
  returned_count         INTEGER,
  latency_ms             REAL,          -- 保留小数：检索常常不到 1ms，记整数全是 0
  config_version         TEXT,
  git_commit             TEXT,
  router_version         TEXT,
  salience_version       TEXT,
  classifier_version     TEXT,
  total_context_chars    INTEGER,
  total_context_tokens_est INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_time ON recall_runs (created_at);
CREATE INDEX IF NOT EXISTS idx_runs_hash ON recall_runs (query_hash);

CREATE TABLE IF NOT EXISTS recall_hits (
  run_id              INTEGER NOT NULL REFERENCES recall_runs (id),
  fact_id             INTEGER NOT NULL REFERENCES facts (id),
  rank                INTEGER NOT NULL,
  lexical_score       REAL,
  salience_score      REAL,
  final_score         REAL,
  layer_before        TEXT,             -- active / cold
  memory_type         TEXT,
  snippet_chars       INTEGER,
  selected_for_context INTEGER NOT NULL DEFAULT 0,
  actually_used       INTEGER,          -- 判不出来就留 NULL，别假装知道
  PRIMARY KEY (run_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_hits_fact ON recall_hits (fact_id);
CREATE INDEX IF NOT EXISTS idx_hits_layer ON recall_hits (layer_before, memory_type);

CREATE TABLE IF NOT EXISTS recall_feedback (
  id            INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL REFERENCES recall_runs (id),
  fact_id       INTEGER REFERENCES facts (id),
  feedback_type TEXT NOT NULL
      CHECK (feedback_type IN ('correct','wrong','stale','irrelevant','missed','bad_source')),
  note          TEXT,
  created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fb_run ON recall_feedback (run_id);

-- 遥测只进不改不删
CREATE TRIGGER IF NOT EXISTS runs_append_only BEFORE UPDATE ON recall_runs
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON recall_runs
BEGIN SELECT RAISE(ABORT, 'telemetry cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS fb_append_only BEFORE UPDATE ON recall_feedback
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;

-- 分层召回的逐层记账（router-0.2 起）。
-- 一次召回会读一层或几层，是一对多，所以照 recall_hits 的样子单独一张表，
-- 不往 recall_runs 上加列。要回答的问题是：
--   哪类问题最容易一层就够、哪类总要升级、升级之后是不是真的答上了、
--   0.40 这个升级线定得对不对。
CREATE TABLE IF NOT EXISTS recall_paths (
  run_id      INTEGER NOT NULL REFERENCES recall_runs (id),
  step_no     INTEGER NOT NULL,          -- 第几层，从 1 起
  task_type   TEXT,                      -- L1 判出来的问题类型
  layer       TEXT NOT NULL,             -- episodic / semantic / preference / procedural
  hits        INTEGER NOT NULL,
  fok         REAL,                      -- 知道感
  tot         INTEGER NOT NULL DEFAULT 0,-- 是否话到嘴边
  enough      INTEGER NOT NULL DEFAULT 0,-- 判够了没
  expects     TEXT,                      -- 问句期待的答案类型（JSON 数组）
  answered    TEXT,                      -- 真答上的类型（JSON 数组）
  reason      TEXT,
  PRIMARY KEY (run_id, step_no)
);

CREATE INDEX IF NOT EXISTS idx_paths_layer ON recall_paths (layer, enough);

CREATE TRIGGER IF NOT EXISTS paths_append_only BEFORE UPDATE ON recall_paths
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS paths_no_delete BEFORE DELETE ON recall_paths
BEGIN SELECT RAISE(ABORT, 'telemetry cannot be deleted'); END;
