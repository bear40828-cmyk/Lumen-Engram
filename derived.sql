-- Lumen Engram · 派生层（可随时删了重建，schema.sql v0.1 一个字不动）
-- 放两样东西：分类置信度、事实之间的关系。都不属于证据层。

-- 分类的置信度和理由。fact_type 本身在 facts 里且不可改写；
-- 这张表只记「当时为什么这么判、判得有多稳」，方便以后回头审分类质量。
CREATE TABLE IF NOT EXISTS fact_class (
  fact_id            INTEGER PRIMARY KEY REFERENCES facts (id),
  fact_type          TEXT NOT NULL,
  confidence         REAL,
  reason             TEXT,
  classifier_version TEXT,          -- 这个分数是哪版分类器打的
  classified_at      TEXT,
  built_at           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_class_type ON fact_class (fact_type, confidence);

-- 事实之间的关系。GPT 第二轮定的规矩：
--   巩固（consolidation）是抽象，不是覆盖。
--   由若干 episodic 推出一条 semantic，旧的 episode 原样留着，
--   用 derived_from / supports 连过去，绝不 supersede 它们。
--   supersede 只用于「观点或事实真的变了」（semantic v1 → v2）。
CREATE TABLE IF NOT EXISTS fact_relations (
  parent_fact_id INTEGER NOT NULL REFERENCES facts (id),   -- 抽象出来的那条
  child_fact_id  INTEGER NOT NULL REFERENCES facts (id),   -- 支撑它的原始那条
  relation       TEXT NOT NULL
      CHECK (relation IN ('derived_from','supports','contradicts')),
  created_at     TEXT NOT NULL,
  PRIMARY KEY (parent_fact_id, child_fact_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_rel_child ON fact_relations (child_fact_id);

-- fact_key 槽位映射。facts 的 schema v0.1 冻结了，所以先放派生层；
-- 将来要正式进 schema v0.2 得再过一轮 review。
-- key 只能从 keys.yaml 选，选不到就不写这条——绝不临场造 key。
CREATE TABLE IF NOT EXISTS fact_key_map (
  fact_id   INTEGER PRIMARY KEY REFERENCES facts (id),
  fact_key  TEXT NOT NULL,
  value     TEXT,
  polarity  TEXT NOT NULL DEFAULT 'pos',   -- pos / neg：「不喜欢紫色」不能跟「喜欢紫色」并一组
  eligible  INTEGER NOT NULL DEFAULT 1,    -- 0 = 命中槽位但不该参与自动巩固
  registry_version TEXT,                   -- keys.yaml 的版本，升级后能重跑
  built_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_keymap_key ON fact_key_map (fact_key, value);

-- 证据分级 + 同源聚簇。GPT 第三轮第一颗螺丝，也是最要紧的一颗：
--   用户原话说一次「考某大学」，我复述一次，GPT再复述一次——
--   如果这三条都算独立支撑，系统就会「自己给自己造共识」。
-- 所以：
--   primary   本人直接说的        → 才算独立一票
--   secondary AI 转述／摘要／引用  → 只给检索用，不加票
--   derived   系统自己生成的      → 同上
-- 并且同一句原话衍生出来的一串东西共用一个 origin_root，永远只算 1 票。
CREATE TABLE IF NOT EXISTS evidence_tier (
  evidence_id INTEGER PRIMARY KEY REFERENCES evidence (id),
  tier        TEXT NOT NULL CHECK (tier IN ('primary','secondary','derived')),
  origin_root TEXT NOT NULL,        -- 同源聚簇键：文件+行号（同一句原话只有一个）
  built_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tier_root ON evidence_tier (origin_root);
CREATE INDEX IF NOT EXISTS idx_tier_tier ON evidence_tier (tier);

-- 归档登记旁表。
-- 清理「不真删」：facts 行原样留着，只把 status 改成 uncertain 让它退出召回；
-- 这张表记「谁、哪天、哪一批、为什么、原来是什么状态」，回滚按 batch_id 一条 SQL。
-- 完整原文另存在 archive/<batch_id>-<类别>.jsonl。
CREATE TABLE IF NOT EXISTS facts_archive (
  archive_id   INTEGER PRIMARY KEY,
  fact_id      INTEGER NOT NULL REFERENCES facts (id),
  batch_id     TEXT NOT NULL,
  category     TEXT NOT NULL,
  reason       TEXT,
  prev_status  TEXT NOT NULL,
  prev_layer   TEXT,
  prev_version INTEGER,
  action       TEXT NOT NULL CHECK (action IN ('archived','registered_only','restored')),
  archived_at  TEXT NOT NULL,
  actor        TEXT NOT NULL DEFAULT 'grok',
  approved_by  TEXT,
  UNIQUE (fact_id, batch_id, action)
);

CREATE INDEX IF NOT EXISTS idx_archive_batch ON facts_archive (batch_id, category);
CREATE INDEX IF NOT EXISTS idx_archive_fact  ON facts_archive (fact_id);

CREATE TRIGGER IF NOT EXISTS archive_no_delete BEFORE DELETE ON facts_archive
BEGIN SELECT RAISE(ABORT, 'facts_archive cannot be deleted: add a restored row instead'); END;


-- ═══ 任务三（2026-10-28）入库精简：全部 CREATE IF NOT EXISTS，加性、可重复执行，不动 schema.sql ═══

-- 杂项键值（目前只放蒸馏基线 distill_baseline_id）
CREATE TABLE IF NOT EXISTS engram_meta (
  key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);

-- 采集游标：每个会话文件读到哪（纯提速，丢了不影响正确性——evidence 幂等）
CREATE TABLE IF NOT EXISTS ingest_cursor (
  source_path TEXT PRIMARY KEY, line_no INTEGER NOT NULL DEFAULT 0,
  file_size INTEGER, file_mtime REAL, updated_at TEXT NOT NULL);

-- 蒸馏游标：每个会话蒸到哪个 evidence.id（含）；失败时记退避状态。游标只在成功或进死信箱时前进。
CREATE TABLE IF NOT EXISTS distill_cursor (
  session_key TEXT PRIMARY KEY,
  last_evidence_id INTEGER NOT NULL DEFAULT 0,
  fail_count INTEGER NOT NULL DEFAULT 0,
  first_fail_at TEXT, next_try_at TEXT, last_error TEXT,
  updated_at TEXT NOT NULL);

-- 每个块一行：范围、路由、token、结果（成本和健康都从这里看）
CREATE TABLE IF NOT EXISTS distill_runs (
  run_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('idle','cap','manual')),
  session_key TEXT NOT NULL, ev_lo INTEGER, ev_hi INTEGER, n_msgs INTEGER, n_chunks INTEGER,
  route TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER,
  kept INTEGER, dup INTEGER, near_dup INTEGER, dropped INTEGER,
  status TEXT NOT NULL CHECK (status IN ('ok','failed','dead')), error TEXT,
  started_at TEXT NOT NULL, finished_at TEXT);

-- 死信箱：重试到头的块。游标已经越过它；人决定重试（retried）还是丢掉（dropped，不是删）。
CREATE TABLE IF NOT EXISTS distill_dead (
  dead_id INTEGER PRIMARY KEY, session_key TEXT NOT NULL, ev_lo INTEGER NOT NULL, ev_hi INTEGER NOT NULL,
  n_msgs INTEGER, attempts INTEGER, first_failed_at TEXT, dead_at TEXT NOT NULL, last_error TEXT,
  status TEXT NOT NULL DEFAULT 'dead' CHECK (status IN ('dead','retried','dropped')),
  resolved_at TEXT, resolved_by TEXT);
CREATE INDEX IF NOT EXISTS idx_dead_status ON distill_dead (status);

-- 每条 fact 怎么来的。facts 本体语义列冻结，所以放旁表。「自记」标记只存这里，**不写进事实正文**。
CREATE TABLE IF NOT EXISTS fact_origin (
  fact_id INTEGER PRIMARY KEY REFERENCES facts (id),
  origin TEXT NOT NULL CHECK (origin IN ('distill','distill-cap','self-note','legacy')),
  run_id TEXT, session_key TEXT, quote TEXT, prompt_version TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_origin_origin ON fact_origin (origin, created_at);
CREATE INDEX IF NOT EXISTS idx_origin_run ON fact_origin (run_id);

-- 大模型用量（llm.chat_ex 追加；只记账）
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, purpose TEXT, route TEXT, model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
  ok INTEGER NOT NULL DEFAULT 1, error TEXT);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON llm_usage (ts);

CREATE TRIGGER IF NOT EXISTS fact_origin_no_delete BEFORE DELETE ON fact_origin
BEGIN SELECT RAISE(ABORT, 'fact_origin cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS fact_origin_immutable BEFORE UPDATE ON fact_origin
BEGIN SELECT RAISE(ABORT, 'fact_origin is append-only'); END;
CREATE TRIGGER IF NOT EXISTS distill_runs_no_delete BEFORE DELETE ON distill_runs
BEGIN SELECT RAISE(ABORT, 'distill_runs cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS distill_dead_no_delete BEFORE DELETE ON distill_dead
BEGIN SELECT RAISE(ABORT, 'distill_dead cannot be deleted: mark dropped instead'); END;
CREATE TRIGGER IF NOT EXISTS llm_usage_no_delete BEFORE DELETE ON llm_usage
BEGIN SELECT RAISE(ABORT, 'llm_usage cannot be deleted'); END;


-- ═══ 任务四·步骤2（2026-10-29）软标「可能过时」：只记「较新的这条（她本人原话）可能让较旧的同槽那条过时」，
-- 不改 facts 任何一行（不封有效期、不改 status），旧的照样召回，只是排后一点并在注入时标「可能已过时」。
-- 一行 = 一次标记；只增不改不删。同一条旧事被多个新事标过时，读的时候取最新的一条。
CREATE TABLE IF NOT EXISTS fact_stale_marks (
  mark_id     INTEGER PRIMARY KEY,
  old_fact_id INTEGER NOT NULL REFERENCES facts (id),
  new_fact_id INTEGER NOT NULL REFERENCES facts (id),
  slot        TEXT,                -- 新旧共同回答的那个问题 / fact_key
  reason      TEXT,
  judged_by   TEXT NOT NULL,       -- 'fact_key' | 'llm' | 'manual'
  created_at  TEXT NOT NULL,
  UNIQUE (old_fact_id, new_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_stale_old ON fact_stale_marks (old_fact_id);
CREATE TRIGGER IF NOT EXISTS stale_marks_no_delete BEFORE DELETE ON fact_stale_marks
BEGIN SELECT RAISE(ABORT, 'fact_stale_marks cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS stale_marks_no_update BEFORE UPDATE ON fact_stale_marks
BEGIN SELECT RAISE(ABORT, 'fact_stale_marks is append-only'); END;


-- ═══ 任务四·步骤3（2026-10-29）夜间向量聚类归纳（maintenance/cluster_summarize.py，开关 ENGRAM_CLUSTER_SUMMARY）═══
-- 一行 = 一个簇的处理结果（成功/没有共同结论/被代码后检拒绝/已被旧归纳覆盖）。同一批成员只处理一次（members_hash 唯一），
-- 簇里加了新成员才会重新判。归纳本身是一条新的派生事实（正文以〔归纳〕开头，subject=她，status=candidate），
-- 用 fact_relations 的 derived_from 连到它引用的原话；原事实一行不改。
CREATE TABLE IF NOT EXISTS cluster_summary_log (
  log_id        INTEGER PRIMARY KEY,
  members_hash  TEXT NOT NULL UNIQUE,     -- 本人原话成员 id 排序后的 sha256
  size          INTEGER NOT NULL,         -- 簇大小（所有成员）
  own_count     INTEGER NOT NULL,         -- 其中本人原话条数
  outcome       TEXT NOT NULL CHECK (outcome IN ('summary','none','rejected','covered','ineligible')),
  reason        TEXT,                     -- 被拒原因（不含正文）
  summary_fact_id INTEGER REFERENCES facts (id),
  model         TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clog_fact ON cluster_summary_log (summary_fact_id);
CREATE TRIGGER IF NOT EXISTS clog_no_delete BEFORE DELETE ON cluster_summary_log
BEGIN SELECT RAISE(ABORT, 'cluster_summary_log cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS clog_no_update BEFORE UPDATE ON cluster_summary_log
BEGIN SELECT RAISE(ABORT, 'cluster_summary_log is append-only'); END;


-- ═══ 任务四·步骤5（2026-10-29）旧事实拆原子（maintenance/split_atomic.py，开关 ENGRAM_ATOMIC_SPLIT）═══
-- 一行 = 一条旧事实的处理结果。拆出来的原子事实是新的派生事实，用 fact_relations derived_from 连回原事实；原事实不改。
CREATE TABLE IF NOT EXISTS atomic_split_log (
  fact_id    INTEGER PRIMARY KEY REFERENCES facts (id),
  outcome    TEXT NOT NULL CHECK (outcome IN ('split','atomic','rejected')),
  n_atoms    INTEGER NOT NULL DEFAULT 0,
  reason     TEXT,
  model      TEXT,
  created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS atomsplit_no_delete BEFORE DELETE ON atomic_split_log
BEGIN SELECT RAISE(ABORT, 'atomic_split_log cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS atomsplit_no_update BEFORE UPDATE ON atomic_split_log
BEGIN SELECT RAISE(ABORT, 'atomic_split_log is append-only'); END;

-- ═══ 2026-10-29 按名单拆原子（split_atomic.py --proposals/--ids）：一批一个 batch_id，可整批撤回 ═══
-- 一行 = 这一批里新写的一条派生事实。撤回 = 只把这批的新事实标 invalid（不删），原事实从头到尾没动过。
CREATE TABLE IF NOT EXISTS atomic_split_batch (
  batch_id       TEXT NOT NULL,
  child_fact_id  INTEGER NOT NULL REFERENCES facts (id),
  parent_fact_id INTEGER NOT NULL REFERENCES facts (id),
  created_at     TEXT NOT NULL,
  PRIMARY KEY (batch_id, child_fact_id)
);
CREATE INDEX IF NOT EXISTS idx_asb_parent ON atomic_split_batch (parent_fact_id);
CREATE TRIGGER IF NOT EXISTS asb_no_delete BEFORE DELETE ON atomic_split_batch
BEGIN SELECT RAISE(ABORT, 'atomic_split_batch cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS asb_no_update BEFORE UPDATE ON atomic_split_batch
BEGIN SELECT RAISE(ABORT, 'atomic_split_batch is append-only'); END;
