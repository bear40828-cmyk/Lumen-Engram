-- Lumen Engram· schema v0.1
-- 两层：evidence（原始证据，append-only）→ facts（晋升层，可作废、不可删、不可改写）
-- 铁律一：模型评分只影响排序与冷热，永不决定记录是否存在。
-- 铁律二：来源字段本身不是证据；对不上原文一律降 uncertain。
-- 铁律三：遗忘是不再容易被叫醒，不是把历史烧掉。
-- 2026-08-17 助手起草，GPT过两轮，四处收紧后冻结。

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────────────────
-- 1. evidence：原始证据层。只进不出。
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS evidence (
  id                  INTEGER PRIMARY KEY,
  ext_id              TEXT NOT NULL UNIQUE,              -- ev_<uuid>
  captured_at         TEXT NOT NULL,                     -- 入库时刻 ISO8601 UTC
  occurred_at         TEXT,                              -- 事情实际发生的时刻 UTC
  source_kind         TEXT NOT NULL
      CHECK (source_kind IN ('chat','file','tool_result','human_stated','external')),
  source_actor        TEXT NOT NULL,                     -- user/claude/gpt/agent_c/system
  source_path         TEXT,
  source_locator      TEXT NOT NULL,                     -- 文件 'L1234-1240' / 聊天 'conv:<id>#msg:<id>'
  content             TEXT NOT NULL,
  content_sha256      TEXT NOT NULL,                     -- 幂等键，不做并发控制
  verification_status TEXT NOT NULL DEFAULT 'unverified'
      CHECK (verification_status IN ('verified','unverified','uncertain','broken')),
  verified_at         TEXT,
  UNIQUE (source_locator, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_evidence_actor    ON evidence (source_actor);
CREATE INDEX IF NOT EXISTS idx_evidence_occurred ON evidence (occurred_at);
CREATE INDEX IF NOT EXISTS idx_evidence_sha      ON evidence (content_sha256);

-- 内容与来源不可改；只有校验状态可回填（provenance 校验器要写）
CREATE TRIGGER IF NOT EXISTS evidence_immutable
BEFORE UPDATE OF content, content_sha256, source_locator, source_path,
                 source_actor, source_kind, captured_at, occurred_at, ext_id
ON evidence
BEGIN
  SELECT RAISE(ABORT, 'evidence is append-only: content/source columns are immutable');
END;

CREATE TRIGGER IF NOT EXISTS evidence_no_delete
BEFORE DELETE ON evidence
BEGIN
  SELECT RAISE(ABORT, 'evidence cannot be deleted');
END;

-- ─────────────────────────────────────────────────────────
-- 2. facts：晋升层。语义字段不可改写，任何行不可删。
--    抽错了也留着，改 status='invalid'，好回头查 extractor 为什么犯错。
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS facts (
  id             INTEGER PRIMARY KEY,
  ext_id         TEXT NOT NULL UNIQUE,                   -- fact_<uuid>
  subject        TEXT NOT NULL,                          -- user/claude/gpt/project:sims/town
  fact_type      TEXT NOT NULL
      CHECK (fact_type IN ('semantic','episodic','procedural','preference')),
  content        TEXT NOT NULL,
  grams          TEXT,                                   -- bigram 串，只补两字词检索
  content_sha256 TEXT NOT NULL,

  status         TEXT NOT NULL DEFAULT 'candidate'
      CHECK (status IN ('candidate','confirmed','superseded','invalid','uncertain')),

  -- 时态。valid_from 允许为 NULL：很多口语只知道「什么时候说的」，
  -- 不知道「什么时候开始成立」。不知道就是不知道，不许硬编一个。
  -- 说话时间去 evidence 的 captured_at / occurred_at 拿。
  valid_from     TEXT,
  valid_to       TEXT,                                   -- NULL = 至今有效
  supersedes_id  INTEGER REFERENCES facts (id),

  user_pinned    INTEGER NOT NULL DEFAULT 0 CHECK (user_pinned IN (0,1)),

  -- 三个分数只影响召回排序与冷热迁移。范围焊死，防抽取器塞个 7.3 进来。
  importance     REAL CHECK (importance IS NULL OR importance BETWEEN 0 AND 1),
  emotion        REAL CHECK (emotion    IS NULL OR emotion    BETWEEN 0 AND 1),
  confidence     REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),

  layer          TEXT NOT NULL DEFAULT 'active' CHECK (layer IN ('active','cold')),
  access_count   INTEGER NOT NULL DEFAULT 0,
  last_accessed  TEXT,

  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  version        INTEGER NOT NULL DEFAULT 1              -- 乐观锁
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject);
CREATE INDEX IF NOT EXISTS idx_facts_live    ON facts (subject, status) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_layer   ON facts (layer, status);
CREATE INDEX IF NOT EXISTS idx_facts_pinned  ON facts (user_pinned) WHERE user_pinned = 1;

-- 语义字段一律不可改写。要改内容就新插一条再 supersede。
-- 可动的只有：status / valid_to / layer / 三个分数 / access_count / last_accessed
--             / user_pinned / version / updated_at
CREATE TRIGGER IF NOT EXISTS facts_semantics_immutable
BEFORE UPDATE OF content, content_sha256, grams, subject, fact_type,
                 valid_from, supersedes_id, created_at, ext_id
ON facts
BEGIN
  SELECT RAISE(ABORT, 'facts semantic columns are immutable: insert a new row and supersede');
END;

-- 任何 fact 都不许物理删，抽错的改 status='invalid' 留着
CREATE TRIGGER IF NOT EXISTS facts_no_delete
BEFORE DELETE ON facts
BEGIN
  SELECT RAISE(ABORT, 'facts cannot be deleted: set status (invalid/superseded) instead');
END;

-- ─────────────────────────────────────────────────────────
-- 3. fact_evidence：每条事实挂得住证据（多对多，支持反证）
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fact_evidence (
  fact_id     INTEGER NOT NULL REFERENCES facts (id),
  evidence_id INTEGER NOT NULL REFERENCES evidence (id),
  role        TEXT NOT NULL DEFAULT 'support'
      CHECK (role IN ('support','contradict')),
  PRIMARY KEY (fact_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_fe_evidence ON fact_evidence (evidence_id);

-- ─────────────────────────────────────────────────────────
-- 4. stances：认 / 不认 / 悬置。append-only，改主意留痕。
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stances (
  id         INTEGER PRIMARY KEY,
  fact_id    INTEGER NOT NULL REFERENCES facts (id),
  actor      TEXT NOT NULL,                              -- claude / gpt / user
  stance     TEXT NOT NULL CHECK (stance IN ('own','disown','suspend')),
  reason     TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stances_fact ON stances (fact_id, actor, created_at DESC, id DESC);

CREATE TRIGGER IF NOT EXISTS stances_append_only
BEFORE UPDATE ON stances
BEGIN
  SELECT RAISE(ABORT, 'stances are append-only: insert a new stance instead');
END;

CREATE TRIGGER IF NOT EXISTS stances_no_delete
BEFORE DELETE ON stances
BEGIN
  SELECT RAISE(ABORT, 'stances cannot be deleted');
END;

-- ─────────────────────────────────────────────────────────
-- 5. 全文检索
--    检索规则（写进检索层，别在 SQL 里猜）：
--      查询 >= 3 字 → facts_fts（trigram）
--      查询 == 2 字 → facts_gram_fts（bigram + unicode61）
--      查询 == 1 字 → 不做正常召回；只在排查时用 LIKE 当诊断
--    indexed 列（content/subject/grams）都是 immutable，所以只需 INSERT 触发器。
-- ─────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5 (
  content, subject,
  content = 'facts', content_rowid = 'id', tokenize = 'trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_gram_fts USING fts5 (
  grams,
  content = 'facts', content_rowid = 'id', tokenize = 'unicode61'
);

CREATE TRIGGER IF NOT EXISTS facts_fts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts      (rowid, content, subject) VALUES (new.id, new.content, new.subject);
  INSERT INTO facts_gram_fts (rowid, grams)            VALUES (new.id, new.grams);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5 (
  content,
  content = 'evidence', content_rowid = 'id', tokenize = 'trigram'
);

-- evidence 第一版只做回证，不做主召回，所以先不补 grams 列。
-- 将来要搜原文，照 facts 那套加一列一张表即可。
CREATE TRIGGER IF NOT EXISTS evidence_fts_ai AFTER INSERT ON evidence BEGIN
  INSERT INTO evidence_fts (rowid, content) VALUES (new.id, new.content);
END;

-- ─────────────────────────────────────────────────────────
-- 6. 视图
-- ─────────────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS live_facts AS
SELECT * FROM facts
WHERE valid_to IS NULL
  AND status IN ('candidate','confirmed');

-- 每条事实每个 actor 只取最新一条立场；同秒用 id 兜底
CREATE VIEW IF NOT EXISTS current_stances AS
SELECT fact_id, actor, stance, reason, created_at
FROM (
  SELECT s.*, ROW_NUMBER() OVER (
           PARTITION BY s.fact_id, s.actor
           ORDER BY s.created_at DESC, s.id DESC
         ) AS rn
  FROM stances s
)
WHERE rn = 1;
