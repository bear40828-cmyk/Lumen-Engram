PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS evidence (
  id                  INTEGER PRIMARY KEY,
  ext_id              TEXT NOT NULL UNIQUE,
  captured_at         TEXT NOT NULL,
  occurred_at         TEXT,
  source_kind         TEXT NOT NULL
      CHECK (source_kind IN ('chat','file','tool_result','human_stated','external')),
  source_actor        TEXT NOT NULL,
  source_path         TEXT,
  source_locator      TEXT NOT NULL,
  content             TEXT NOT NULL,
  content_sha256      TEXT NOT NULL,
  verification_status TEXT NOT NULL DEFAULT 'unverified'
      CHECK (verification_status IN ('verified','unverified','uncertain','broken')),
  verified_at         TEXT,
  UNIQUE (source_locator, content_sha256)
);
CREATE INDEX IF NOT EXISTS idx_evidence_actor    ON evidence (source_actor);
CREATE INDEX IF NOT EXISTS idx_evidence_occurred ON evidence (occurred_at);
CREATE INDEX IF NOT EXISTS idx_evidence_sha      ON evidence (content_sha256);
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
CREATE TABLE IF NOT EXISTS facts (
  id             INTEGER PRIMARY KEY,
  ext_id         TEXT NOT NULL UNIQUE,
  subject        TEXT NOT NULL,
  fact_type      TEXT NOT NULL
      CHECK (fact_type IN ('semantic','episodic','procedural','preference')),
  content        TEXT NOT NULL,
  grams          TEXT,
  content_sha256 TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'candidate'
      CHECK (status IN ('candidate','confirmed','superseded','invalid','uncertain')),
  valid_from     TEXT,
  valid_to       TEXT,
  supersedes_id  INTEGER REFERENCES facts (id),
  user_pinned    INTEGER NOT NULL DEFAULT 0 CHECK (user_pinned IN (0,1)),
  importance     REAL CHECK (importance IS NULL OR importance BETWEEN 0 AND 1),
  emotion        REAL CHECK (emotion    IS NULL OR emotion    BETWEEN 0 AND 1),
  confidence     REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
  layer          TEXT NOT NULL DEFAULT 'active' CHECK (layer IN ('active','cold')),
  access_count   INTEGER NOT NULL DEFAULT 0,
  last_accessed  TEXT,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  version        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject);
CREATE INDEX IF NOT EXISTS idx_facts_live    ON facts (subject, status) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_layer   ON facts (layer, status);
CREATE INDEX IF NOT EXISTS idx_facts_pinned  ON facts (user_pinned) WHERE user_pinned = 1;
CREATE TRIGGER IF NOT EXISTS facts_semantics_immutable
BEFORE UPDATE OF content, content_sha256, grams, subject, fact_type,
                 valid_from, supersedes_id, created_at, ext_id
ON facts
BEGIN
  SELECT RAISE(ABORT, 'facts semantic columns are immutable: insert a new row and supersede');
END;
CREATE TRIGGER IF NOT EXISTS facts_no_delete
BEFORE DELETE ON facts
BEGIN
  SELECT RAISE(ABORT, 'facts cannot be deleted: set status (invalid/superseded) instead');
END;
CREATE TABLE IF NOT EXISTS fact_evidence (
  fact_id     INTEGER NOT NULL REFERENCES facts (id),
  evidence_id INTEGER NOT NULL REFERENCES evidence (id),
  role        TEXT NOT NULL DEFAULT 'support'
      CHECK (role IN ('support','contradict')),
  PRIMARY KEY (fact_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS idx_fe_evidence ON fact_evidence (evidence_id);
CREATE TABLE IF NOT EXISTS stances (
  id         INTEGER PRIMARY KEY,
  fact_id    INTEGER NOT NULL REFERENCES facts (id),
  actor      TEXT NOT NULL,
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
CREATE TRIGGER IF NOT EXISTS evidence_fts_ai AFTER INSERT ON evidence BEGIN
  INSERT INTO evidence_fts (rowid, content) VALUES (new.id, new.content);
END;
CREATE VIEW IF NOT EXISTS live_facts AS
SELECT * FROM facts
WHERE valid_to IS NULL
  AND status IN ('candidate','confirmed');
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
