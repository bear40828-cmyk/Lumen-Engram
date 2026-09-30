CREATE TABLE IF NOT EXISTS recall_runs (
  id                     INTEGER PRIMARY KEY,
  created_at             TEXT NOT NULL,
  query_text             TEXT,
  query_hash             TEXT,
  router_decision        TEXT,
  router_reason          TEXT,
  extracted_terms        TEXT,
  retrieval_mode         TEXT,
  candidate_count        INTEGER,
  returned_count         INTEGER,
  latency_ms             REAL,
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
  layer_before        TEXT,
  memory_type         TEXT,
  snippet_chars       INTEGER,
  selected_for_context INTEGER NOT NULL DEFAULT 0,
  actually_used       INTEGER,
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
CREATE TRIGGER IF NOT EXISTS runs_append_only BEFORE UPDATE ON recall_runs
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON recall_runs
BEGIN SELECT RAISE(ABORT, 'telemetry cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS fb_append_only BEFORE UPDATE ON recall_feedback
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;
CREATE TABLE IF NOT EXISTS recall_paths (
  run_id      INTEGER NOT NULL REFERENCES recall_runs (id),
  step_no     INTEGER NOT NULL,
  task_type   TEXT,
  layer       TEXT NOT NULL,
  hits        INTEGER NOT NULL,
  fok         REAL,
  tot         INTEGER NOT NULL DEFAULT 0,
  enough      INTEGER NOT NULL DEFAULT 0,
  expects     TEXT,
  answered    TEXT,
  reason      TEXT,
  PRIMARY KEY (run_id, step_no)
);
CREATE INDEX IF NOT EXISTS idx_paths_layer ON recall_paths (layer, enough);
CREATE TRIGGER IF NOT EXISTS paths_append_only BEFORE UPDATE ON recall_paths
BEGIN SELECT RAISE(ABORT, 'telemetry is append-only'); END;
CREATE TRIGGER IF NOT EXISTS paths_no_delete BEFORE DELETE ON recall_paths
BEGIN SELECT RAISE(ABORT, 'telemetry cannot be deleted'); END;
