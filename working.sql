CREATE TABLE IF NOT EXISTS working_memory (
  id          INTEGER PRIMARY KEY,
  session_id  TEXT NOT NULL,
  kind        TEXT NOT NULL
      CHECK (kind IN ('goal','task','topic','state','emotion')),
  content     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open'
      CHECK (status IN ('open','done','dropped','evicted')),
  turn_no     INTEGER,
  confidence  REAL NOT NULL DEFAULT 0.6,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  closed_at   TEXT,
  evidence_id INTEGER REFERENCES evidence (id),
  note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_wm_live ON working_memory (status, kind, updated_at);
CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory (session_id, kind);
CREATE TRIGGER IF NOT EXISTS wm_no_delete BEFORE DELETE ON working_memory
BEGIN SELECT RAISE(ABORT, 'working memory is not deleted: use status evicted/dropped'); END;
