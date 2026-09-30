CREATE TABLE IF NOT EXISTS fact_class (
  fact_id            INTEGER PRIMARY KEY REFERENCES facts (id),
  fact_type          TEXT NOT NULL,
  confidence         REAL,
  reason             TEXT,
  classifier_version TEXT,
  classified_at      TEXT,
  built_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_class_type ON fact_class (fact_type, confidence);
CREATE TABLE IF NOT EXISTS fact_relations (
  parent_fact_id INTEGER NOT NULL REFERENCES facts (id),
  child_fact_id  INTEGER NOT NULL REFERENCES facts (id),
  relation       TEXT NOT NULL
      CHECK (relation IN ('derived_from','supports','contradicts')),
  created_at     TEXT NOT NULL,
  PRIMARY KEY (parent_fact_id, child_fact_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_rel_child ON fact_relations (child_fact_id);
CREATE TABLE IF NOT EXISTS fact_key_map (
  fact_id   INTEGER PRIMARY KEY REFERENCES facts (id),
  fact_key  TEXT NOT NULL,
  value     TEXT,
  polarity  TEXT NOT NULL DEFAULT 'pos',
  eligible  INTEGER NOT NULL DEFAULT 1,
  registry_version TEXT,
  built_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_keymap_key ON fact_key_map (fact_key, value);
CREATE TABLE IF NOT EXISTS evidence_tier (
  evidence_id INTEGER PRIMARY KEY REFERENCES evidence (id),
  tier        TEXT NOT NULL CHECK (tier IN ('primary','secondary','derived')),
  origin_root TEXT NOT NULL,
  built_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tier_root ON evidence_tier (origin_root);
CREATE INDEX IF NOT EXISTS idx_tier_tier ON evidence_tier (tier);
