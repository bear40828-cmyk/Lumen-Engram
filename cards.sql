CREATE TABLE IF NOT EXISTS fact_cards (
  fact_id    INTEGER PRIMARY KEY REFERENCES facts (id),
  subject    TEXT NOT NULL,
  card       TEXT NOT NULL,
  card_grams TEXT NOT NULL,
  full_len   INTEGER NOT NULL,
  built_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_subject ON fact_cards (subject);
CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5 (
  card, subject,
  content = 'fact_cards', content_rowid = 'fact_id', tokenize = 'trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS cards_gram_fts USING fts5 (
  card_grams,
  content = 'fact_cards', content_rowid = 'fact_id', tokenize = 'unicode61'
);
