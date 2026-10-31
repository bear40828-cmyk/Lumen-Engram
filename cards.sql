-- Lumen Engram · 薄发现层（derived，可随时删了重建）
-- 借的是 Table-GitHub-Capability-Router 的双层结构：
--   先查一张很薄的汇总表拿候选，命中了才去读全文。
-- 这张表不是事实源，schema.sql v0.1 一个字没动。删掉重建即可。

CREATE TABLE IF NOT EXISTS fact_cards (
  fact_id    INTEGER PRIMARY KEY REFERENCES facts (id),
  subject    TEXT NOT NULL,
  card       TEXT NOT NULL,          -- 一行摘要，最多 40 字
  card_grams TEXT NOT NULL,          -- 两字滑窗，补两字词检索
  full_len   INTEGER NOT NULL,       -- 全文多长，用来算展开代价
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
