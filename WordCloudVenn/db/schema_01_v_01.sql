-- db/schema_01_v_01.sql
-- Postgres schema for:
--  - words: unique word list + frequency per hard-coded source columns
--  - titles: one row per scraped URL/item
--  - title_words: presence mapping (unique words per title)
--
-- Uses IDENTITY columns.
-- run only once.

BEGIN;

CREATE TABLE IF NOT EXISTS words (
  id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  word         TEXT NOT NULL UNIQUE,

  -- Hard-coded per-source document frequency:
  -- number of titles from that source that contain this word
  freq_omega INTEGER NOT NULL DEFAULT 0 CHECK (freq_omega >= 0),
  freq_cn INTEGER NOT NULL DEFAULT 0 CHECK (freq_cn >= 0),
  freq_guardian INTEGER NOT NULL DEFAULT 0 CHECK (freq_guardian >= 0)
);

CREATE TABLE IF NOT EXISTS titles (
  id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source       TEXT NOT NULL CHECK (source IN ('OMEGA', 'CN', 'GUARDIAN')),
  url          TEXT NOT NULL UNIQUE,
  fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS title_words (
  title_id BIGINT NOT NULL REFERENCES titles(id) ON DELETE CASCADE,
  word_id  BIGINT NOT NULL REFERENCES words(id)  ON DELETE CASCADE,

  -- Presence-only: a word can appear at most once per title
  PRIMARY KEY (title_id, word_id)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_titles_source ON titles(source);
CREATE INDEX IF NOT EXISTS idx_title_words_word_id ON title_words(word_id);

COMMIT;
