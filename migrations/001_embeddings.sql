-- Idempotent: adds the embedding column to databases created before it existed.
-- The vector size (1536 here) is a default; scripts/embed.py resizes it to match the chosen model.
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE cases ADD COLUMN IF NOT EXISTS complaint_embedding vector(1536);
