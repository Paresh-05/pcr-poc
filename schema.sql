CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cases (
    id                BIGSERIAL PRIMARY KEY,
    source_row        INTEGER NOT NULL UNIQUE,
    dtc_code          TEXT NOT NULL,
    model_line        TEXT NOT NULL,
    customer_verbatim TEXT NOT NULL,
    diagnosis         TEXT NOT NULL,
    failed_parts      JSONB NOT NULL DEFAULT '[]'::jsonb,
    complaint_tsv     TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(customer_verbatim, ''))
    ) STORED,
    complaint_embedding vector(1536)
);

CREATE INDEX IF NOT EXISTS idx_cases_dtc_model ON cases (dtc_code, model_line);
CREATE INDEX IF NOT EXISTS idx_cases_model ON cases (model_line);
CREATE INDEX IF NOT EXISTS idx_cases_complaint_tsv ON cases USING GIN (complaint_tsv);
