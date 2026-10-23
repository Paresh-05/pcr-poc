-- New schema: embedding-based retrieval over past complaints (PCR) + DTC manuals.
CREATE EXTENSION IF NOT EXISTS vector;

-- CONTENT STORE: one row per past complaint ticket. No embedding here - never searched directly.
CREATE TABLE IF NOT EXISTS pcr_cases (
    id                BIGSERIAL PRIMARY KEY,
    ticket_no         TEXT NOT NULL UNIQUE,
    model_line        TEXT NOT NULL,
    model_variant     TEXT,
    fuel              TEXT,
    dtc_code          TEXT NOT NULL,
    customer_verbatim TEXT NOT NULL,
    comments_raw      TEXT NOT NULL,       -- full conversation thread, kept for citation/display only
    conclusion_text   TEXT NOT NULL,       -- last substantive turn in comments_raw (the actual resolution)
    failed_parts      JSONB NOT NULL DEFAULT '[]'::jsonb,  -- normalized: [{"part_code": "...", "part_name": "..."}]
    conclusion_matches_parts BOOLEAN,      -- data-quality flag: does conclusion_text agree with failed_parts?
    resolution_status TEXT NOT NULL DEFAULT 'resolved',
    updated_date      DATE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pcr_dtc_model ON pcr_cases (dtc_code, model_line);
CREATE INDEX IF NOT EXISTS idx_pcr_model ON pcr_cases (model_line);

-- SEARCH INDEX: summary + likely questions + embedding, same split as manual_chunk_summaries.
CREATE TABLE IF NOT EXISTS pcr_case_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    pcr_case_id         BIGINT NOT NULL REFERENCES pcr_cases (id) ON DELETE CASCADE,
    summary_for_search  TEXT NOT NULL,
    embedding           vector(1536),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pcr_summary_case ON pcr_case_summaries (pcr_case_id);
CREATE INDEX IF NOT EXISTS idx_pcr_summary_embedding ON pcr_case_summaries USING hnsw (embedding vector_cosine_ops);

-- CONTENT STORE: one row per manual section (= one TOC entry = one shared procedure; may cover several
-- dtc_codes at once; the same code under different submodules gets separate rows, since that content
-- genuinely differs). No embedding here - this table is never searched directly.
CREATE TABLE IF NOT EXISTS manual_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    dtc_codes           TEXT[] NOT NULL,
    model_line          TEXT NOT NULL,
    submodule           TEXT,              -- system/module this procedure belongs to, from the TOC
    start_page          INTEGER,
    end_page            INTEGER,
    full_markdown       TEXT NOT NULL,     -- the complete verbatim section: facts + procedure + analysis
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- SEARCH INDEX: the summary + likely-questions text and its embedding, kept separate from the content
-- store above. Retrieval searches THIS table by embedding, then joins back to manual_chunks for the
-- full content. One row per manual_chunks row (1:1 today, but kept as its own table so a chunk could later
-- have more than one summary variant without touching the content store).
CREATE TABLE IF NOT EXISTS manual_chunk_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    manual_chunk_id     BIGINT NOT NULL REFERENCES manual_chunks (id) ON DELETE CASCADE,
    summary_for_search  TEXT NOT NULL,     -- summary + likely user questions
    embedding           vector(1536),      -- embedding of summary_for_search
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_manual_dtc_model ON manual_chunks USING gin (dtc_codes);
CREATE INDEX IF NOT EXISTS idx_manual_model ON manual_chunks (model_line);
CREATE INDEX IF NOT EXISTS idx_manual_embedding ON manual_chunks USING hnsw (embedding vector_cosine_ops);

-- Cached diagnosis tree for one (dtc_code, model_line [, symptom_hash]). Rebuilt only when evidence changes.
CREATE TABLE IF NOT EXISTS diagnosis_trees (
    id            BIGSERIAL PRIMARY KEY,
    dtc_code      TEXT NOT NULL,
    model_line    TEXT NOT NULL,
    symptom_hash  TEXT NOT NULL DEFAULT '',  -- '' for the catalog (code-only) flow; sha256 of free-text query otherwise
    tree          JSONB NOT NULL,            -- {"root": <node_id>, "nodes": {...}}
    evidence      JSONB NOT NULL,            -- what was retrieved: counts, source ids, used to render "evidence" chips
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (dtc_code, model_line, symptom_hash)
);

-- One guided-diagnosis run: the whole tree is sent to the client in one response and walked there
-- (no per-step server round-trip, no resume) - this row just anchors the outcome/feedback step.
CREATE TABLE IF NOT EXISTS diagnoses (
    id            UUID PRIMARY KEY,
    dtc_code      TEXT NOT NULL,
    model_line    TEXT NOT NULL,
    tree_id       BIGINT REFERENCES diagnosis_trees (id),
    status        TEXT NOT NULL CHECK (status IN ('asking', 'concluded', 'no_evidence')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- What actually happened after the repair (reported by the mechanic); one row per diagnosis.
CREATE TABLE IF NOT EXISTS diagnosis_outcomes (
    diagnosis_id   UUID PRIMARY KEY REFERENCES diagnoses (id) ON DELETE CASCADE,
    result         TEXT NOT NULL CHECK (result IN ('worked', 'partly', 'not_worked')),
    actual_cause   TEXT,
    parts_replaced JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes          TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Cost tracking (unchanged from the old system). Prices are per million tokens.
CREATE TABLE IF NOT EXISTS model_prices (
    model                 TEXT PRIMARY KEY,
    input_per_mtok        NUMERIC(12, 6) NOT NULL,
    cached_input_per_mtok NUMERIC(12, 6) NOT NULL,
    output_per_mtok       NUMERIC(12, 6) NOT NULL,
    source                TEXT NOT NULL,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              UUID,
    run_type            TEXT NOT NULL,
    purpose             TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL,
    reasoning_tokens    INTEGER NOT NULL DEFAULT 0,
    total_tokens        INTEGER NOT NULL,
    latency_ms          INTEGER,
    cost_usd            NUMERIC(14, 8),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id);

CREATE TABLE IF NOT EXISTS run_costs (
    run_id              UUID PRIMARY KEY,
    run_type            TEXT NOT NULL,
    dtc_code            TEXT,
    model_line          TEXT,
    calls               INTEGER NOT NULL DEFAULT 0,
    unpriced_calls      INTEGER NOT NULL DEFAULT 0,
    input_tokens        BIGINT NOT NULL DEFAULT 0,
    cached_input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens       BIGINT NOT NULL DEFAULT 0,
    reasoning_tokens    BIGINT NOT NULL DEFAULT 0,
    cost_usd            NUMERIC(14, 8) NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO model_prices (model, input_per_mtok, cached_input_per_mtok, output_per_mtok, source) VALUES
    ('gpt-6-luna', 0.10, 0.01, 0.50, 'OpenRouter catalog openai/gpt-6-luna, 2026-10-13 (rates for prompts under 272k tokens)'),
    ('gpt-6-sol',  2.00, 0.20, 10.00, 'OpenRouter catalog openai/gpt-6-sol, 2026-10-13 (rates for prompts under 272k tokens)'),
    ('text-embedding-3-small', 0.02, 0.02, 0.00, 'OpenAI published rate, 2026-10-13')
ON CONFLICT (model) DO NOTHING;
