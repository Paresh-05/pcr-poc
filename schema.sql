CREATE TABLE IF NOT EXISTS cases (
    id                BIGSERIAL PRIMARY KEY,
    source_row        INTEGER NOT NULL UNIQUE,
    dtc_code          TEXT NOT NULL,
    model_line        TEXT NOT NULL,
    customer_verbatim TEXT NOT NULL,
    diagnosis         TEXT NOT NULL,
    failed_parts      JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_cases_dtc_model ON cases (dtc_code, model_line);
CREATE INDEX IF NOT EXISTS idx_cases_model ON cases (model_line);

-- Condensed summary of all past cases for one DTC (+ model, or '*' for all models); rebuilt when the case count changes.
CREATE TABLE IF NOT EXISTS case_digests (
    dtc_code    TEXT NOT NULL,
    model_scope TEXT NOT NULL,
    n_cases     INTEGER NOT NULL,
    digest      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dtc_code, model_scope)
);

-- Manual content stored locally (no external service call).
-- manual_model is the manual's own model name; manual_model_map links it to the model_line values used in cases.
CREATE TABLE IF NOT EXISTS manual_model_map (
    manual_model TEXT NOT NULL,
    model_line   TEXT NOT NULL,
    PRIMARY KEY (manual_model, model_line)
);

CREATE TABLE IF NOT EXISTS manual_dtcs (
    id           BIGSERIAL PRIMARY KEY,
    manual_model TEXT NOT NULL,
    dtc_code     TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (manual_model, dtc_code)
);

CREATE TABLE IF NOT EXISTS manual_subsystems (
    id            BIGSERIAL PRIMARY KEY,
    manual_dtc_id BIGINT NOT NULL REFERENCES manual_dtcs (id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    subsystem     TEXT NOT NULL,
    content       TEXT NOT NULL,
    UNIQUE (manual_dtc_id, subsystem)
);

CREATE INDEX IF NOT EXISTS idx_manual_dtcs_code ON manual_dtcs (dtc_code);

-- Guided diagnosis (yes/no questions). `state` holds candidates, pending question, history and the conclusion.
CREATE TABLE IF NOT EXISTS diagnoses (
    id            UUID PRIMARY KEY,
    dtc_code      TEXT NOT NULL,
    model_line    TEXT NOT NULL,
    scope         TEXT NOT NULL,
    manual_status TEXT NOT NULL,
    case_file     TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('asking', 'concluded', 'no_evidence')),
    state         JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS diagnosis_steps (
    id           BIGSERIAL PRIMARY KEY,
    diagnosis_id UUID NOT NULL REFERENCES diagnoses (id) ON DELETE CASCADE,
    step_no      INTEGER NOT NULL,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL CHECK (answer IN ('yes', 'no', 'not_sure')),
    note         TEXT,
    candidates   JSONB NOT NULL,  -- ranked candidate causes after this answer
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (diagnosis_id, step_no)
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

-- Cost tracking. Prices are per million tokens; reasoning tokens are billed as output tokens.
CREATE TABLE IF NOT EXISTS model_prices (
    model                 TEXT PRIMARY KEY,
    input_per_mtok        NUMERIC(12, 6) NOT NULL,
    cached_input_per_mtok NUMERIC(12, 6) NOT NULL,
    output_per_mtok       NUMERIC(12, 6) NOT NULL,
    source                TEXT NOT NULL,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per model call. input_tokens includes cached_input_tokens; output_tokens includes reasoning_tokens.
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
    cost_usd            NUMERIC(14, 8),  -- NULL when the model has no price row
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls (run_id);

-- One row per run (a run = one diagnosis, including any case digest it had to build).
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

-- List prices from OpenAI's published rates as listed in OpenRouter's public model catalog on 2026-10-13.
INSERT INTO model_prices (model, input_per_mtok, cached_input_per_mtok, output_per_mtok, source) VALUES
    ('gpt-6-luna', 0.10, 0.01, 0.50, 'OpenRouter catalog openai/gpt-6-luna, 2026-10-13 (rates for prompts under 272k tokens)'),
    ('gpt-6-sol',  2.00, 0.20, 10.00, 'OpenRouter catalog openai/gpt-6-sol, 2026-10-13 (rates for prompts under 272k tokens)')
ON CONFLICT (model) DO NOTHING;

-- Knowledge pack: what is known about one fault code, built once from the manual and the past repairs (see app/pack.py)
CREATE TABLE IF NOT EXISTS knowledge_packs (
    dtc_code      TEXT NOT NULL,
    model_line    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    source_counts JSONB NOT NULL,
    checks        JSONB NOT NULL,
    pack          JSONB NOT NULL,
    built_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (dtc_code, model_line)
);

ALTER TABLE diagnosis_steps ADD COLUMN IF NOT EXISTS explanation TEXT;
ALTER TABLE knowledge_packs ADD COLUMN IF NOT EXISTS build_cost_usd NUMERIC(12, 6);
