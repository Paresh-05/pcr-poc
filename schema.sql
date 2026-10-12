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

CREATE TABLE IF NOT EXISTS conversations (
    id            UUID PRIMARY KEY,
    dtc_code      TEXT NOT NULL,
    model_line    TEXT NOT NULL,
    scope         TEXT NOT NULL,
    manual_status TEXT NOT NULL,
    context       TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    turn            JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id);

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
