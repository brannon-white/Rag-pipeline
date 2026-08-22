-- 0001_initial: studies, chunks, and the hybrid-retrieval indexes.
--
-- One Postgres holds the structured metadata, the dense vectors and the
-- lexical index. That is the central architectural bet (ADR-001): because both
-- retrieval arms and the metadata filter live in the same query planner, a
-- selective filter is applied *before* the ANN search rather than after it.
-- A separate vector store can only post-filter, which means a query like
-- "Phase 3 recruiting trials for X" either over-fetches wildly or silently
-- returns fewer than k results.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- studies: the structured half of a protocol record.
-- Every column here is both a retrieval filter and eval ground truth.
-- ---------------------------------------------------------------------------
CREATE TABLE studies (
    nct_id              text PRIMARY KEY,
    brief_title         text        NOT NULL,
    official_title      text,

    overall_status      text        NOT NULL DEFAULT 'UNKNOWN',
    study_type          text        NOT NULL DEFAULT 'UNKNOWN',
    phases              text[]      NOT NULL DEFAULT '{}',

    conditions          text[]      NOT NULL DEFAULT '{}',
    keywords            text[]      NOT NULL DEFAULT '{}',
    interventions       jsonb       NOT NULL DEFAULT '[]',

    lead_sponsor        text,
    sponsor_class       text,
    enrollment          integer,
    enrollment_type     text,

    -- Ages are stored as numeric years, normalised at parse time from the
    -- registry's "12 Years" / "6 Months" strings. Comparing those as text
    -- makes "9 Years" sort above "12 Years".
    min_age_years       numeric(6, 3),
    max_age_years       numeric(6, 3),
    sex                 text        NOT NULL DEFAULT 'ALL',
    healthy_volunteers  boolean,
    std_ages            text[]      NOT NULL DEFAULT '{}',

    allocation          text,
    intervention_model  text,
    primary_purpose     text,
    masking             text,

    start_date          date,
    completion_date     date,
    last_update_posted  date,

    countries           text[]      NOT NULL DEFAULT '{}',
    location_count      integer     NOT NULL DEFAULT 0,
    has_results         boolean     NOT NULL DEFAULT false,

    -- Content hash of the raw payload; drives the incremental-skip check.
    source_hash         text        NOT NULL DEFAULT '',
    raw_s3_key          text,
    ingested_at         timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT studies_nct_id_format CHECK (nct_id ~ '^NCT[0-9]{8}$'),
    CONSTRAINT studies_age_order CHECK (
        min_age_years IS NULL OR max_age_years IS NULL OR min_age_years <= max_age_years
    )
);

-- ---------------------------------------------------------------------------
-- chunks: the retrievable half.
-- ---------------------------------------------------------------------------
CREATE TABLE chunks (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    nct_id          text        NOT NULL REFERENCES studies (nct_id) ON DELETE CASCADE,
    section         text        NOT NULL,
    ordinal         integer     NOT NULL,
    label           text,

    content         text        NOT NULL,
    -- Deterministic contextual-retrieval prefix, synthesised from the study's
    -- structured fields. Anthropic's technique normally costs an LLM call per
    -- chunk; the registry's metadata gives us the same signal for free.
    context_header  text        NOT NULL DEFAULT '',
    token_count     integer     NOT NULL DEFAULT 0,

    -- halfvec (float16) rather than vector (float32): half the storage and
    -- half the index size, for a recall delta we measure in the dimension
    -- ablation rather than assume. Dimension is pinned by migration, so a
    -- config change to embed_dim requires a migration -- deliberately, since
    -- mixing dimensions in one column is unrecoverable.
    embedding       halfvec(512),

    -- Generated, not application-maintained: a tsvector that can drift from
    -- its own content is a silent lexical-recall hole. Both retrieval arms
    -- must see exactly the same text.
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(context_header, '') || ' ' || content)
    ) STORED,

    content_hash    text        NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),

    -- Stable natural key: re-ingesting a study upserts its chunks in place
    -- instead of duplicating them.
    CONSTRAINT chunks_natural_key UNIQUE (nct_id, ordinal)
);

-- --- Dense arm -------------------------------------------------------------
-- HNSW over cosine distance. m=16/ef_construction=64 are pgvector's defaults
-- and a reasonable build-time/recall balance at this corpus size; ef_search is
-- set per-query at runtime so it can be swept in the ablation harness.
CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- --- Lexical arm -----------------------------------------------------------
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);

-- --- Joins and filters -----------------------------------------------------
CREATE INDEX chunks_nct_id ON chunks (nct_id);
CREATE INDEX chunks_section ON chunks (section);

CREATE INDEX studies_conditions_gin ON studies USING gin (conditions);
CREATE INDEX studies_keywords_gin ON studies USING gin (keywords);
CREATE INDEX studies_phases_gin ON studies USING gin (phases);
-- Composite ordering matches the most common filter shape: status first
-- (highly selective, users overwhelmingly want recruiting trials), then type.
CREATE INDEX studies_status_type ON studies (overall_status, study_type);
CREATE INDEX studies_age_range ON studies (min_age_years, max_age_years);

-- ---------------------------------------------------------------------------
-- query_log: per-request cost and latency attribution.
-- This table is the /stats dashboard and the load-test evidence. Writing cost
-- per request from day one is what makes "cost per 1,000 queries" a measured
-- number rather than an estimate reconstructed from a monthly bill.
-- ---------------------------------------------------------------------------
CREATE TABLE query_log (
    id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts                  timestamptz NOT NULL DEFAULT now(),
    query_text          text        NOT NULL,
    filters             jsonb       NOT NULL DEFAULT '{}',
    retrieved_chunk_ids bigint[]    NOT NULL DEFAULT '{}',
    -- Per-stage: {"parse": 120, "retrieve": 45, "rerank": 90, "generate": 2100}
    latency_ms          jsonb       NOT NULL DEFAULT '{}',
    tokens              jsonb       NOT NULL DEFAULT '{}',
    cost_usd            numeric(12, 8) NOT NULL DEFAULT 0,
    model               text,
    abstained           boolean     NOT NULL DEFAULT false,
    trace_id            text,
    -- Hashed, never raw: enforcing a per-IP daily quota does not require
    -- retaining an identifier for the address itself.
    client_hash         text
);

CREATE INDEX query_log_ts ON query_log (ts DESC);
CREATE INDEX query_log_client_ts ON query_log (client_hash, ts DESC);

CREATE TABLE feedback (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    query_log_id bigint      NOT NULL REFERENCES query_log (id) ON DELETE CASCADE,
    rating       smallint    NOT NULL,
    comment      text,
    ts           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT feedback_rating_range CHECK (rating IN (-1, 1))
);

CREATE TABLE eval_runs (
    id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    git_sha  text        NOT NULL,
    config   jsonb       NOT NULL DEFAULT '{}',
    metrics  jsonb       NOT NULL DEFAULT '{}',
    ts       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX eval_runs_ts ON eval_runs (ts DESC);
