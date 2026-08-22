"""Persist parsed studies and embedded chunks to Postgres.

Two upsert shapes, chosen for different reasons:

* **Studies** upsert one row at a time inside a batch transaction. There are
  thousands of them, not tens of thousands, and each carries enough distinct
  array/jsonb columns that a hand-rolled multi-row ``VALUES`` list would be
  harder to read than it saves.
* **Chunks** upsert via ``UNNEST``-based bulk statements, because there are an
  order of magnitude more of them and each carries a 512-d vector. Sending them
  as one multi-row statement per batch is the difference between a five-minute
  load and a fifty-minute one.

Both are keyed on natural keys (studies by ``nct_id``, chunks by
``(nct_id, ordinal)``) so re-running ingest after a crash, a chunking-strategy
change, or an incremental sync converges to the same state rather than
duplicating rows.

Deleting stale chunks matters as much as inserting new ones: if a study's
chunk count shrinks (shorter protocol amendment, revised chunking config), the
old tail of ordinals must be removed or dead chunks keep surfacing in results.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from trialrag.db.pool import Database
from trialrag.domain.models import Chunk, ChunkCandidate, Study
from trialrag.ingest.embed import Vector, content_hash

# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

_UPSERT_STUDY = """
INSERT INTO studies (
    nct_id, brief_title, official_title, overall_status, study_type, phases,
    conditions, keywords, interventions, lead_sponsor, sponsor_class,
    enrollment, enrollment_type, min_age_years, max_age_years, sex,
    healthy_volunteers, std_ages, allocation, intervention_model,
    primary_purpose, masking, start_date, completion_date, last_update_posted,
    countries, location_count, has_results, source_hash, raw_s3_key, ingested_at
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16,
    $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, now()
)
ON CONFLICT (nct_id) DO UPDATE SET
    brief_title = EXCLUDED.brief_title,
    official_title = EXCLUDED.official_title,
    overall_status = EXCLUDED.overall_status,
    study_type = EXCLUDED.study_type,
    phases = EXCLUDED.phases,
    conditions = EXCLUDED.conditions,
    keywords = EXCLUDED.keywords,
    interventions = EXCLUDED.interventions,
    lead_sponsor = EXCLUDED.lead_sponsor,
    sponsor_class = EXCLUDED.sponsor_class,
    enrollment = EXCLUDED.enrollment,
    enrollment_type = EXCLUDED.enrollment_type,
    min_age_years = EXCLUDED.min_age_years,
    max_age_years = EXCLUDED.max_age_years,
    sex = EXCLUDED.sex,
    healthy_volunteers = EXCLUDED.healthy_volunteers,
    std_ages = EXCLUDED.std_ages,
    allocation = EXCLUDED.allocation,
    intervention_model = EXCLUDED.intervention_model,
    primary_purpose = EXCLUDED.primary_purpose,
    masking = EXCLUDED.masking,
    start_date = EXCLUDED.start_date,
    completion_date = EXCLUDED.completion_date,
    last_update_posted = EXCLUDED.last_update_posted,
    countries = EXCLUDED.countries,
    location_count = EXCLUDED.location_count,
    has_results = EXCLUDED.has_results,
    source_hash = EXCLUDED.source_hash,
    raw_s3_key = EXCLUDED.raw_s3_key,
    ingested_at = now()
WHERE studies.source_hash IS DISTINCT FROM EXCLUDED.source_hash
"""


def _study_params(study: Study) -> tuple[object, ...]:
    return (
        study.nct_id,
        study.brief_title,
        study.official_title,
        study.overall_status.value,
        study.study_type.value,
        list(study.phases),
        list(study.conditions),
        list(study.keywords),
        json.dumps([i.model_dump() for i in study.interventions]),
        study.lead_sponsor,
        study.sponsor_class,
        study.enrollment,
        study.enrollment_type,
        study.age_range.min_years,
        study.age_range.max_years,
        study.sex.value,
        study.healthy_volunteers,
        list(study.std_ages),
        study.allocation,
        study.intervention_model,
        study.primary_purpose,
        study.masking,
        study.start_date,
        study.completion_date,
        study.last_update_posted,
        list(study.countries),
        study.location_count,
        study.has_results,
        study.source_hash,
        study.raw_s3_key,
    )


async def upsert_studies(db: Database, studies: Sequence[Study]) -> int:
    """Upsert studies, skipping any whose ``source_hash`` is unchanged.

    Returns the number of rows actually written (the ``WHERE ... IS DISTINCT
    FROM`` clause makes an unchanged row's upsert a no-op, but asyncpg still
    reports it as touched, so the count comes from a follow-up check rather
    than the executemany return value).
    """
    if not studies:
        return 0
    async with db.transaction() as conn:
        hashes_before = {
            row["nct_id"]: row["source_hash"]
            for row in await conn.fetch(
                "SELECT nct_id, source_hash FROM studies WHERE nct_id = ANY($1)",
                [s.nct_id for s in studies],
            )
        }
        await conn.executemany(_UPSERT_STUDY, [_study_params(s) for s in studies])
    # A study absent from hashes_before is a new insert; one present with a
    # different hash is a real update. Either way the WHERE clause in
    # _UPSERT_STUDY only actually wrote a row when the hash differs (or is new).
    return sum(1 for s in studies if hashes_before.get(s.nct_id) != s.source_hash)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

_UPSERT_CHUNKS = """
INSERT INTO chunks (nct_id, section, ordinal, label, content, context_header,
                     token_count, embedding, content_hash)
SELECT * FROM UNNEST(
    $1::text[], $2::text[], $3::int[], $4::text[], $5::text[], $6::text[],
    $7::int[], $8::text[]::halfvec(512)[], $9::text[]
)
ON CONFLICT (nct_id, ordinal) DO UPDATE SET
    section = EXCLUDED.section,
    label = EXCLUDED.label,
    content = EXCLUDED.content,
    context_header = EXCLUDED.context_header,
    token_count = EXCLUDED.token_count,
    embedding = EXCLUDED.embedding,
    content_hash = EXCLUDED.content_hash
WHERE chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
"""


def _vector_literal(vector: Vector) -> str:
    """Render a vector as pgvector's ``"[0.1,0.2,...]"`` input syntax.

    asyncpg's pgvector codec (:func:`pgvector.asyncpg.register_vector`) only
    registers the *scalar* ``halfvec`` type -- it does not compose over arrays,
    so passing a Python list of vectors as a ``halfvec(512)[]`` bind parameter
    fails with "expected list or ndarray" (asyncpg tries to encode the whole
    array as one scalar). The workaround, verified against a live pgvector
    0.8.1 instance: send each vector as its text literal inside a plain
    ``text[]`` parameter, and cast server-side (``$n::text[]::halfvec(512)[]``).
    Casting a scalar type through its own text input function is exactly what
    ``UNNEST(...)::type[]`` triggers, and it round-trips correctly (verified:
    HNSW index scan still applies, cosine distance against the original vector
    comes back as 0.0 up to float16 rounding).
    """
    return "[" + ",".join(repr(component) for component in vector) + "]"


_DELETE_STALE_ORDINALS = """
DELETE FROM chunks WHERE nct_id = $1 AND ordinal >= $2
"""


@dataclass
class LoadStats:
    studies_written: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    embeddings_missing: list[str] = field(default_factory=list)


async def upsert_chunks(
    db: Database,
    nct_id: str,
    chunks: Sequence[ChunkCandidate],
    embeddings: dict[str, Vector],
    *,
    model: str,
    dim: int,
) -> LoadStats:
    """Upsert one study's chunks and delete any ordinal past the new tail.

    Args:
        embeddings: ``content_hash -> vector``, as returned by
            :meth:`trialrag.ingest.embed.Embedder.embed_chunks`, covering *at
            least* the chunks being written here (it may also hold entries for
            other studies in the same ingest batch).
    """
    stats = LoadStats()
    if not chunks:
        async with db.transaction() as conn:
            deleted = await conn.execute("DELETE FROM chunks WHERE nct_id = $1", nct_id)
        stats.chunks_deleted = _count_from_tag(deleted)
        return stats

    rows: list[tuple[object, ...]] = []
    for chunk in chunks:
        digest = content_hash(chunk.embedding_input, model=model, dim=dim)
        vector = embeddings.get(digest)
        if vector is None:
            # A chunk with no embedding must not silently vanish from the
            # corpus with an empty vector -- that would make it unrankable in
            # dense search while still surfacing in lexical search, which is a
            # confusing half-indexed state. Skip it and report it.
            stats.embeddings_missing.append(f"{nct_id}:{chunk.ordinal}")
            continue
        rows.append(
            (
                chunk.nct_id,
                chunk.kind.value,
                chunk.ordinal,
                chunk.label,
                chunk.content,
                chunk.context_header,
                chunk.token_count,
                _vector_literal(vector),
                digest,
            )
        )

    async with db.transaction() as conn:
        if rows:
            columns = list(zip(*rows, strict=True))
            tag = await conn.execute(_UPSERT_CHUNKS, *columns)
            # The command tag counts only rows the WHERE guard actually wrote
            # (verified against a live Postgres: an unchanged content_hash
            # yields "INSERT 0 0"), so this is real write volume, not just the
            # candidate count.
            stats.chunks_written = _count_from_tag(tag)

        max_ordinal = max((c.ordinal for c in chunks), default=-1)
        deleted = await conn.execute(_DELETE_STALE_ORDINALS, nct_id, max_ordinal + 1)
        stats.chunks_deleted = _count_from_tag(deleted)

    return stats


def _count_from_tag(tag: str) -> int:
    """Parse asyncpg's ``"DELETE 3"`` command-completion tag into an int."""
    parts = tag.rsplit(" ", 1)
    return int(parts[-1]) if len(parts) == 2 and parts[-1].isdigit() else 0


# ---------------------------------------------------------------------------
# Corpus manifest
# ---------------------------------------------------------------------------


async def write_manifest(db: Database, path: Path) -> dict[str, object]:
    """Snapshot corpus composition to ``path`` as JSON.

    This file is committed to the repo. It is what makes "the corpus is 5,000
    studies across 8 condition areas" a checked-in fact rather than a claim --
    a reviewer, or a future ablation run, can diff it against a later ingest.
    """
    async with db.acquire() as conn:
        totals = await conn.fetchrow(
            "SELECT count(*) AS studies, sum(location_count) AS locations FROM studies"
        )
        chunk_totals = await conn.fetchrow(
            "SELECT count(*) AS chunks, sum(token_count) AS tokens, "
            "avg(token_count) AS avg_tokens FROM chunks"
        )
        by_section = await conn.fetch(
            "SELECT section, count(*) AS n FROM chunks GROUP BY section ORDER BY section"
        )
        by_status = await conn.fetch(
            "SELECT overall_status, count(*) AS n FROM studies "
            "GROUP BY overall_status ORDER BY n DESC"
        )
        by_phase = await conn.fetch(
            "SELECT unnest(phases) AS phase, count(*) AS n FROM studies "
            "GROUP BY phase ORDER BY n DESC"
        )
        top_conditions = await conn.fetch(
            "SELECT unnest(conditions) AS condition, count(*) AS n FROM studies "
            "GROUP BY condition ORDER BY n DESC LIMIT 25"
        )

    manifest = {
        "study_count": totals["studies"] if totals else 0,
        "location_count": int(totals["locations"] or 0) if totals else 0,
        "chunk_count": chunk_totals["chunks"] if chunk_totals else 0,
        "total_tokens": int(chunk_totals["tokens"] or 0) if chunk_totals else 0,
        "avg_chunk_tokens": round(float(chunk_totals["avg_tokens"] or 0), 1)
        if chunk_totals
        else 0.0,
        "chunks_by_section": {row["section"]: row["n"] for row in by_section},
        "studies_by_status": {row["overall_status"]: row["n"] for row in by_status},
        "studies_by_phase": {row["phase"]: row["n"] for row in by_phase},
        "top_conditions": {row["condition"]: row["n"] for row in top_conditions},
    }
    # Manifest writing is a one-shot, end-of-ingest step, not request-serving
    # code -- but the event loop is still shared with the DB pool above, so
    # blocking file I/O runs off-thread rather than stalling it.
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, text, encoding="utf-8")
    return manifest


def chunk_to_domain(candidate: ChunkCandidate, *, chunk_id: int, content_hash_hex: str) -> Chunk:
    """Attach a persisted identity to a chunk candidate.

    Used after a load to hand callers (retrieval tests, the API) a fully
    identified :class:`Chunk` rather than the pre-persistence candidate.
    """
    return Chunk(
        **candidate.model_dump(),
        id=chunk_id,
        content_hash=content_hash_hex,
    )
