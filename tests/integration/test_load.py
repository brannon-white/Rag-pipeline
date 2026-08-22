"""Load-layer tests against a real Postgres.

These exist because several failure modes here are invisible to unit tests
with a fake connection: whether asyncpg can actually bind a ``jsonb`` column
from a plain Python string, whether the halfvec array workaround
(``text[]::halfvec(512)[]``, see ``load._vector_literal``) round-trips, and
whether the ``ON CONFLICT ... WHERE`` guard produces the write counts the code
assumes. Every one of those was checked by hand against the compose Postgres
before being written into this file -- these tests pin that behaviour so a
pgvector or asyncpg upgrade cannot silently break it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from trialrag.db.pool import Database
from trialrag.domain.models import ChunkCandidate, SectionKind, Study
from trialrag.ingest.chunk import DEFAULT_CONFIG, chunk_study
from trialrag.ingest.embed import content_hash
from trialrag.ingest.load import LoadStats, upsert_chunks, upsert_studies, write_manifest
from trialrag.ingest.parse import iter_sections, parse_study
from trialrag.ingest.tokens import HeuristicTokenCounter

pytestmark = pytest.mark.integration

MODEL, DIM = "voyage-4-lite", 512


def _study(nct_id: str = "NCT00000001", title: str = "A study") -> Study:
    return Study(nct_id=nct_id, brief_title=title)


def _chunk(nct_id: str, ordinal: int, content: str | None = None) -> ChunkCandidate:
    # Default content is ordinal-dependent on purpose: two chunks with
    # identical text hash identically (content_hash covers text, model and
    # dim -- not ordinal), which silently aliases them in the embeddings map.
    # An explicit `content` is still available for tests that want that.
    return ChunkCandidate(
        nct_id=nct_id,
        kind=SectionKind.ELIGIBILITY_INCLUSION,
        ordinal=ordinal,
        content=content if content is not None else f"criterion number {ordinal}",
        context_header="hdr",
        token_count=4,
    )


def _embeddings_for(chunks: list[ChunkCandidate]) -> dict[str, list[float]]:
    return {
        content_hash(chunk.embedding_input, model=MODEL, dim=DIM): [float(ordinal)] * DIM
        for ordinal, chunk in enumerate(chunks)
    }


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------


async def test_upsert_study_round_trips_all_field_kinds(db: Database) -> None:
    """Exercises every distinct column type in one row: arrays, jsonb, numeric
    ages, an enum-as-text, dates and booleans."""
    study = Study(
        nct_id="NCT00000001",
        brief_title="A study of things",
        overall_status="RECRUITING",  # type: ignore[arg-type]
        study_type="INTERVENTIONAL",  # type: ignore[arg-type]
        phases=("PHASE2", "PHASE3"),
        conditions=("Type 2 Diabetes", "Obesity"),
        interventions=(),
        lead_sponsor="Example Sponsor",
        enrollment=42,
        healthy_volunteers=True,
        countries=("United States", "Canada"),
        location_count=3,
        source_hash="abc123",
    )
    written = await upsert_studies(db, [study])
    assert written == 1

    row = await db.fetchrow("SELECT * FROM studies WHERE nct_id = $1", "NCT00000001")
    assert row is not None
    assert row["phases"] == ["PHASE2", "PHASE3"]
    assert row["conditions"] == ["Type 2 Diabetes", "Obesity"]
    assert row["countries"] == ["United States", "Canada"]
    assert row["healthy_volunteers"] is True
    assert row["enrollment"] == 42
    # jsonb round-trip: a plain Python str passed for a jsonb column binds
    # correctly without an explicit ::jsonb cast in the query text.
    assert json.loads(row["interventions"]) == []


async def test_upsert_study_is_idempotent_on_unchanged_hash(db: Database) -> None:
    study = _study()
    study = study.model_copy(update={"source_hash": "same-hash"})
    assert await upsert_studies(db, [study]) == 1
    assert await upsert_studies(db, [study]) == 0  # unchanged -> no-op write count


async def test_upsert_study_writes_again_when_hash_changes(db: Database) -> None:
    study = _study().model_copy(update={"source_hash": "v1"})
    await upsert_studies(db, [study])
    updated = study.model_copy(update={"brief_title": "Revised title", "source_hash": "v2"})
    assert await upsert_studies(db, [updated]) == 1

    row = await db.fetchrow("SELECT brief_title FROM studies WHERE nct_id = $1", study.nct_id)
    assert row is not None
    assert row["brief_title"] == "Revised title"


async def test_upsert_studies_empty_list_is_a_noop(db: Database) -> None:
    assert await upsert_studies(db, []) == 0


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


async def test_upsert_chunks_round_trips_vector_and_is_hnsw_searchable(db: Database) -> None:
    """The load-bearing check: the ``text[]::halfvec(512)[]`` workaround must
    both write correctly and remain usable by the ANN index."""
    study = _study()
    await upsert_studies(db, [study])
    chunks = [_chunk(study.nct_id, 0, "alpha criterion"), _chunk(study.nct_id, 1, "beta criterion")]
    embeddings = _embeddings_for(chunks)

    stats = await upsert_chunks(db, study.nct_id, chunks, embeddings, model=MODEL, dim=DIM)
    assert stats.chunks_written == 2
    assert stats.embeddings_missing == []

    # Once bound to a `halfvec` parameter position (rather than `text[]`, as
    # load.py's insert uses), the pool's registered pgvector codec takes over
    # and expects a plain Python list/ndarray, not the text-literal form.
    query_vector = [0.0] * DIM
    rows = await db.fetch(
        f"SELECT ordinal, embedding <=> $1::halfvec({DIM}) AS dist "
        "FROM chunks WHERE nct_id = $2 ORDER BY dist",
        query_vector,
        study.nct_id,
    )
    assert [r["ordinal"] for r in rows] == [0, 1]  # ordinal 0 embedded as all-zeros: closest


async def test_chunks_without_an_embedding_are_skipped_not_written_blank(db: Database) -> None:
    """A chunk missing from the embeddings map must not become a zero-vector
    row -- that would rank nowhere in dense search while still appearing in
    lexical search, a silently half-indexed state."""
    study = _study()
    await upsert_studies(db, [study])
    chunks = [_chunk(study.nct_id, 0), _chunk(study.nct_id, 1)]
    embeddings = _embeddings_for(chunks[:1])  # only ordinal 0 has a vector

    stats = await upsert_chunks(db, study.nct_id, chunks, embeddings, model=MODEL, dim=DIM)
    assert stats.chunks_written == 1
    assert stats.embeddings_missing == [f"{study.nct_id}:1"]

    count = await db.fetchval("SELECT count(*) FROM chunks WHERE nct_id = $1", study.nct_id)
    assert count == 1


async def test_re_ingest_deletes_ordinals_past_the_new_tail(db: Database) -> None:
    """A shorter re-chunk (e.g. a revised chunking config) must retract the
    stale tail, or dead chunks keep surfacing in results forever."""
    study = _study()
    await upsert_studies(db, [study])
    first = [_chunk(study.nct_id, i) for i in range(4)]
    await upsert_chunks(db, study.nct_id, first, _embeddings_for(first), model=MODEL, dim=DIM)
    assert await db.fetchval("SELECT count(*) FROM chunks WHERE nct_id=$1", study.nct_id) == 4

    shorter = [_chunk(study.nct_id, i) for i in range(2)]
    stats = await upsert_chunks(
        db, study.nct_id, shorter, _embeddings_for(shorter), model=MODEL, dim=DIM
    )
    assert stats.chunks_deleted == 2
    remaining = await db.fetch(
        "SELECT ordinal FROM chunks WHERE nct_id=$1 ORDER BY ordinal", study.nct_id
    )
    assert [r["ordinal"] for r in remaining] == [0, 1]


async def test_empty_chunk_list_deletes_all_of_a_studys_chunks(db: Database) -> None:
    study = _study()
    await upsert_studies(db, [study])
    existing = [_chunk(study.nct_id, 0)]
    await upsert_chunks(db, study.nct_id, existing, _embeddings_for(existing), model=MODEL, dim=DIM)

    stats = await upsert_chunks(db, study.nct_id, [], {}, model=MODEL, dim=DIM)
    assert stats.chunks_deleted == 1
    assert await db.fetchval("SELECT count(*) FROM chunks WHERE nct_id=$1", study.nct_id) == 0


async def test_deleting_a_study_cascades_to_its_chunks(db: Database) -> None:
    study = _study()
    await upsert_studies(db, [study])
    chunks = [_chunk(study.nct_id, 0)]
    await upsert_chunks(db, study.nct_id, chunks, _embeddings_for(chunks), model=MODEL, dim=DIM)

    await db.execute("DELETE FROM studies WHERE nct_id = $1", study.nct_id)
    assert await db.fetchval("SELECT count(*) FROM chunks WHERE nct_id=$1", study.nct_id) == 0


async def test_re_upserting_identical_chunks_is_a_noop_write(db: Database) -> None:
    study = _study()
    await upsert_studies(db, [study])
    chunks = [_chunk(study.nct_id, 0)]
    embeddings = _embeddings_for(chunks)
    first = await upsert_chunks(db, study.nct_id, chunks, embeddings, model=MODEL, dim=DIM)
    second = await upsert_chunks(db, study.nct_id, chunks, embeddings, model=MODEL, dim=DIM)
    assert first.chunks_written == 1
    assert second.chunks_written == 0  # content_hash unchanged -> WHERE guard skips it


# ---------------------------------------------------------------------------
# Full pipeline, real corpus fixture
# ---------------------------------------------------------------------------


async def test_full_pipeline_on_a_real_study_record(
    db: Database, bounded_age_study: dict[str, Any]
) -> None:
    """fetch fixture -> parse -> chunk -> (fake) embed -> load, end to end."""
    study = parse_study(bounded_age_study)
    counter = HeuristicTokenCounter()
    chunks = chunk_study(iter_sections(bounded_age_study, study), study, counter, DEFAULT_CONFIG)
    embeddings = {
        content_hash(c.embedding_input, model=MODEL, dim=DIM): [0.1 * (i + 1)] * DIM
        for i, c in enumerate(chunks)
    }

    assert await upsert_studies(db, [study]) == 1
    stats: LoadStats = await upsert_chunks(
        db, study.nct_id, chunks, embeddings, model=MODEL, dim=DIM
    )
    assert stats.chunks_written == len(chunks)
    assert stats.embeddings_missing == []

    stored = await db.fetchval("SELECT count(*) FROM chunks WHERE nct_id = $1", study.nct_id)
    assert stored == len(chunks)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


async def test_manifest_reflects_loaded_corpus(db: Database, tmp_path: Any) -> None:
    study = Study(
        nct_id="NCT00000001",
        brief_title="T",
        overall_status="RECRUITING",  # type: ignore[arg-type]
        phases=("PHASE3",),
        conditions=("Diabetes",),
        location_count=2,
    )
    await upsert_studies(db, [study])
    chunks = [_chunk(study.nct_id, 0), _chunk(study.nct_id, 1)]
    await upsert_chunks(db, study.nct_id, chunks, _embeddings_for(chunks), model=MODEL, dim=DIM)

    manifest_path = tmp_path / "corpus_manifest.json"
    manifest = await write_manifest(db, manifest_path)

    assert manifest["study_count"] == 1
    assert manifest["chunk_count"] == 2
    assert manifest["studies_by_phase"] == {"PHASE3": 1}
    assert manifest["top_conditions"] == {"Diabetes": 1}
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest
