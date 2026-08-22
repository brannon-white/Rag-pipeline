"""Hybrid retrieval tests against a real Postgres.

Fixture data uses hand-crafted 512-d embeddings (orthonormal-ish basis
vectors) rather than real Voyage output, so dense-arm rankings are exact and
assertable -- these tests need to prove the *SQL* is correct (filter
pre-application, RRF arithmetic, NULL handling across a FULL OUTER JOIN), which
a real embedding model's actual semantics would only obscure. The query itself
was additionally verified live against real embeddings and the real ingested
corpus (documented in service.py's module docstring context); what's here is
what pins that behaviour against regression.
"""

from __future__ import annotations

import pytest

from trialrag.db.pool import Database
from trialrag.domain.models import AgeRange, ChunkCandidate, SectionKind, Study
from trialrag.ingest.embed import content_hash
from trialrag.ingest.load import upsert_chunks, upsert_studies
from trialrag.retrieval.filters import QueryFilters
from trialrag.retrieval.service import RetrievalConfig, hybrid_search

pytestmark = pytest.mark.integration

DIM = 512
MODEL = "voyage-4-lite"


def _basis(i: int) -> list[float]:
    """The i-th standard basis vector in R^512 -- exactly orthogonal to every
    other basis vector used here, so cosine distance is either 0 or 1 and dense
    ranking is unambiguous."""
    v = [0.0] * DIM
    v[i] = 1.0
    return v


async def _seed_study(
    db: Database,
    *,
    min_age_years: float | None = None,
    max_age_years: float | None = None,
    **overrides: object,
) -> Study:
    defaults: dict[str, object] = {
        "nct_id": "NCT00000001",
        "brief_title": "Test study",
        "source_hash": "h1",
        "age_range": AgeRange(min_years=min_age_years, max_years=max_age_years),
    }
    defaults.update(overrides)
    study = Study(**defaults)  # type: ignore[arg-type]
    await upsert_studies(db, [study])
    return study


async def _seed_chunk(
    db: Database, nct_id: str, ordinal: int, content: str, vector: list[float]
) -> None:
    chunk = ChunkCandidate(
        nct_id=nct_id,
        kind=SectionKind.BRIEF_SUMMARY,
        ordinal=ordinal,
        content=content,
        context_header="hdr",
        token_count=len(content.split()),
    )
    digest = content_hash(chunk.embedding_input, model=MODEL, dim=DIM)
    await upsert_chunks(db, nct_id, [chunk], {digest: vector}, model=MODEL, dim=DIM)


# ---------------------------------------------------------------------------
# Dense-arm ranking
# ---------------------------------------------------------------------------


async def test_dense_ranking_follows_cosine_distance(db: Database) -> None:
    await _seed_study(db)
    await _seed_chunk(db, "NCT00000001", 0, "exact match content", _basis(0))
    await _seed_chunk(db, "NCT00000001", 1, "orthogonal content unrelated", _basis(1))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="nonexistent lexical terms zzqx",
        filters=QueryFilters(),
        config=RetrievalConfig(dense_weight=1.0, sparse_k=0),
    )

    assert [r.dense_rank for r in results] == [1, 2]
    assert results[0].content == "exact match content"
    assert results[0].dense_score == pytest.approx(1.0, abs=1e-3)
    assert results[1].dense_score == pytest.approx(0.0, abs=1e-3)


async def test_sparse_k_zero_disables_the_lexical_arm(db: Database) -> None:
    await _seed_study(db)
    await _seed_chunk(db, "NCT00000001", 0, "diabetes management protocol", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="diabetes",
        filters=QueryFilters(),
        config=RetrievalConfig(sparse_k=0),
    )
    assert results[0].sparse_rank is None
    assert results[0].sparse_score is None
    assert results[0].dense_rank == 1


# ---------------------------------------------------------------------------
# RRF fusion arithmetic
# ---------------------------------------------------------------------------


async def test_rrf_score_matches_manual_calculation(db: Database) -> None:
    await _seed_study(db)
    # Chunk A: best dense match, no lexical overlap with the query terms.
    await _seed_chunk(db, "NCT00000001", 0, "unrelated words entirely here", _basis(0))
    # Chunk B: worse dense match, but contains the exact query term.
    await _seed_chunk(db, "NCT00000001", 1, "hypertension treatment outcomes", _basis(1))

    config = RetrievalConfig(dense_weight=0.5, rrf_k=60, sparse_k=10, dense_k=10)
    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="hypertension",
        filters=QueryFilters(),
        config=config,
    )
    by_content = {r.content: r for r in results}

    chunk_a = by_content["unrelated words entirely here"]
    chunk_b = by_content["hypertension treatment outcomes"]

    # A: dense rank 1, no sparse match at all.
    assert chunk_a.dense_rank == 1 and chunk_a.sparse_rank is None
    expected_a = 0.5 * (1.0 / (60 + 1))
    assert chunk_a.rrf_score == pytest.approx(expected_a, rel=1e-6)

    # B: dense rank 2, sparse rank 1 (the only lexical match).
    assert chunk_b.dense_rank == 2 and chunk_b.sparse_rank == 1
    expected_b = 0.5 * (1.0 / (60 + 2)) + 0.5 * (1.0 / (60 + 1))
    assert chunk_b.rrf_score == pytest.approx(expected_b, rel=1e-6)

    # The fused ranking must actually reorder B above A once its lexical
    # match is accounted for -- this is the entire point of hybrid search.
    assert results[0].content == "hypertension treatment outcomes"


async def test_dense_weight_zero_ranks_by_lexical_arm_alone(db: Database) -> None:
    await _seed_study(db)
    # Chunk A is the closer dense match but has no lexical overlap.
    await _seed_chunk(db, "NCT00000001", 0, "completely unrelated filler text", _basis(0))
    await _seed_chunk(db, "NCT00000001", 1, "asthma inhaler corticosteroid", _basis(1))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="asthma inhaler",
        filters=QueryFilters(),
        config=RetrievalConfig(dense_weight=0.0),
    )
    assert results[0].content == "asthma inhaler corticosteroid"


def test_config_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="dense_weight"):
        RetrievalConfig(dense_weight=1.5)


# ---------------------------------------------------------------------------
# Metadata pre-filtering
# ---------------------------------------------------------------------------


async def test_phase_filter_excludes_non_matching_studies(db: Database) -> None:
    await _seed_study(db, nct_id="NCT00000001", phases=("PHASE3",), source_hash="a")
    await _seed_study(db, nct_id="NCT00000002", phases=("PHASE1",), source_hash="b")
    await _seed_chunk(db, "NCT00000001", 0, "phase three content", _basis(0))
    await _seed_chunk(db, "NCT00000002", 0, "phase one content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(phases=["PHASE3"]),
        config=RetrievalConfig(),
    )
    assert {r.nct_id for r in results} == {"NCT00000001"}


async def test_status_filter_excludes_non_matching_studies(db: Database) -> None:
    await _seed_study(db, nct_id="NCT00000001", overall_status="RECRUITING", source_hash="a")
    await _seed_study(db, nct_id="NCT00000002", overall_status="COMPLETED", source_hash="b")
    await _seed_chunk(db, "NCT00000001", 0, "recruiting study content", _basis(0))
    await _seed_chunk(db, "NCT00000002", 0, "completed study content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(statuses=["RECRUITING"]),
        config=RetrievalConfig(),
    )
    assert {r.nct_id for r in results} == {"NCT00000001"}


@pytest.mark.parametrize(
    ("study_min", "study_max", "query_min", "query_max", "expect_match"),
    [
        (0.0, 17.0, None, 12.0, True),  # pediatric study overlaps a stated child age
        (18.0, None, None, 12.0, False),  # adults-only study excludes a stated child age
        (0.0, None, 65.0, None, True),  # no upper bound on the study side: always overlaps
        (18.0, 65.0, 20.0, 30.0, True),  # query range nested inside the study's range
        (18.0, 30.0, 40.0, 50.0, False),  # disjoint ranges
    ],
)
async def test_age_interval_overlap(
    db: Database,
    study_min: float | None,
    study_max: float | None,
    query_min: float | None,
    query_max: float | None,
    expect_match: bool,
) -> None:
    await _seed_study(
        db,
        min_age_years=study_min,
        max_age_years=study_max,
    )
    await _seed_chunk(db, "NCT00000001", 0, "age eligibility content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(min_age_years=query_min, max_age_years=query_max),
        config=RetrievalConfig(),
    )
    assert bool(results) == expect_match


@pytest.mark.parametrize(
    ("study_sex", "query_sex", "expect_match"),
    [
        ("ALL", "FEMALE", True),  # a study open to all sexes matches any query constraint
        ("MALE", "FEMALE", False),
        ("MALE", "MALE", True),
        ("FEMALE", None, True),  # no query constraint -> no filtering at all
    ],
)
async def test_sex_filter(
    db: Database, study_sex: str, query_sex: str | None, expect_match: bool
) -> None:
    await _seed_study(db, sex=study_sex)
    await _seed_chunk(db, "NCT00000001", 0, "eligibility content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(sex=query_sex),  # type: ignore[arg-type]
        config=RetrievalConfig(),
    )
    assert bool(results) == expect_match


async def test_healthy_volunteers_filter(db: Database) -> None:
    await _seed_study(db, nct_id="NCT00000001", healthy_volunteers=True, source_hash="a")
    await _seed_study(db, nct_id="NCT00000002", healthy_volunteers=False, source_hash="b")
    await _seed_chunk(db, "NCT00000001", 0, "content a", _basis(0))
    await _seed_chunk(db, "NCT00000002", 0, "content b", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(healthy_volunteers=True),
        config=RetrievalConfig(),
    )
    assert {r.nct_id for r in results} == {"NCT00000001"}


async def test_null_healthy_volunteers_is_not_excluded_by_a_filter(db: Database) -> None:
    """A study that never states healthy-volunteer eligibility must not be
    excluded just because the query asked for one value or the other --
    NULL means unknown, not "no"."""
    await _seed_study(db, healthy_volunteers=None)
    await _seed_chunk(db, "NCT00000001", 0, "content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(healthy_volunteers=True),
        config=RetrievalConfig(),
    )
    assert len(results) == 1


async def test_impossible_filter_combination_returns_empty_not_an_error(db: Database) -> None:
    await _seed_study(db, phases=("PHASE1",), overall_status="COMPLETED")
    await _seed_chunk(db, "NCT00000001", 0, "content", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(phases=["PHASE3"], statuses=["WITHDRAWN"]),
        config=RetrievalConfig(),
    )
    assert results == []


async def test_multiple_filters_combine_with_and(db: Database) -> None:
    await _seed_study(
        db,
        nct_id="NCT00000001",
        phases=("PHASE3",),
        overall_status="RECRUITING",
        source_hash="a",
    )
    await _seed_study(
        db,
        nct_id="NCT00000002",
        phases=("PHASE3",),
        overall_status="COMPLETED",
        source_hash="b",
    )
    await _seed_chunk(db, "NCT00000001", 0, "content a", _basis(0))
    await _seed_chunk(db, "NCT00000002", 0, "content b", _basis(0))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(phases=["PHASE3"], statuses=["RECRUITING"]),
        config=RetrievalConfig(),
    )
    assert {r.nct_id for r in results} == {"NCT00000001"}


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


async def test_final_rank_is_contiguous_from_one(db: Database) -> None:
    await _seed_study(db)
    for i in range(3):
        await _seed_chunk(db, "NCT00000001", i, f"content number {i}", _basis(i))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(),
        config=RetrievalConfig(),
    )
    assert [r.final_rank for r in results] == list(range(1, len(results) + 1))


async def test_limit_bounds_the_result_count(db: Database) -> None:
    await _seed_study(db)
    for i in range(5):
        await _seed_chunk(db, "NCT00000001", i, f"content number {i}", _basis(i))

    results = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(),
        config=RetrievalConfig(limit=2),
    )
    assert len(results) == 2


async def test_result_carries_study_title_and_section(db: Database) -> None:
    await _seed_study(db, brief_title="A Specific Study Title")
    await _seed_chunk(db, "NCT00000001", 0, "content", _basis(0))

    (result,) = await hybrid_search(
        db,
        query_vector=_basis(0),
        search_query="content",
        filters=QueryFilters(),
        config=RetrievalConfig(),
    )
    assert result.study_title == "A Specific Study Title"
    assert result.kind == SectionKind.BRIEF_SUMMARY
    assert result.source_url == "https://clinicaltrials.gov/study/NCT00000001"
