"""Reranker tests.

Hermetic: the Voyage rerank client is replaced with a deterministic fake, since
the properties under test (score-based reordering, the diversity cap, empty-
input short-circuiting) are client-agnostic. The full pipeline (parse -> embed
-> hybrid search -> rerank) was additionally verified live against the real
ingested corpus; see the retrieval module docstrings for what that confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from trialrag.domain.models import RetrievedChunk, SectionKind
from trialrag.retrieval.rerank import Reranker, apply_diversity_cap


def _chunk(nct_label: str, chunk_id: int, content: str = "text") -> RetrievedChunk:
    """``nct_label`` is a short human label ("NCT1", "NCT2", ...); padded to a
    valid 8-digit NCT ID since RetrievedChunk validates that format."""
    digits = nct_label.removeprefix("NCT")
    return RetrievedChunk(
        chunk_id=chunk_id,
        nct_id=f"NCT{digits:0>8}",
        kind=SectionKind.BRIEF_SUMMARY,
        content=content,
        context_header="hdr",
        study_title="A study",
    )


# ---------------------------------------------------------------------------
# apply_diversity_cap
# ---------------------------------------------------------------------------


def test_diversity_cap_keeps_best_ranked_per_study() -> None:
    chunks = [
        _chunk("NCT1", 1),
        _chunk("NCT1", 2),
        _chunk("NCT1", 3),
        _chunk("NCT2", 4),
    ]
    capped = apply_diversity_cap(chunks, max_per_study=2)
    assert [c.chunk_id for c in capped] == [1, 2, 4]


def test_diversity_cap_preserves_input_order() -> None:
    """The cap is applied to an already-ranked list -- it must not reorder,
    only drop, or a lower-ranked chunk could jump ahead of a higher one."""
    chunks = [_chunk("NCT1", 1), _chunk("NCT2", 2), _chunk("NCT1", 3)]
    capped = apply_diversity_cap(chunks, max_per_study=1)
    assert [c.chunk_id for c in capped] == [1, 2]


def test_diversity_cap_of_zero_yields_nothing() -> None:
    assert apply_diversity_cap([_chunk("NCT1", 1)], max_per_study=0) == []


def test_diversity_cap_noop_when_under_the_limit() -> None:
    chunks = [_chunk("NCT1", 1), _chunk("NCT2", 2)]
    assert apply_diversity_cap(chunks, max_per_study=5) == chunks


# ---------------------------------------------------------------------------
# Reranker, against a fake Voyage client
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    index: int
    document: str
    relevance_score: float


@dataclass
class _FakeRerankResponse:
    results: list[_Result]


@dataclass
class _FakeVoyageClient:
    scores: list[float]  # relevance score per input document, in input order
    calls: int = 0
    last_documents: list[str] | None = None

    async def rerank(self, query: str, documents: list[str], model: str) -> _FakeRerankResponse:
        self.calls += 1
        self.last_documents = documents
        results = [
            _Result(index=i, document=doc, relevance_score=self.scores[i])
            for i, doc in enumerate(documents)
        ]
        return _FakeRerankResponse(results=results)


def _reranker(scores: list[float]) -> Reranker:
    reranker = Reranker(api_key="unused", rate_limit_rpm=6000.0)
    reranker._client = _FakeVoyageClient(scores=scores)  # type: ignore[assignment]
    return reranker


async def test_rerank_reorders_by_relevance_score() -> None:
    candidates = [_chunk("NCT1", 1, "low relevance"), _chunk("NCT2", 2, "high relevance")]
    reranker = _reranker(scores=[0.1, 0.9])

    result = await reranker.rerank("query", candidates, top_n=5, max_per_study=5)

    assert [r.chunk_id for r in result] == [2, 1]
    assert result[0].rerank_score == pytest.approx(0.9)
    assert result[1].rerank_score == pytest.approx(0.1)


async def test_rerank_includes_context_header_in_the_reranked_text() -> None:
    """Stripping the header would discard exactly the disambiguating signal
    contextual retrieval was built to add (which study a criterion belongs
    to)."""
    candidates = [_chunk("NCT1", 1, "some criterion")]
    reranker = _reranker(scores=[0.5])

    await reranker.rerank("query", candidates, top_n=5, max_per_study=5)

    (sent,) = reranker._client.last_documents  # type: ignore[misc]
    assert "hdr" in sent
    assert "some criterion" in sent


async def test_rerank_applies_diversity_cap_after_reordering() -> None:
    candidates = [
        _chunk("NCT1", 1),  # will score lowest
        _chunk("NCT1", 2),  # highest
        _chunk("NCT1", 3),  # second highest
    ]
    reranker = _reranker(scores=[0.1, 0.9, 0.8])

    result = await reranker.rerank("query", candidates, top_n=5, max_per_study=2)

    assert [r.chunk_id for r in result] == [2, 3]  # best two, low-scorer capped out


async def test_rerank_truncates_to_top_n_after_cap() -> None:
    candidates = [_chunk(f"NCT{i}", i) for i in range(5)]
    reranker = _reranker(scores=[0.5, 0.9, 0.1, 0.8, 0.3])

    result = await reranker.rerank("query", candidates, top_n=2, max_per_study=5)

    assert [r.chunk_id for r in result] == [1, 3]  # scores 0.9, 0.8


async def test_final_rank_is_renumbered_from_the_new_order() -> None:
    candidates = [_chunk("NCT1", 1), _chunk("NCT2", 2)]
    reranker = _reranker(scores=[0.1, 0.9])

    result = await reranker.rerank("query", candidates, top_n=5, max_per_study=5)

    assert [r.final_rank for r in result] == [1, 2]


async def test_empty_candidates_short_circuits_without_an_api_call() -> None:
    reranker = _reranker(scores=[])
    result = await reranker.rerank("query", [], top_n=5, max_per_study=5)
    assert result == []
    assert reranker._client.calls == 0  # type: ignore[attr-defined]
