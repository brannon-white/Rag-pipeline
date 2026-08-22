"""Chunker tests.

Chunking is where retrieval quality is silently won or lost, and the failure
modes are invariant violations rather than wrong-looking output: an oversized
chunk truncates at the embedding API, a criterion split in half inverts its
meaning, dropped text is unretrievable forever. Those are stated as properties
and checked with generated input, because handwritten examples only cover the
shapes we already thought of.

Tests use the deterministic heuristic counter so they stay hermetic and give
identical boundaries on every machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trialrag.domain.models import Section, SectionKind, Study
from trialrag.ingest.chunk import (
    ChunkingConfig,
    chunk_section,
    chunk_study,
    split_criteria,
)
from trialrag.ingest.parse import iter_sections, parse_study
from trialrag.ingest.tokens import HeuristicTokenCounter

COUNT = HeuristicTokenCounter()
CONFIG = ChunkingConfig(max_tokens=120, min_tokens=16, overlap_tokens=20)


@pytest.fixture
def study(bounded_age_study: dict[str, Any]) -> Study:
    return parse_study(bounded_age_study)


def _section(kind: SectionKind, text: str, nct_id: str = "NCT00000102") -> Section:
    return Section(nct_id=nct_id, kind=kind, text=text)


# ---------------------------------------------------------------------------
# Criterion splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("* alpha\n* beta\n* gamma", 3),
        ("• alpha\n• beta", 2),
        ("- alpha\n- beta", 2),
        ("1. alpha\n2. beta\n3. gamma", 3),
        ("a) alpha\nb) beta", 2),
        ("alpha\n\nbeta\n\ngamma", 3),  # no bullets -> paragraphs
        ("just one prose sentence", 1),
    ],
)
def test_split_criteria_handles_marker_styles(text: str, expected: int) -> None:
    assert len(split_criteria(text)) == expected


def test_split_criteria_drops_empties_not_content() -> None:
    parts = split_criteria("* alpha\n\n*  \n* beta")
    assert parts == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Core invariants
# ---------------------------------------------------------------------------

# Each criterion is prefixed with its index so no criterion is a substring of
# another. Without that, containment assertions below report false duplicates
# whenever the generator happens to emit "aa" and "aa aa".
criteria_text = st.lists(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=122), min_size=1, max_size=200),
    min_size=1,
    max_size=25,
).map(lambda items: "\n".join(f"* c{i}z {item}" for i, item in enumerate(items)))

_words = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=9), min_size=1, max_size=90
).map(" ".join)

# Multi-paragraph prose with sentence structure and paragraph lengths that
# straddle the budget. Unstructured random text never exercises the packing
# path's overlap carry-forward, which is exactly where the budget can be
# exceeded -- an earlier generator missed a real bug that real records hit.
_paragraph = st.lists(_words, min_size=1, max_size=5).map(lambda s: ". ".join(s) + ".")

prose_text = st.one_of(
    st.just(""),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=122), min_size=0, max_size=800),
    st.lists(_paragraph, min_size=1, max_size=8).map("\n\n".join),
)


@given(text=criteria_text)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_no_chunk_exceeds_budget_criteria(text: str, study: Study) -> None:
    for chunk in chunk_section(
        _section(SectionKind.ELIGIBILITY_INCLUSION, text), study, COUNT, CONFIG
    ):
        assert chunk.token_count <= CONFIG.max_tokens


@given(text=prose_text)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_no_chunk_exceeds_budget_prose(text: str, study: Study) -> None:
    for chunk in chunk_section(_section(SectionKind.BRIEF_SUMMARY, text), study, COUNT, CONFIG):
        assert chunk.token_count <= CONFIG.max_tokens


@given(text=criteria_text)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_criteria_are_never_split_across_chunks(text: str, study: Study) -> None:
    """Every criterion that fits the budget survives intact inside some chunk.

    This is the property that protects meaning: half of a negated criterion
    reads as an assertion of the thing it excludes.
    """
    chunks = chunk_section(_section(SectionKind.ELIGIBILITY_INCLUSION, text), study, COUNT, CONFIG)
    bodies = [chunk.content for chunk in chunks]

    for criterion in split_criteria(text):
        if COUNT(criterion) > CONFIG.max_tokens:
            continue  # oversized single criteria are legitimately broken down
        assert any(criterion in body for body in bodies), f"criterion lost or split: {criterion!r}"


@given(text=criteria_text)
@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_criteria_chunks_do_not_duplicate_content(text: str, study: Study) -> None:
    """Criteria chunking is overlap-free, so no criterion appears twice.

    Duplicated criteria make one proposition compete with itself for top-k
    slots, displacing distinct ones.
    """
    chunks = chunk_section(_section(SectionKind.ELIGIBILITY_INCLUSION, text), study, COUNT, CONFIG)
    for criterion in split_criteria(text):
        if COUNT(criterion) > CONFIG.max_tokens or COUNT(criterion) < 3:
            continue
        occurrences = sum(criterion in chunk.content for chunk in chunks)
        assert occurrences <= 1, f"{criterion!r} duplicated across {occurrences} chunks"


@given(text=prose_text)
@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_chunking_terminates_and_is_total(text: str, study: Study) -> None:
    """Non-blank input always yields at least one chunk; blank yields none."""
    chunks = chunk_section(_section(SectionKind.BRIEF_SUMMARY, text), study, COUNT, CONFIG)
    assert bool(chunks) == bool(text.strip())


@given(text=criteria_text)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_ordinals_are_contiguous_from_start(text: str, study: Study) -> None:
    chunks = chunk_section(
        _section(SectionKind.ELIGIBILITY_INCLUSION, text), study, COUNT, CONFIG, start_ordinal=7
    )
    assert [c.ordinal for c in chunks] == list(range(7, 7 + len(chunks)))


@given(text=st.one_of(criteria_text, prose_text))
@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_chunking_is_deterministic(text: str, study: Study) -> None:
    """Same input, same chunks -- otherwise eval runs are not comparable."""
    section = _section(SectionKind.BRIEF_SUMMARY, text)
    first = chunk_section(section, study, COUNT, CONFIG)
    second = chunk_section(section, study, COUNT, CONFIG)
    assert [c.content for c in first] == [c.content for c in second]


# ---------------------------------------------------------------------------
# Behaviour on real records
# ---------------------------------------------------------------------------


def test_slivers_are_merged_away(study: Study) -> None:
    """NCT03840798's exclusion section is the single criterion "None"."""
    chunks = chunk_section(
        _section(SectionKind.ELIGIBILITY_INCLUSION, "* Adequate organ function\n* None"),
        study,
        COUNT,
        CONFIG,
    )
    assert len(chunks) == 1
    assert "None" in chunks[0].content


def test_every_chunk_carries_the_context_header(study: Study) -> None:
    chunks = chunk_section(
        _section(SectionKind.ELIGIBILITY_EXCLUSION, "* alpha\n* beta"), study, COUNT, CONFIG
    )
    for chunk in chunks:
        assert study.nct_id in chunk.context_header
        assert "eligibility exclusion" in chunk.context_header
        # Header and body must both reach the index, via one shared string.
        assert chunk.embedding_input.startswith(chunk.context_header)
        assert chunk.content in chunk.embedding_input


def test_oversized_single_criterion_is_broken_down(study: Study) -> None:
    giant = "* " + " ".join(f"word{i}" for i in range(600))
    chunks = chunk_section(_section(SectionKind.ELIGIBILITY_INCLUSION, giant), study, COUNT, CONFIG)
    assert len(chunks) > 1
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)


@pytest.mark.parametrize(
    "fixture_name", ["study_nct00000102", "study_nct03840798", "study_nct04368728"]
)
def test_real_studies_chunk_within_budget(fixture_name: str) -> None:
    raw = json.loads(
        (Path(__file__).parents[1] / "fixtures" / f"{fixture_name}.json").read_text(
            encoding="utf-8"
        )
    )
    parsed = parse_study(raw)
    chunks = chunk_study(iter_sections(raw, parsed), parsed, COUNT, CONFIG)

    assert chunks, "a real study must produce chunks"
    assert all(c.token_count <= CONFIG.max_tokens for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.nct_id == parsed.nct_id for c in chunks)
    assert all(c.content.strip() for c in chunks)


def test_flat_strategy_ignores_structure(study: Study) -> None:
    """The ablation baseline must actually differ from the structural path."""
    text = "* " + ("alpha " * 60) + "\n* " + ("beta " * 60)
    section = _section(SectionKind.ELIGIBILITY_INCLUSION, text)

    structural = chunk_section(section, study, COUNT, CONFIG)
    flat = chunk_section(
        section, study, COUNT, ChunkingConfig(**{**CONFIG.__dict__, "strategy": "flat"})
    )
    assert [c.content for c in structural] != [c.content for c in flat]


def test_config_rejects_incoherent_budgets() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ChunkingConfig(max_tokens=50, min_tokens=50)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(max_tokens=100, overlap_tokens=100)
