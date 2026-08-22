"""Query-parser tests.

Hermetic: the Anthropic client is replaced with a fake that returns a
pre-built ``ParsedMessage``-shaped object, since the properties under test
(normalisation, refusal handling, empty-query rejection) don't need a live
model call to verify. The real call is exercised once, live, in
``test_filters_live.py`` (network-marked).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from trialrag.retrieval.filters import (
    KNOWN_PHASES,
    KNOWN_STATUSES,
    ParsedQuery,
    QueryFilters,
    QueryParseError,
    QueryParser,
)

# ---------------------------------------------------------------------------
# QueryFilters.normalised()
# ---------------------------------------------------------------------------


def test_normalised_keeps_known_tokens() -> None:
    filters = QueryFilters(phases=["PHASE3"], statuses=["RECRUITING"])
    assert filters.normalised().phases == ["PHASE3"]
    assert filters.normalised().statuses == ["RECRUITING"]


def test_normalised_drops_unrecognised_phase() -> None:
    """The model might write "Phase III" instead of "PHASE3" -- passing that
    through to a SQL array-membership filter would just silently match
    nothing, which reads as an empty result set for no visible reason."""
    filters = QueryFilters(phases=["Phase III", "PHASE3"])
    assert filters.normalised().phases == ["PHASE3"]


def test_normalised_drops_unrecognised_status() -> None:
    filters = QueryFilters(statuses=["Active", "RECRUITING"])
    assert filters.normalised().statuses == ["RECRUITING"]


def test_normalised_empty_after_filtering_becomes_none_not_empty_list() -> None:
    """An empty list and a null filter must behave identically downstream --
    otherwise `phases = ANY($1)` with `$1 = '{}'` matches nothing instead of
    everything, silently zeroing out every result."""
    filters = QueryFilters(phases=["nonsense"])
    assert filters.normalised().phases is None


def test_normalised_preserves_other_fields() -> None:
    filters = QueryFilters(min_age_years=18.0, sex="FEMALE")
    normalised = filters.normalised()
    assert normalised.min_age_years == 18.0
    assert normalised.sex == "FEMALE"


def test_known_vocab_excludes_unknown_status() -> None:
    """UNKNOWN is a domain fallback for unparseable registry data, never
    something a user could meaningfully ask to filter by."""
    assert "UNKNOWN" not in KNOWN_STATUSES


def test_known_phases_cover_registry_vocabulary() -> None:
    assert {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"} == KNOWN_PHASES


# ---------------------------------------------------------------------------
# QueryParser, against a fake Anthropic client
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    parsed_output: ParsedQuery | None
    stop_reason: str = "end_turn"


@dataclass
class _FakeMessages:
    response: _FakeResponse
    captured_kwargs: dict[str, Any] | None = None

    async def parse(self, **kwargs: Any) -> _FakeResponse:
        if self.captured_kwargs is not None:
            self.captured_kwargs.update(kwargs)
        return self.response


@dataclass
class _FakeClient:
    messages: _FakeMessages


def _parser_with(response: _FakeResponse, captured: dict[str, Any] | None = None) -> QueryParser:
    parser = QueryParser(api_key="test-key-not-used")
    parser.client = _FakeClient(messages=_FakeMessages(response, captured))  # type: ignore[assignment]
    return parser


async def test_parse_returns_normalised_filters() -> None:
    response = _FakeResponse(
        parsed_output=ParsedQuery(
            filters=QueryFilters(phases=["PHASE3", "Phase III"]), search_query="diabetes"
        )
    )
    result = await _parser_with(response).parse("phase 3 diabetes trials")
    assert result.filters.phases == ["PHASE3"]
    assert result.search_query == "diabetes"


async def test_empty_query_raises_without_calling_the_api() -> None:
    captured: dict[str, Any] = {}
    parser = _parser_with(_FakeResponse(parsed_output=None), captured)
    with pytest.raises(QueryParseError, match="empty"):
        await parser.parse("   ")
    assert captured == {}


async def test_refusal_raises_query_parse_error() -> None:
    response = _FakeResponse(
        parsed_output=ParsedQuery(filters=QueryFilters(), search_query=""),
        stop_reason="refusal",
    )
    with pytest.raises(QueryParseError, match="refused"):
        await _parser_with(response).parse("some query")


async def test_missing_parsed_output_raises() -> None:
    with pytest.raises(QueryParseError, match="parsed_output"):
        await _parser_with(_FakeResponse(parsed_output=None)).parse("some query")


async def test_effort_and_model_are_forwarded() -> None:
    captured: dict[str, Any] = {}
    parser = _parser_with(
        _FakeResponse(parsed_output=ParsedQuery(filters=QueryFilters(), search_query="x")),
        captured,
    )
    parser.model = "claude-opus-5"
    parser.effort = "low"
    await parser.parse("query text")

    assert captured["model"] == "claude-opus-5"
    assert captured["output_config"] == {"effort": "low"}
    assert captured["output_format"] is ParsedQuery
    assert captured["messages"] == [{"role": "user", "content": "query text"}]


async def test_system_prompt_is_cached() -> None:
    """A per-request-varying system prompt would defeat prompt caching --
    this call happens on every query, so the cache_control breakpoint is
    load-bearing for cost, not decorative."""
    captured: dict[str, Any] = {}
    parser = _parser_with(
        _FakeResponse(parsed_output=ParsedQuery(filters=QueryFilters(), search_query="x")),
        captured,
    )
    await parser.parse("query text")

    system = captured["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
