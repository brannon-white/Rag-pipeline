"""Live contract check for query parsing. Hits the real Anthropic API.

Deselected by default via ``-m 'not network'`` (see ``make test`` vs.
``make test-network``). Guards the assumption the hermetic tests in
test_filters.py encode: that ``messages.parse()`` with a Pydantic
``output_format`` still round-trips the way this module expects.
"""

from __future__ import annotations

import pytest

from trialrag.config import get_settings
from trialrag.retrieval.filters import QueryParser


@pytest.mark.network
async def test_live_query_parse_extracts_structured_filters() -> None:
    settings = get_settings()
    parser = QueryParser(api_key=settings.anthropic_api_key.get_secret_value())

    result = await parser.parse(
        "What Phase 3 trials for type 2 diabetes are recruiting adults over 18?"
    )

    assert result.filters.phases == ["PHASE3"]
    assert result.filters.statuses == ["RECRUITING"]
    assert result.filters.min_age_years == 18.0
    assert "diabetes" in result.search_query.lower()


@pytest.mark.network
async def test_live_query_parse_degrades_gracefully_on_gibberish() -> None:
    settings = get_settings()
    parser = QueryParser(api_key=settings.anthropic_api_key.get_secret_value())

    result = await parser.parse("asdkjfh laksjdf")

    assert result.filters.model_dump(exclude_none=True) == {}
