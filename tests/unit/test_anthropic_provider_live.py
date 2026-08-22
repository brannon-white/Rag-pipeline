"""Live contract check for streaming generation with native citations. Hits
the real Anthropic API.

Deselected by default via ``-m 'not network'``. Guards the assumption the
hermetic tests in ``test_anthropic_provider.py`` encode: that
``messages.stream()`` with ``document`` content blocks actually yields
``citations_delta`` events whose ``cited_text`` is a real, verifiable
substring of the chunk that was sent -- the single-example version of the
citation-validity check the plan's Tier-2 eval will later run at scale.
"""

from __future__ import annotations

import pytest

from trialrag.config import get_settings
from trialrag.domain.models import RetrievedChunk, SectionKind
from trialrag.generation.anthropic_provider import AnthropicProvider
from trialrag.generation.provider import CitationDelta, Done, TextDelta


@pytest.mark.network
async def test_live_generation_cites_real_source_text() -> None:
    settings = get_settings()
    provider = AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value())

    chunk = RetrievedChunk(
        chunk_id=1,
        nct_id="NCT00000001",
        kind=SectionKind.BRIEF_SUMMARY,
        content=(
            "This is a Phase 3, randomized, double-blind study evaluating a novel "
            "oral medication for adults with type 2 diabetes mellitus. The study "
            "is sponsored by Acme Pharmaceuticals."
        ),
        context_header="NCT00000001 -- Brief Summary",
        study_title="A study of a novel diabetes medication",
    )

    try:
        events = [
            event
            async for event in provider.generate_stream(
                "What phase is this trial and who sponsors it?",
                [chunk],
                model=settings.answer_model,
                effort=settings.answer_effort,
                max_tokens=1024,
            )
        ]
    finally:
        await provider.aclose()

    citations = [e for e in events if isinstance(e, CitationDelta)]
    done = next(e for e in events if isinstance(e, Done))

    assert any(isinstance(e, TextDelta) for e in events)
    assert citations, "expected at least one citation against a real chunk"
    for citation in citations:
        assert citation.cited_text in chunk.content
        assert citation.nct_id == "NCT00000001"
    assert done.stop_reason != "refusal"
    assert done.cost_usd > 0
