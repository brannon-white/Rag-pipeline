"""AnthropicProvider tests.

Hermetic: the Anthropic client is replaced with a fake streaming client, since
what's under test (event mapping, citation-to-chunk resolution via
``document_index``, refusal-as-abstention, cost computation) is independent of
a live model call. Same convention as ``test_rerank.py``: construct the real
dataclass, then swap ``.client`` for a fake before calling.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trialrag.domain.models import RetrievedChunk, SectionKind
from trialrag.generation.anthropic_provider import AnthropicProvider, build_document_blocks
from trialrag.generation.pricing import compute_cost_usd
from trialrag.generation.provider import CitationDelta, Done, TextDelta


def _chunk(nct_label: str, chunk_id: int, content: str = "text") -> RetrievedChunk:
    digits = nct_label.removeprefix("NCT")
    return RetrievedChunk(
        chunk_id=chunk_id,
        nct_id=f"NCT{digits:0>8}",
        kind=SectionKind.BRIEF_SUMMARY,
        content=content,
        context_header="hdr",
        study_title="A study",
    )


def _text_event(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=text)
    )


def _citation_event(
    cited_text: str, document_index: int, start: int, end: int
) -> SimpleNamespace:
    citation = SimpleNamespace(
        type="char_location",
        cited_text=cited_text,
        document_index=document_index,
        start_char_index=start,
        end_char_index=end,
    )
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type="citations_delta", citation=citation)
    )


def _final_message(
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read_input_tokens: int | None = 0,
    cache_creation_input_tokens: int | None = 0,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    return SimpleNamespace(stop_reason=stop_reason, usage=usage)


class _FakeMessageStream:
    def __init__(self, events: list[SimpleNamespace], final_message: SimpleNamespace) -> None:
        self._events = events
        self._final_message = final_message

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for event in self._events:
            yield event

    async def get_final_message(self) -> SimpleNamespace:
        return self._final_message


class _FakeStreamManager:
    def __init__(self, stream: _FakeMessageStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> _FakeMessageStream:
        return self._stream

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeMessagesAPI:
    def __init__(self, events: list[SimpleNamespace], final_message: SimpleNamespace) -> None:
        self.events = events
        self.final_message = final_message
        self.last_kwargs: dict[str, object] = {}

    def stream(self, **kwargs: object) -> _FakeStreamManager:
        self.last_kwargs = kwargs
        return _FakeStreamManager(_FakeMessageStream(self.events, self.final_message))


class _FakeAnthropicClient:
    def __init__(self, events: list[SimpleNamespace], final_message: SimpleNamespace) -> None:
        self.messages = _FakeMessagesAPI(events, final_message)

    async def close(self) -> None:
        pass


def _wired_provider(events: list[SimpleNamespace], final_message: SimpleNamespace) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="test-key")
    provider.client = _FakeAnthropicClient(events, final_message)  # type: ignore[assignment]
    return provider


# ---------------------------------------------------------------------------
# build_document_blocks
# ---------------------------------------------------------------------------


def test_only_last_document_block_has_cache_control() -> None:
    chunks = [_chunk("NCT1", 1), _chunk("NCT2", 2), _chunk("NCT3", 3)]
    blocks = build_document_blocks(chunks)
    assert "cache_control" not in blocks[0]
    assert "cache_control" not in blocks[1]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}


def test_build_document_blocks_empty_input() -> None:
    assert build_document_blocks([]) == []


# ---------------------------------------------------------------------------
# generate_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_yields_text_then_citation_then_done() -> None:
    chunks = [_chunk("NCT1", 101), _chunk("NCT2", 202)]
    events = [
        _text_event("The trial is Phase 3"),
        _citation_event("Phase 3", document_index=1, start=10, end=20),
    ]
    provider = _wired_provider(events, _final_message())

    received = [
        event async for event in provider.generate_stream(
            "what phase", chunks, model="claude-opus-5", effort="low", max_tokens=1024
        )
    ]

    assert isinstance(received[0], TextDelta)
    assert received[0].text == "The trial is Phase 3"

    citation = received[1]
    assert isinstance(citation, CitationDelta)
    # document_index=1 must resolve to chunks[1], not chunks[0]
    assert citation.chunk_id == 202
    assert citation.nct_id == "NCT00000002"

    done = received[2]
    assert isinstance(done, Done)
    assert done.abstained is False
    assert done.stop_reason == "end_turn"
    assert done.cost_usd > 0


@pytest.mark.asyncio
async def test_refusal_stop_reason_sets_abstained() -> None:
    provider = _wired_provider([], _final_message(stop_reason="refusal"))

    events = [
        event async for event in provider.generate_stream(
            "query", [_chunk("NCT1", 1)], model="claude-opus-5", effort="low", max_tokens=1024
        )
    ]

    done = events[-1]
    assert isinstance(done, Done)
    assert done.abstained is True
    assert done.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_non_char_location_citations_are_skipped() -> None:
    """Page/content-block/search-result citation kinds carry no char span to
    highlight against plain-text chunks -- only ``char_location`` citations
    are meaningful here, since every document is sent as ``type: "text"``."""
    page_citation = SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(
            type="citations_delta", citation=SimpleNamespace(type="page_location")
        ),
    )
    provider = _wired_provider([page_citation], _final_message())

    events = [
        event async for event in provider.generate_stream(
            "query", [_chunk("NCT1", 1)], model="claude-opus-5", effort="low", max_tokens=1024
        )
    ]

    assert all(not isinstance(e, CitationDelta) for e in events)


@pytest.mark.asyncio
async def test_cost_matches_pricing_module() -> None:
    provider = _wired_provider(
        [], _final_message(input_tokens=1000, output_tokens=500, cache_read_input_tokens=200)
    )

    events = [
        event async for event in provider.generate_stream(
            "query", [], model="claude-sonnet-5", effort="low", max_tokens=1024
        )
    ]

    done = events[-1]
    assert isinstance(done, Done)
    expected = compute_cost_usd(
        model="claude-sonnet-5",
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=0,
    )
    assert done.cost_usd == pytest.approx(expected)


def test_count_tokens_is_positive_and_scales_with_length() -> None:
    provider = AnthropicProvider(api_key="test-key")
    assert provider.count_tokens("") >= 1
    assert provider.count_tokens("a" * 400) > provider.count_tokens("a" * 40)
