"""Embedder tests.

Hermetic: the Voyage client is monkeypatched with a deterministic fake, since
the properties under test (caching, truncation-normalisation, dedup, query vs.
document asymmetry) are all client-agnostic. A live contract test would only
duplicate ``test_fetch.py``'s pattern for a provider we do not need to
re-verify per module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from trialrag.domain.models import ChunkCandidate, SectionKind
from trialrag.ingest.embed import Embedder, content_hash, hashes_for, truncate_and_normalise


def _chunk(text: str, *, header: str = "hdr", ordinal: int = 0) -> ChunkCandidate:
    return ChunkCandidate(
        nct_id="NCT00000001",
        kind=SectionKind.BRIEF_SUMMARY,
        ordinal=ordinal,
        content=text,
        context_header=header,
        token_count=len(text.split()),
    )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_truncate_and_normalise_unit_length() -> None:
    out = truncate_and_normalise([3.0, 4.0, 0.0, 0.0], dim=2)
    assert out == [pytest.approx(0.6), pytest.approx(0.8)]
    assert math.hypot(*out) == pytest.approx(1.0)


def test_truncate_and_normalise_rejects_undersized_source() -> None:
    from trialrag.ingest.embed import EmbeddingError

    with pytest.raises(EmbeddingError, match="cannot truncate"):
        truncate_and_normalise([1.0, 2.0], dim=4)


def test_truncate_and_normalise_rejects_zero_vector() -> None:
    from trialrag.ingest.embed import EmbeddingError

    with pytest.raises(EmbeddingError, match="zero vector"):
        truncate_and_normalise([0.0, 0.0, 0.0], dim=3)


def test_content_hash_distinguishes_model_and_dim() -> None:
    """A cache keyed on text alone would serve stale vectors after a model or
    dimension change -- silent corpus corruption that only shows up as
    unexplained recall loss."""
    base = content_hash("same text", model="voyage-4-lite", dim=512)
    assert content_hash("same text", model="voyage-4-lite", dim=256) != base
    assert content_hash("same text", model="voyage-4", dim=512) != base
    assert content_hash("same text", model="voyage-4-lite", dim=512) == base


def test_content_hash_distinguishes_text() -> None:
    a = content_hash("alpha", model="m", dim=8)
    b = content_hash("beta", model="m", dim=8)
    assert a != b


def test_hashes_for_keys_by_ordinal() -> None:
    chunks = [_chunk("alpha", ordinal=0), _chunk("beta", ordinal=1)]
    hashes = hashes_for(chunks, model="voyage-4-lite", dim=512)
    assert set(hashes) == {0, 1}
    assert hashes[0] == content_hash(chunks[0].embedding_input, model="voyage-4-lite", dim=512)


# ---------------------------------------------------------------------------
# Embedder, against a fake Voyage client
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    embeddings: list[list[float]]
    total_tokens: int = 10


@dataclass
class _FakeVoyageClient:
    """Deterministic stand-in: embeds a string as its length, in ``dim`` slots."""

    dim_seen: list[int] = field(default_factory=list)
    input_types_seen: list[str] = field(default_factory=list)
    batches_seen: list[list[str]] = field(default_factory=list)
    calls: int = 0

    async def embed(
        self, texts: list[str], model: str, input_type: str, output_dimension: int, **_: Any
    ) -> _FakeResult:
        self.calls += 1
        self.dim_seen.append(output_dimension)
        self.input_types_seen.append(input_type)
        self.batches_seen.append(list(texts))
        vectors = []
        for text in texts:
            # A distinct, non-degenerate vector per string, sized to dim.
            base = float(len(text) + 1)
            vectors.append([base] * output_dimension)
        return _FakeResult(embeddings=vectors)


def _embedder(**kwargs: Any) -> Embedder:
    kwargs.setdefault("batch_size", 8)
    # Fast by default: the real 3 req/min production default would make any
    # test issuing more than one request wait tens of seconds for real.
    kwargs.setdefault("rate_limit_rpm", 6000.0)
    embedder = Embedder(model="voyage-4-lite", dim=16, **kwargs)
    embedder._client = _FakeVoyageClient()  # bypass real construction
    return embedder


async def test_embed_query_uses_query_input_type() -> None:
    embedder = _embedder()
    vector = await embedder.embed_query("what is the eligible age range?")
    assert len(vector) == 16
    assert embedder._client.input_types_seen == ["query"]


async def test_embed_chunks_uses_document_input_type() -> None:
    embedder = _embedder()
    await embedder.embed_chunks([_chunk("alpha"), _chunk("beta")])
    assert embedder._client.input_types_seen == ["document"]


async def test_embed_chunks_returns_normalised_vectors() -> None:
    embedder = _embedder()
    out = await embedder.embed_chunks([_chunk("alpha")])
    (vector,) = out.values()
    assert math.hypot(*vector) == pytest.approx(1.0)


async def test_cached_hashes_are_not_re_embedded() -> None:
    embedder = _embedder()
    chunks = [_chunk("alpha", ordinal=0), _chunk("beta", ordinal=1)]
    already = content_hash(chunks[0].embedding_input, model="voyage-4-lite", dim=16)

    out = await embedder.embed_chunks(chunks, cache={already: [0.0] * 16})

    assert len(out) == 1  # only "beta" was embedded
    assert embedder.stats.cache_hits == 1
    assert embedder.stats.embedded == 1


async def test_fully_cached_batch_makes_no_api_call() -> None:
    embedder = _embedder()
    chunks = [_chunk("alpha", ordinal=0)]
    digest = content_hash(chunks[0].embedding_input, model="voyage-4-lite", dim=16)

    out = await embedder.embed_chunks(chunks, cache={digest: [1.0] * 16})

    assert out == {}
    assert embedder._client.calls == 0


async def test_duplicate_text_within_a_batch_is_deduplicated() -> None:
    """Boilerplate criteria recur verbatim across thousands of protocols;
    embedding the same string twice in one call is pure waste."""
    embedder = _embedder()
    chunks = [
        _chunk("Signed informed consent", header="hdr", ordinal=0),
        _chunk("Signed informed consent", header="hdr", ordinal=1),
        _chunk("A distinct criterion", header="hdr", ordinal=2),
    ]
    out = await embedder.embed_chunks(chunks)

    assert len(out) == 2  # one hash for the duplicate pair, one for the distinct
    assert sum(len(batch) for batch in embedder._client.batches_seen) == 2


async def test_batching_respects_batch_size() -> None:
    embedder = _embedder(batch_size=2)
    chunks = [_chunk(f"criterion number {i}", ordinal=i) for i in range(5)]
    await embedder.embed_chunks(chunks)

    assert embedder._client.calls == 3  # ceil(5/2)
    assert all(len(batch) <= 2 for batch in embedder._client.batches_seen)


async def test_empty_input_makes_no_api_call() -> None:
    embedder = _embedder()
    assert await embedder.embed_chunks([]) == {}
    assert embedder._client.calls == 0


async def test_mismatched_response_length_raises() -> None:
    from trialrag.ingest.embed import EmbeddingError

    class _BadClient:
        async def embed(self, texts: list[str], **_: Any) -> _FakeResult:
            return _FakeResult(embeddings=[[1.0] * 16])  # always 1, regardless of input

    embedder = _embedder()
    embedder._client = _BadClient()  # type: ignore[assignment]
    with pytest.raises(EmbeddingError, match="returned 1 embeddings for 2"):
        await embedder.embed_chunks([_chunk("alpha"), _chunk("beta")])


async def test_stats_track_hit_rate() -> None:
    embedder = _embedder()
    chunks = [_chunk("alpha", ordinal=0), _chunk("beta", ordinal=1)]
    cached = content_hash(chunks[0].embedding_input, model="voyage-4-lite", dim=16)

    await embedder.embed_chunks(chunks, cache={cached: [0.0] * 16})

    assert embedder.stats.cache_hits == 1
    assert embedder.stats.embedded == 1
    assert embedder.stats.hit_rate == pytest.approx(0.5)


async def test_requests_are_paced_by_the_rate_limiter() -> None:
    """Regression guard for the production incident: an unpaid Voyage account
    is hard-capped at 3 req/min, and a real ingest run failed outright with
    RateLimitError before the embedder paced its own requests. Uses a scaled
    rate (120/min) so the property -- successive batches wait rather than
    firing immediately -- is checked in well under a second."""
    import asyncio

    embedder = _embedder(batch_size=1, rate_limit_rpm=120.0)  # 2 req/s
    chunks = [_chunk(f"criterion {i}", ordinal=i) for i in range(2)]

    loop = asyncio.get_running_loop()
    start = loop.time()
    await embedder.embed_chunks(chunks, concurrency=1)
    elapsed = loop.time() - start

    assert embedder._client.calls == 2
    assert elapsed >= 0.4  # 2nd of 2 sequential batches waits ~0.5s behind the 1st


async def test_default_rate_limit_matches_unpaid_voyage_ceiling() -> None:
    """Pins the production default to the observed real ceiling (3 req/min) --
    a regression in this default is exactly what caused the original failure."""
    embedder = Embedder(model="voyage-4-lite", dim=16)
    embedder._client = _FakeVoyageClient()
    assert embedder.rate_limit_rpm == 3.0
