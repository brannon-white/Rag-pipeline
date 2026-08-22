"""Chunk embedding via Voyage.

Three properties matter more than raw throughput here:

* **Re-runs must be free.** Embeddings are keyed by a content hash that covers
  the text *and* the model and dimension that produced it. Re-chunking a corpus,
  sweeping a retrieval parameter, or re-running ingest after a crash then costs
  nothing for anything unchanged. Without this, every ablation pass would
  re-embed 45,000 chunks.
* **Dimension truncation is explicit.** Voyage's embeddings are Matryoshka, so a
  1024-d vector can be truncated to 512 or 256 and re-normalised while
  remaining a valid embedding. That halves storage and index size for a recall
  delta we *measure* (the dimension ablation) rather than assume. Truncating
  without re-normalising is the classic mistake: cosine distance stops being
  comparable across vectors.
* **Query and document embeddings are asymmetric.** Voyage's models take an
  ``input_type`` and produce different vectors for the same string depending on
  it. Embedding a query as a document silently degrades recall, so the two
  paths are separate functions rather than a boolean argument.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from trialrag.domain.models import ChunkCandidate

logger = logging.getLogger(__name__)

Vector = list[float]


class EmbeddingError(RuntimeError):
    """The embedding provider failed, or returned a shape we cannot use."""


def content_hash(text: str, *, model: str, dim: int) -> str:
    """Cache key for an embedding.

    Model and dimension are part of the key on purpose. A cache keyed on text
    alone silently serves ``voyage-4-lite`` vectors after a model switch, which
    produces a corpus whose vectors are not mutually comparable -- a corruption
    that shows up only as unexplained recall loss.
    """
    payload = f"{model}\x00{dim}\x00{text}".encode()
    return hashlib.sha256(payload).hexdigest()


def truncate_and_normalise(vector: Sequence[float], dim: int) -> Vector:
    """Matryoshka-truncate to ``dim`` and re-normalise to unit length.

    Voyage's API accepts ``output_dimension`` and truncates server-side (less
    bandwidth, and the officially supported path), so in practice this usually
    receives a vector already at ``dim``. It still runs unconditionally rather
    than only as a fallback: re-normalising an already-unit vector is a cheap
    no-op, and it is the one line of defence against a provider-side rounding
    quirk or a client that did not request server-side truncation.

    Raises:
        EmbeddingError: The source vector is shorter than the requested dim.
    """
    if len(vector) < dim:
        raise EmbeddingError(
            f"cannot truncate a {len(vector)}-d embedding to {dim}-d; "
            "check embed_dim against the model's native dimension"
        )
    head = list(vector[:dim])
    norm = sum(component * component for component in head) ** 0.5
    if norm == 0.0:
        # Degenerate but not impossible on empty/whitespace input; a zero
        # vector has undefined cosine distance and would poison the index.
        raise EmbeddingError("embedding truncated to a zero vector")
    return [component / norm for component in head]


@dataclass
class EmbedStats:
    requests: int = 0
    embedded: int = 0
    cache_hits: int = 0
    tokens: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.embedded + self.cache_hits
        return self.cache_hits / total if total else 0.0


@dataclass
class Embedder:
    """Batched, retrying, cache-aware embedding client.

    ``cache`` maps content hash to a stored vector. The ingest pipeline passes
    the set of hashes already present in Postgres, so a re-run embeds only what
    genuinely changed.
    """

    model: str = "voyage-4-lite"
    dim: int = 512
    batch_size: int = 64
    max_attempts: int = 5
    api_key: str | None = None
    stats: EmbedStats = field(default_factory=EmbedStats)

    def __post_init__(self) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(
            api_key=self.api_key,
            max_retries=0,  # retries are handled here, with jitter
        )

    async def _embed_batch(self, texts: Sequence[str], input_type: str) -> list[Vector]:
        import voyageai.error as voyage_error

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            retry=retry_if_exception_type(
                (voyage_error.RateLimitError, voyage_error.ServiceUnavailableError, TimeoutError)
            ),
            reraise=True,
        ):
            with attempt:
                self.stats.requests += 1
                result = await self._client.embed(
                    list(texts),
                    model=self.model,
                    input_type=input_type,
                    output_dimension=self.dim,
                )
                self.stats.tokens += getattr(result, "total_tokens", 0) or 0
                vectors = result.embeddings
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"provider returned {len(vectors)} embeddings for {len(texts)} inputs"
                    )
                return [truncate_and_normalise(v, self.dim) for v in vectors]
        raise EmbeddingError("retries exhausted")  # pragma: no cover - reraise fires first

    async def embed_query(self, text: str) -> Vector:
        """Embed a search query.

        ``input_type="query"`` is not cosmetic: Voyage encodes queries and
        documents into the same space with different projections, and mixing
        them costs measurable recall.
        """
        vectors = await self._embed_batch([text], "query")
        return vectors[0]

    async def embed_chunks(
        self,
        chunks: Sequence[ChunkCandidate],
        *,
        cache: Mapping[str, Vector] | None = None,
        concurrency: int = 4,
    ) -> dict[str, Vector]:
        """Embed chunks, returning ``content_hash -> vector``.

        Cached hashes are skipped. Duplicate text within the batch is collapsed
        to one API call -- boilerplate criteria ("Signed informed consent") recur
        across thousands of protocols, so this is a real saving, not a
        micro-optimisation.
        """
        cache = cache or {}
        wanted: dict[str, str] = {}  # hash -> embedding input

        for chunk in chunks:
            digest = content_hash(chunk.embedding_input, model=self.model, dim=self.dim)
            if digest in cache:
                self.stats.cache_hits += 1
                continue
            wanted.setdefault(digest, chunk.embedding_input)

        if not wanted:
            return {}

        hashes = list(wanted)
        batches = [hashes[i : i + self.batch_size] for i in range(0, len(hashes), self.batch_size)]

        out: dict[str, Vector] = {}
        semaphore = asyncio.Semaphore(concurrency)

        async def run(batch: list[str]) -> None:
            async with semaphore:
                vectors = await self._embed_batch([wanted[h] for h in batch], "document")
            out.update(zip(batch, vectors, strict=True))
            self.stats.embedded += len(batch)

        await asyncio.gather(*(run(batch) for batch in batches))
        return out

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


def hashes_for(chunks: Iterable[ChunkCandidate], *, model: str, dim: int) -> dict[int, str]:
    """Map each chunk's ordinal to its content hash, without embedding anything."""
    return {
        chunk.ordinal: content_hash(chunk.embedding_input, model=model, dim=dim) for chunk in chunks
    }
