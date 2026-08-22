"""Cross-encoder reranking over the fused candidate set.

RRF fusion is a cheap, rank-based signal; a reranker actually reads the query
against each candidate's text and is the more expensive, more accurate second
pass. Run over the top ~50 fused candidates rather than the whole corpus, so
the cost stays bounded regardless of corpus size.

Reranking happens on the same ``context_header + content`` text that was
embedded (see :attr:`trialrag.domain.models.ChunkCandidate.embedding_input`),
not on bare content: the header is what disambiguates "Exclusion Criteria"
text from *which study's* exclusion criteria, and stripping it here would
throw away exactly the signal contextual retrieval was built to add.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from trialrag.domain.models import RetrievedChunk
from trialrag.ratelimit import TokenBucket

logger = logging.getLogger(__name__)


def _rerank_text(chunk: RetrievedChunk) -> str:
    return f"{chunk.context_header}\n\n{chunk.content}"


def apply_diversity_cap(
    chunks: list[RetrievedChunk], *, max_per_study: int
) -> list[RetrievedChunk]:
    """Drop chunks beyond ``max_per_study`` for any one ``nct_id``.

    Applied *after* rerank ordering, so each study keeps only its best-ranked
    chunks. Without this, one heavily-matched study can fill the whole answer
    context with near-duplicate passages (multiple secondary outcomes saying
    much the same thing) and crowd out every other relevant study -- exactly
    what NCT05211375 does in the real corpus, with 14 secondary-outcome chunks
    alone.
    """
    kept: list[RetrievedChunk] = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        count = counts.get(chunk.nct_id, 0)
        if count >= max_per_study:
            continue
        counts[chunk.nct_id] = count + 1
        kept.append(chunk)
    return kept


@dataclass
class Reranker:
    """Rate-limited, retrying wrapper over Voyage's rerank endpoint."""

    model: str = "rerank-2.5-lite"
    max_attempts: int = 5
    api_key: str | None = None
    rate_limit_rpm: float = 3.0

    def __post_init__(self) -> None:
        import voyageai

        self._client = voyageai.AsyncClient(api_key=self.api_key, max_retries=0)
        self._bucket = TokenBucket(self.rate_limit_rpm, capacity=1)

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        top_n: int = 8,
        max_per_study: int = 3,
    ) -> list[RetrievedChunk]:
        """Rerank ``candidates``, apply the diversity cap, then truncate to
        ``top_n``.

        An empty candidate list is a legitimate outcome (an impossible filter
        combination, or a corpus with no matches) and returns empty rather
        than making an API call with no documents.
        """
        if not candidates:
            return []

        import voyageai.error as voyage_error

        documents = [_rerank_text(c) for c in candidates]

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            retry=retry_if_exception_type(
                (voyage_error.RateLimitError, voyage_error.ServiceUnavailableError, TimeoutError)
            ),
            reraise=True,
        ):
            with attempt:
                await self._bucket.acquire()
                result = await self._client.rerank(query, documents, model=self.model)

        reranked = sorted(result.results, key=lambda r: r.relevance_score, reverse=True)
        scored = [
            candidates[r.index].model_copy(update={"rerank_score": r.relevance_score})
            for r in reranked
        ]
        capped = apply_diversity_cap(scored, max_per_study=max_per_study)
        top = capped[:top_n]
        return [
            chunk.model_copy(update={"final_rank": rank}) for rank, chunk in enumerate(top, start=1)
        ]

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()
