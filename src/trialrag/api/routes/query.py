"""POST /v1/query -- the full pipeline, streamed as SSE.

Gates run cheapest/fastest first, all *before* the SSE stream opens: rate
limit -> daily quota -> daily spend circuit breaker -> off-topic
classification. Anything that fails there returns a plain JSON
``QueryRejection``, never a stream -- a client shouldn't have to open an
event stream just to learn it was rejected. Only once all four pass does the
real pipeline run: parse -> embed -> hybrid search -> rerank -> generate.

The circuit breaker gates this route only. ``/v1/search`` never checks it --
that is the whole point of degrading to retrieval-only rather than going
fully dark when the daily budget trips.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from trialrag.api.deps import (
    get_db,
    get_embedder,
    get_provider,
    get_query_classifier,
    get_query_parser,
    get_rate_limiter,
    get_reranker,
    get_settings_dep,
)
from trialrag.api.schemas import QueryRejection, QueryRequest
from trialrag.api.security import RateLimiter, check_budget, check_daily_quota, hash_client
from trialrag.config import Settings
from trialrag.db.pool import Database
from trialrag.domain.models import RetrievedChunk
from trialrag.generation.anthropic_provider import AnthropicProvider
from trialrag.generation.guardrails import QueryClassifier
from trialrag.generation.provider import Done
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.filters import QueryFilters, QueryParser
from trialrag.retrieval.rerank import Reranker
from trialrag.retrieval.service import hybrid_search

router = APIRouter(prefix="/v1", tags=["query"])


async def _log_query(
    db: Database,
    *,
    query_text: str,
    filters: QueryFilters,
    chunks: list[RetrievedChunk],
    latency_ms: dict[str, float],
    done: Done,
    model: str,
    trace_id: str,
    client_hash: str,
) -> int:
    row_id = await db.fetchval(
        """
        INSERT INTO query_log (
            query_text, filters, retrieved_chunk_ids, latency_ms, tokens,
            cost_usd, model, abstained, trace_id, client_hash
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING id
        """,
        query_text,
        json.dumps(filters.model_dump()),
        [c.chunk_id for c in chunks],
        json.dumps(latency_ms),
        done.usage.model_dump_json(),
        done.cost_usd,
        model,
        done.abstained,
        trace_id,
        client_hash,
    )
    return int(row_id)


def _reject(status_code: int, rejection: QueryRejection) -> Response:
    return JSONResponse(status_code=status_code, content=rejection.model_dump())


@router.post("/query")
async def query(
    body: QueryRequest,
    request: Request,
    db: Database = Depends(get_db),  # noqa: B008 - FastAPI's documented pattern
    embedder: Embedder = Depends(get_embedder),  # noqa: B008 - FastAPI's documented pattern
    reranker: Reranker = Depends(get_reranker),  # noqa: B008 - FastAPI's documented pattern
    query_parser: QueryParser = Depends(get_query_parser),  # noqa: B008 - FastAPI's documented pattern
    classifier: QueryClassifier = Depends(get_query_classifier),  # noqa: B008 - FastAPI's documented pattern
    provider: AnthropicProvider = Depends(get_provider),  # noqa: B008 - FastAPI's documented pattern
    rate_limiter: RateLimiter = Depends(get_rate_limiter),  # noqa: B008 - FastAPI's documented pattern
    settings: Settings = Depends(get_settings_dep),  # noqa: B008 - FastAPI's documented pattern
) -> Response:
    trace_id = str(getattr(request.state, "trace_id", uuid.uuid4()))
    client_hash = hash_client(request)

    if not await rate_limiter.allow(client_hash):
        return _reject(
            429,
            QueryRejection(
                reason="rate_limited", detail="Too many requests; slow down.", retry_search=False
            ),
        )

    if not await check_daily_quota(db, client_hash, settings):
        return _reject(
            429,
            QueryRejection(
                reason="quota_exceeded",
                detail="Daily query limit reached for this client.",
                retry_search=False,
            ),
        )

    budget = await check_budget(db, settings)
    if not budget.ok:
        return _reject(
            503,
            QueryRejection(
                reason="budget_exceeded",
                detail="Daily spend limit reached. Retrieval search still works.",
                retry_search=True,
            ),
        )

    classification = await classifier.classify(body.query)
    if not classification.on_topic:
        return _reject(
            400,
            QueryRejection(reason="off_topic", detail=classification.reason, retry_search=False),
        )

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        timings: dict[str, float] = {}

        t0 = time.monotonic()
        parsed = await query_parser.parse(body.query)
        timings["parse"] = (time.monotonic() - t0) * 1000

        t1 = time.monotonic()
        vector = await embedder.embed_query(parsed.search_query)
        candidates = await hybrid_search(
            db, query_vector=vector, search_query=parsed.search_query, filters=parsed.filters
        )
        chunks = await reranker.rerank(
            parsed.search_query,
            candidates,
            top_n=settings.rerank_top_n,
            max_per_study=settings.max_chunks_per_study,
        )
        timings["retrieve"] = (time.monotonic() - t1) * 1000

        yield {"event": "trace", "data": json.dumps({"trace_id": trace_id})}

        t2 = time.monotonic()
        done: Done | None = None
        async for gen_event in provider.generate_stream(
            body.query,
            chunks,
            model=settings.answer_model,
            effort=settings.answer_effort,
            max_tokens=settings.answer_max_tokens,
        ):
            yield {"event": gen_event.type, "data": gen_event.model_dump_json()}
            if isinstance(gen_event, Done):
                done = gen_event
        timings["generate"] = (time.monotonic() - t2) * 1000

        if done is not None:
            query_log_id = await _log_query(
                db,
                query_text=body.query,
                filters=parsed.filters,
                chunks=chunks,
                latency_ms=timings,
                done=done,
                model=settings.answer_model,
                trace_id=trace_id,
                client_hash=client_hash,
            )
            yield {"event": "logged", "data": json.dumps({"query_log_id": query_log_id})}

    return EventSourceResponse(event_stream())
