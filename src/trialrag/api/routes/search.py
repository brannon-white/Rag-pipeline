"""Retrieval-only search: no generation, no cost/rate gating.

Powers both the eval harness (structurally -- ``evals/retrieval_eval.py``
calls ``hybrid_search`` directly rather than through HTTP, but this route is
the same pipeline) and the frontend's "retrieval debug" toggle, which is why
every per-arm score already sitting on ``RetrievedChunk`` is returned as-is
rather than trimmed down.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from trialrag.api.deps import get_db, get_embedder, get_query_parser, get_reranker, get_settings_dep
from trialrag.api.schemas import SearchRequest, SearchResponse
from trialrag.config import Settings
from trialrag.db.pool import Database
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.filters import QueryParser
from trialrag.retrieval.rerank import Reranker
from trialrag.retrieval.service import hybrid_search

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    db: Database = Depends(get_db),  # noqa: B008 - FastAPI's documented pattern
    embedder: Embedder = Depends(get_embedder),  # noqa: B008 - FastAPI's documented pattern
    reranker: Reranker = Depends(get_reranker),  # noqa: B008 - FastAPI's documented pattern
    query_parser: QueryParser = Depends(get_query_parser),  # noqa: B008 - FastAPI's documented pattern
    settings: Settings = Depends(get_settings_dep),  # noqa: B008 - FastAPI's documented pattern
) -> SearchResponse:
    parsed = await query_parser.parse(body.query)
    vector = await embedder.embed_query(parsed.search_query)
    candidates = await hybrid_search(
        db, query_vector=vector, search_query=parsed.search_query, filters=parsed.filters
    )
    results = await reranker.rerank(
        parsed.search_query,
        candidates,
        top_n=settings.rerank_top_n,
        max_per_study=settings.max_chunks_per_study,
    )
    return SearchResponse(results=results, filters=parsed.filters, search_query=parsed.search_query)
