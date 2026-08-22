"""FastAPI app: async throughout, one process-wide instance of every
downstream client, constructed in the lifespan and torn down gracefully on
shutdown.

``min_size=0`` on the DB pool (already the default in ``config.Settings``)
plus this lifespan's explicit ``close()`` calls are what let Neon actually
suspend between requests rather than being pinned awake by a lingering
connection -- see ``db/pool.py``'s module docstring for the cost model this
protects.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.api.routes import feedback, health, metrics, query, search, stats, studies
from trialrag.api.security import RateLimiter
from trialrag.config import get_settings
from trialrag.db.pool import Database
from trialrag.generation.anthropic_provider import AnthropicProvider
from trialrag.generation.guardrails import QueryClassifier
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.filters import QueryParser
from trialrag.retrieval.rerank import Reranker

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    anthropic_key = settings.anthropic_api_key.get_secret_value() or None
    voyage_key = settings.voyage_api_key.get_secret_value() or None

    db = await Database(settings).connect()
    embedder = Embedder(
        model=settings.embed_model,
        dim=settings.embed_dim,
        api_key=voyage_key,
        rate_limit_rpm=settings.voyage_rate_limit_rpm,
    )
    reranker = Reranker(
        model=settings.rerank_model, api_key=voyage_key, rate_limit_rpm=settings.voyage_rate_limit_rpm
    )
    query_parser = QueryParser(
        model=settings.query_parse_model, effort=settings.query_parse_effort, api_key=anthropic_key
    )
    query_classifier = QueryClassifier(api_key=anthropic_key)
    provider = AnthropicProvider(api_key=anthropic_key)
    rate_limiter = RateLimiter(rate_per_minute=settings.rate_limit_per_minute)

    app.state.settings = settings
    app.state.db = db
    app.state.embedder = embedder
    app.state.reranker = reranker
    app.state.query_parser = query_parser
    app.state.query_classifier = query_classifier
    app.state.provider = provider
    app.state.rate_limiter = rate_limiter

    try:
        yield
    finally:
        await db.close()
        await embedder.aclose()
        await reranker.aclose()
        await query_parser.aclose()
        await query_classifier.aclose()
        await provider.aclose()


app = FastAPI(title="TrialRAG", lifespan=lifespan)


@app.middleware("http")
async def observe_request(request: Request, call_next: RequestResponseEndpoint) -> Response:
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id
    start = time.monotonic()
    response = await call_next(request)
    metrics.REQUEST_LATENCY_SECONDS.labels(route=request.url.path).observe(
        time.monotonic() - start
    )
    metrics.REQUEST_COUNT.labels(route=request.url.path, status=str(response.status_code)).inc()
    response.headers["X-Trace-Id"] = trace_id
    return response


app.include_router(health.router)
app.include_router(search.router)
app.include_router(studies.router)
app.include_router(feedback.router)
app.include_router(query.router)
app.include_router(stats.router)
app.include_router(metrics.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
