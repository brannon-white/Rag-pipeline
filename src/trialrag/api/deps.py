"""FastAPI ``Depends()`` wrappers around the app-wide singletons.

Everything here is constructed once in ``app.py``'s lifespan and stashed on
``app.state`` -- these functions just hand it back per-request. No new
instances are ever created per-request; a fresh ``Embedder``/``AnthropicProvider``
per request would defeat their own connection pooling and rate limiting.
"""

from __future__ import annotations

from fastapi import Request

from trialrag.api.security import RateLimiter
from trialrag.config import Settings
from trialrag.db.pool import Database
from trialrag.generation.anthropic_provider import AnthropicProvider
from trialrag.generation.guardrails import QueryClassifier
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.filters import QueryParser
from trialrag.retrieval.rerank import Reranker


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_db(request: Request) -> Database:
    return request.app.state.db  # type: ignore[no-any-return]


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder  # type: ignore[no-any-return]


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker  # type: ignore[no-any-return]


def get_query_parser(request: Request) -> QueryParser:
    return request.app.state.query_parser  # type: ignore[no-any-return]


def get_query_classifier(request: Request) -> QueryClassifier:
    return request.app.state.query_classifier  # type: ignore[no-any-return]


def get_provider(request: Request) -> AnthropicProvider:
    return request.app.state.provider  # type: ignore[no-any-return]


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter  # type: ignore[no-any-return]
