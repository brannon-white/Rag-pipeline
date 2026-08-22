"""Request/response models for the HTTP surface.

``RetrievedChunk``, ``Study`` etc. from ``domain.models`` are returned
directly where they already carry exactly what a route needs -- no value in
a parallel schema that would just re-declare the same fields and drift from
it over time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from trialrag.domain.models import RetrievedChunk
from trialrag.retrieval.filters import QueryFilters


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class SearchResponse(BaseModel):
    results: list[RetrievedChunk]
    filters: QueryFilters
    search_query: str


class QueryRejection(BaseModel):
    """Returned as a plain JSON response (never an SSE stream) when a query
    is stopped before any generation call is made -- rate limit, daily quota,
    the spend circuit breaker, or the off-topic classifier."""

    reason: Literal["rate_limited", "quota_exceeded", "budget_exceeded", "off_topic"]
    detail: str
    retry_search: bool = Field(
        description="True when /v1/search is still usable for this request "
        "(always true for budget_exceeded; false for the others, which block "
        "generation and retrieval alike)."
    )


class FeedbackRequest(BaseModel):
    query_log_id: int
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: int


class ReadyStatus(BaseModel):
    database: bool


class StatsResponse(BaseModel):
    window_hours: int
    query_count: int
    abstention_rate: float
    cost_per_1k_queries_usd: float
    latency_p50_ms: dict[str, float]
    latency_p95_ms: dict[str, float]
