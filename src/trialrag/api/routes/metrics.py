"""GET /metrics -- Prometheus text exposition.

The counter/histogram objects live here; ``app.py``'s request middleware
records to them on every request, since that's the one place that sees every
route uniformly.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter(tags=["metrics"])

REQUEST_COUNT = Counter(
    "trialrag_requests_total", "Requests by route and status code", ["route", "status"]
)
REQUEST_LATENCY_SECONDS = Histogram(
    "trialrag_request_latency_seconds", "Request latency by route", ["route"]
)


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
