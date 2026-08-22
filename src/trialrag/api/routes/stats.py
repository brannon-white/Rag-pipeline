"""GET /stats -- the public dashboard.

Percentiles are computed in Python from a bounded recent ``query_log`` fetch
rather than SQL ``percentile_cont`` -- simplest correct thing at the query
volumes this project targets (hundreds/month). Move server-side only if
fetching the window itself ever becomes the expensive part.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from trialrag.api.deps import get_db
from trialrag.api.schemas import StatsResponse
from trialrag.db.pool import Database

router = APIRouter(tags=["stats"])

_WINDOW_HOURS = 24


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


@router.get("/stats", response_model=StatsResponse)
async def stats(db: Database = Depends(get_db)) -> StatsResponse:  # noqa: B008
    rows = await db.fetch(
        "SELECT latency_ms, cost_usd, abstained FROM query_log "
        "WHERE ts >= now() - make_interval(hours => $1::int)",
        _WINDOW_HOURS,
    )

    if not rows:
        return StatsResponse(
            window_hours=_WINDOW_HOURS,
            query_count=0,
            abstention_rate=0.0,
            cost_per_1k_queries_usd=0.0,
            latency_p50_ms={},
            latency_p95_ms={},
        )

    stage_latencies: dict[str, list[float]] = {}
    for row in rows:
        for stage, ms in json.loads(row["latency_ms"]).items():
            stage_latencies.setdefault(stage, []).append(float(ms))

    query_count = len(rows)
    abstained_count = sum(1 for row in rows if row["abstained"])
    total_cost = sum(float(row["cost_usd"]) for row in rows)

    return StatsResponse(
        window_hours=_WINDOW_HOURS,
        query_count=query_count,
        abstention_rate=abstained_count / query_count,
        cost_per_1k_queries_usd=(total_cost / query_count) * 1000,
        latency_p50_ms={stage: _percentile(v, 0.5) for stage, v in stage_latencies.items()},
        latency_p95_ms={stage: _percentile(v, 0.95) for stage, v in stage_latencies.items()},
    )
