"""Wallet protection for a public LLM endpoint.

Three independent gates, all checked before ``/v1/query`` spends anything on
a generation call: a per-IP sliding-window rate limit (fails fast, does not
block the request), a per-IP daily query quota, and a server-wide daily spend
circuit breaker. None of these gate ``/v1/search`` -- retrieval alone costs
only a Voyage embed call, and degrading to it is exactly what the breaker is
for.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from fastapi import Request
from pydantic import BaseModel

from trialrag.config import Settings
from trialrag.db.pool import Database
from trialrag.ratelimit import TokenBucket


def hash_client(request: Request) -> str:
    """Hash of the client's remote address -- never the raw IP.

    Matches ``query_log.client_hash``'s documented contract ("hashed, never
    raw IP"). Truncated to 16 hex chars: this only needs to distinguish
    clients from each other for rate-limiting and quota purposes, not to be a
    cryptographically strong identifier.
    """
    host = request.client.host if request.client else "unknown"
    return hashlib.sha256(host.encode()).hexdigest()[:16]


@dataclass
class RateLimiter:
    """Per-client-hash token buckets, keyed lazily.

    Single-process only: each API instance holds its own buckets, so a client
    could get ``rate_limit_per_minute`` per instance under a multi-instance
    deployment. Acceptable at the current single-instance scale; revisit
    (shared store, e.g. Redis) only if that ever changes -- not before.

    Also unbounded: a bucket is never evicted once created, so a long-running
    process accumulates one entry per distinct client ever seen. At the
    traffic this project targets (hundreds of queries/month) that's a
    trivially small dict, not worth the complexity of an eviction policy yet.
    """

    rate_per_minute: float
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)

    async def allow(self, client_hash: str) -> bool:
        bucket = self._buckets.get(client_hash)
        if bucket is None:
            bucket = TokenBucket(self.rate_per_minute)
            self._buckets[client_hash] = bucket
        return await bucket.try_acquire()


class BudgetStatus(BaseModel):
    ok: bool
    spent_usd: float
    limit_usd: float


async def check_budget(db: Database, settings: Settings) -> BudgetStatus:
    """Has today's total generation spend crossed the daily ceiling?

    Reads directly from ``query_log`` rather than an in-process counter: the
    breaker must hold across process restarts and (eventually) multiple API
    instances, and ``query_log.cost_usd`` is already the authoritative record
    of what was spent.
    """
    spent = await db.fetchval(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM query_log WHERE ts >= date_trunc('day', now())"
    )
    spent_usd = float(spent)
    return BudgetStatus(
        ok=spent_usd < settings.max_daily_spend_usd,
        spent_usd=spent_usd,
        limit_usd=settings.max_daily_spend_usd,
    )


async def check_daily_quota(db: Database, client_hash: str, settings: Settings) -> bool:
    """Has this client already used its daily query allowance?"""
    count = await db.fetchval(
        "SELECT COUNT(*) FROM query_log WHERE client_hash = $1 AND ts >= date_trunc('day', now())",
        client_hash,
    )
    return int(count) < settings.per_ip_daily_queries
