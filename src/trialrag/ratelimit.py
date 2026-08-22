"""A reusable async token-bucket rate limiter.

Originally written for the ClinicalTrials.gov client, then needed again for
Voyage: an unpaid Voyage account is capped at **3 requests/minute** on both the
embeddings and reranking endpoints (verified live -- a `trialrag ingest` run
against 80 studies failed outright with ``RateLimitError`` once several
studies' embed calls landed close together). Reactive retry-on-429 cannot
survive a ceiling that low; the client has to pace itself proactively instead.
Promoted here so every outbound client in the ingestion and retrieval paths
shares one implementation rather than three near-identical copies.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    """Async token bucket.

    Chosen over a fixed sleep between calls because it permits a short burst
    after an idle stretch while still holding the long-run average under the
    limit -- which is the shape of both a resumed ingest and a bursty retrieval
    workload.
    """

    def __init__(self, rate_per_minute: float, *, capacity: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate = rate_per_minute / 60.0
        self._capacity = capacity if capacity is not None else max(1.0, rate_per_minute / 10.0)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_for = deficit / self._rate
            await asyncio.sleep(wait_for)
