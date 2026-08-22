"""Token-bucket rate limiter tests.

Shared by the ClinicalTrials.gov client and the Voyage embedder/reranker --
the latter needed it after a real ingest run against an unpaid Voyage account
(hard-capped at 3 req/min) failed outright with ``RateLimitError`` because
nothing paced requests proactively; see ``trialrag/ratelimit.py``'s docstring.
Tested against the event loop's own clock rather than by sleeping for real
durations, so the suite stays fast.
"""

from __future__ import annotations

import asyncio

import pytest

from trialrag.ratelimit import TokenBucket


async def test_allows_initial_burst() -> None:
    bucket = TokenBucket(60, capacity=5)
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(5):
        await bucket.acquire()
    assert loop.time() - start < 0.1


async def test_throttles_beyond_capacity() -> None:
    # 600/min = 10/s; capacity 1 means the 2nd token costs ~100ms.
    bucket = TokenBucket(600, capacity=1)
    loop = asyncio.get_running_loop()
    await bucket.acquire()
    start = loop.time()
    await bucket.acquire()
    assert loop.time() - start >= 0.05


async def test_is_concurrency_safe() -> None:
    """Parallel callers must not oversubscribe the shared budget."""
    bucket = TokenBucket(600, capacity=2)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(bucket.acquire() for _ in range(10)))
    # 2 free + 8 at 10/s -> at least ~0.7s of enforced spacing.
    assert loop.time() - start >= 0.5


def test_rejects_nonsense_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        TokenBucket(0)


async def test_try_acquire_succeeds_within_capacity() -> None:
    bucket = TokenBucket(60, capacity=2)
    assert await bucket.try_acquire() is True
    assert await bucket.try_acquire() is True


async def test_try_acquire_fails_without_blocking_when_exhausted() -> None:
    bucket = TokenBucket(60, capacity=1)
    assert await bucket.try_acquire() is True
    loop = asyncio.get_running_loop()
    start = loop.time()
    assert await bucket.try_acquire() is False
    assert loop.time() - start < 0.05


async def test_wait_scales_with_how_far_under_capacity_a_low_rate_sits() -> None:
    """Regression guard: the prod incident was Voyage's 3 req/min ceiling --
    a rate far too low for reactive retry-on-429 to survive. Uses a scaled-up
    rate (120/min, i.e. the same 1-token-per-2-request-interval shape as 3/min
    at 1/20th the wall-clock) so the property is checked in well under a
    second rather than by actually waiting out a real 3/min budget.
    """
    bucket = TokenBucket(120, capacity=1)  # 2 req/s -> 2nd acquire waits ~0.5s
    loop = asyncio.get_running_loop()
    await bucket.acquire()
    start = loop.time()
    await bucket.acquire()
    assert loop.time() - start >= 0.4
