"""asyncpg connection pool.

The pool configuration here is load-bearing for the project's cost model, not
just its performance. Neon's free tier bills 100 compute-hours per month and
suspends compute after ~5 minutes idle; a pool holding even one connection open
keeps the meter running 24/7, which is 730 hours against a 100-hour allowance.
So ``min_size`` defaults to 0 and idle connections are reaped inside Neon's
suspend window. The matching half of that decision lives in the API layer,
where ``/healthz`` is deliberately shallow and never touches the database.

Vector values cross the wire as pgvector's ``halfvec`` type, registered once per
connection via :func:`_init_connection`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Self, cast

import asyncpg

from trialrag.config import Settings, get_settings

logger = logging.getLogger(__name__)


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup, run by asyncpg on every new connection.

    Registering the pgvector codecs here rather than at call sites is what lets
    query code pass and receive plain Python lists of floats.
    """
    from pgvector.asyncpg import register_vector

    await register_vector(conn)


class Database:
    """Owns the pool and hands out connections.

    Constructed once per process and shared. Not a singleton by construction --
    tests build isolated instances against throwaway databases -- but
    :func:`get_database` provides the process-wide one for the API.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._pool

    async def connect(self) -> Self:
        if self._pool is not None:
            return self
        settings = self._settings
        self._pool = await asyncpg.create_pool(
            dsn=settings.asyncpg_dsn,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            max_inactive_connection_lifetime=settings.db_idle_timeout_s,
            command_timeout=settings.db_command_timeout_s,
            init=_init_connection,
            # Neon routes through a pooler that does not support prepared
            # statement caching across connections; asyncpg's default cache
            # produces "prepared statement already exists" under it.
            statement_cache_size=0,
        )
        logger.info(
            "database pool ready (min=%d max=%d idle_timeout=%.0fs)",
            settings.db_pool_min_size,
            settings.db_pool_max_size,
            settings.db_idle_timeout_s,
        )
        return self

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> Self:
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # --- Convenience wrappers ------------------------------------------------

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        async with self.acquire() as conn:
            # asyncpg ships no type stubs, so every call here returns Any to
            # mypy; the casts assert the documented return shape rather than
            # papering over a real type error.
            return cast("list[asyncpg.Record]", await conn.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        async with self.acquire() as conn:
            return cast("str", await conn.execute(query, *args))

    async def executemany(self, query: str, args: Sequence[Sequence[Any]]) -> None:
        async with self.acquire() as conn:
            await conn.executemany(query, args)

    async def healthy(self) -> bool:
        """Deep health check. Used by ``/readyz``, never by ``/healthz``."""
        try:
            return bool(await self.fetchval("SELECT 1") == 1)
        except (asyncpg.PostgresError, OSError) as exc:
            logger.warning("database health check failed: %s", exc)
            return False


_DATABASE: Database | None = None


async def get_database() -> Database:
    """Process-wide pool, created on first use."""
    global _DATABASE
    if _DATABASE is None:
        _DATABASE = await Database().connect()
    return _DATABASE


async def close_database() -> None:
    global _DATABASE
    if _DATABASE is not None:
        await _DATABASE.close()
        _DATABASE = None
