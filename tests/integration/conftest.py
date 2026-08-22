"""Fixtures for tests that need a real Postgres.

Targets the docker-compose instance (``make db-up``) by default. Every test in
this package is marked ``integration`` and skipped unless a database is
reachable, so ``pytest`` with no local Postgres running still passes -- only
``make test-integration`` (or CI's dedicated job, against an ephemeral Neon
branch) requires one.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import asyncpg
import pytest

from trialrag.config import Settings
from trialrag.db.migrate import migrate
from trialrag.db.pool import Database

TEST_DSN = os.environ.get(
    "TRIALRAG_TEST_DATABASE_URL", "postgresql://trialrag:trialrag@localhost:5432/trialrag"
)


def _database_reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(TEST_DSN)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 5432), timeout=1):
            return True
    except OSError:
        return False


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    """A connected :class:`Database` against a freshly-migrated, empty schema.

    Each test gets a clean ``public`` schema rather than a truncate, so a test
    that (accidentally) changes the schema cannot leak into the next one.
    Skips (not errors) when no Postgres is reachable, so the default `pytest`
    invocation with no local database running still passes -- only
    ``make test-integration`` (or CI's job against an ephemeral Neon branch)
    is expected to actually exercise these.
    """
    if not _database_reachable():
        pytest.skip(
            "no Postgres reachable at TRIALRAG_TEST_DATABASE_URL / localhost:5432 "
            "(run `make db-up`)"
        )

    conn = await asyncpg.connect(TEST_DSN)
    try:
        await conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        await migrate(conn)
    finally:
        await conn.close()

    database = Database(Settings(database_url=TEST_DSN, db_pool_min_size=1, db_pool_max_size=4))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()
