"""Fixtures for tests that need a real Postgres.

Targets the docker-compose instance (``make db-up``) by default, against the
**dedicated** ``trialrag_test`` database (see ``docker/initdb/`` for how it
gets created) -- never ``trialrag``, the dev/ingest database. That separation
is not a style preference: the ``db`` fixture below runs ``DROP SCHEMA public
CASCADE`` before every test, and this suite once pointed at ``trialrag``
itself, which silently destroyed roughly 30 real API calls' worth of an
in-progress corpus ingest the moment the test suite ran. The ingest process
didn't crash -- Postgres doesn't care that the tables it's inserting into were
just recreated -- so the data loss was invisible until someone checked row
counts. ``_assert_not_the_dev_database`` exists so a misconfigured
``TRIALRAG_TEST_DATABASE_URL`` fails loudly instead of repeating that.

Every test in this package is marked ``integration`` and skipped unless a
database is reachable, so ``pytest`` with no local Postgres running still
passes -- only ``make test-integration`` (or CI's dedicated job, against an
ephemeral Neon branch) requires one.
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
    "TRIALRAG_TEST_DATABASE_URL", "postgresql://trialrag:trialrag@localhost:5432/trialrag_test"
)


def _assert_not_the_dev_database(dsn: str) -> None:
    """Refuse to run a schema-destroying fixture against the dev database.

    A cheap, specific guard against the exact incident this module's docstring
    describes: a schema-dropping test fixture pointed at ``trialrag`` by a
    misconfigured (or reverted) ``TRIALRAG_TEST_DATABASE_URL``.
    """
    if dsn.rstrip("/").rsplit("/", 1)[-1] == "trialrag":
        raise RuntimeError(
            "TRIALRAG_TEST_DATABASE_URL points at 'trialrag', the dev/ingest database. "
            "The integration suite drops and recreates its schema before every test -- "
            "running it here would destroy real ingested data. Point it at a dedicated "
            "database instead, e.g. 'trialrag_test'."
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
    _assert_not_the_dev_database(TEST_DSN)

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
