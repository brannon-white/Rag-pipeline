"""Forward-only SQL migrations.

Numbered ``migrations/NNNN_name.sql`` files, applied in order, each inside a
transaction, each recorded in ``schema_migrations`` with a checksum. This is the
``golang-migrate``/``dbmate`` model rather than Alembic's, because Alembic's
main draw is autogenerating diffs from ORM models, and this project has no ORM
(ADR-002). What is left is a version table and an ordered replay, which is
sixty lines.

Two properties worth stating, since they are what make this safe to run from
CI against a shared database:

* **Checksummed.** An already-applied file that changes on disk is an error,
  not a no-op. Editing applied migrations is how two environments silently
  diverge.
* **Advisory-locked.** Concurrent deploys serialise instead of racing to create
  the same index.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

# Arbitrary but fixed: any value works so long as every deployer uses the same.
_LOCK_KEY = 0x7B1A_16A6

_FILENAME_RE = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     integer     PRIMARY KEY,
    name        text        NOT NULL,
    checksum    text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


class MigrationError(RuntimeError):
    """A migration could not be applied, or the on-disk set is inconsistent."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    path: Path

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()[:16]


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Load and order migrations, rejecting malformed or duplicated versions."""
    if not directory.is_dir():
        raise MigrationError(f"migrations directory not found: {directory}")

    found: dict[int, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if match is None:
            raise MigrationError(
                f"{path.name}: expected NNNN_snake_case_name.sql (four digits, then a name)"
            )
        version = int(match["version"])
        if version in found:
            # Two people numbering 0003 on separate branches apply in an order
            # that depends on filename sort. Fail loudly at the merge instead.
            raise MigrationError(
                f"duplicate migration version {version:04d}: "
                f"{found[version].path.name} and {path.name}"
            )
        found[version] = Migration(
            version=version,
            name=match["name"],
            sql=path.read_text(encoding="utf-8"),
            path=path,
        )
    return [found[v] for v in sorted(found)]


async def applied_versions(conn: asyncpg.Connection) -> dict[int, str]:
    """``version -> checksum`` for migrations already recorded."""
    await conn.execute(_BOOTSTRAP)
    rows = await conn.fetch("SELECT version, checksum FROM schema_migrations")
    return {row["version"]: row["checksum"] for row in rows}


async def migrate(
    conn: asyncpg.Connection,
    *,
    directory: Path = MIGRATIONS_DIR,
    dry_run: bool = False,
) -> list[Migration]:
    """Apply every pending migration in order. Returns those applied.

    Raises:
        MigrationError: An applied migration's file has since been edited.
    """
    migrations = discover(directory)

    # Serialise concurrent deployers. Released when the session ends, so a
    # crashed deploy does not wedge the next one.
    await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_KEY)
    try:
        already = await applied_versions(conn)

        for migration in migrations:
            recorded = already.get(migration.version)
            if recorded is not None and recorded != migration.checksum:
                raise MigrationError(
                    f"{migration.path.name} was modified after being applied "
                    f"(recorded {recorded}, on disk {migration.checksum}). "
                    "Migrations are immutable once applied -- add a new one instead."
                )

        pending = [m for m in migrations if m.version not in already]
        if dry_run:
            return pending

        for migration in pending:
            logger.info("applying migration %04d_%s", migration.version, migration.name)
            # One transaction per migration: a failure leaves the database at
            # the last good version rather than half-way through this one.
            async with conn.transaction():
                await conn.execute(migration.sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name, checksum) VALUES ($1, $2, $3)",
                    migration.version,
                    migration.name,
                    migration.checksum,
                )
        return pending
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
