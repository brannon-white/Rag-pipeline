"""Database access.

Plain SQL over asyncpg, no ORM. The system's central query is a multi-CTE
hybrid retrieval statement that fuses an HNSW scan with a GIN full-text scan
under a shared metadata filter; expressing that through an ORM would mean
writing the SQL anyway and then hiding it. Migrations are numbered ``.sql``
files (see :mod:`trialrag.db.migrate`) for the same reason -- Alembic's value is
autogeneration from ORM models we do not have.
"""

from trialrag.db.pool import Database, get_database

__all__ = ["Database", "get_database"]
