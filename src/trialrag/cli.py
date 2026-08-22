"""Operator CLI.

Thin: every command delegates to the same functions the test suite exercises
(`trialrag.db.migrate`, `trialrag.ingest.*`). Nothing here is pipeline logic --
it is argument parsing, progress reporting and process exit codes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg
import structlog
import typer
from rich.console import Console
from rich.table import Table

from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.config import get_settings
from trialrag.db.migrate import MigrationError, migrate
from trialrag.db.pool import Database
from trialrag.ingest.chunk import ChunkingConfig, chunk_study
from trialrag.ingest.embed import Embedder
from trialrag.ingest.fetch import CtGovClient, CtGovError, StudyFilter
from trialrag.ingest.load import upsert_chunks, upsert_studies, write_manifest
from trialrag.ingest.parse import ParseError, iter_sections, parse_study
from trialrag.ingest.tokens import get_counter

app = typer.Typer(add_completion=False, help="TrialRAG operator CLI.")
console = Console()
logger = structlog.get_logger(__name__)

DEFAULT_MANIFEST = Path("docs/corpus_manifest.json")

# The condition areas the bounded corpus draws from (see docs/DESIGN.md's
# capacity estimate). A CLI flag overrides this for smoke runs and ablations.
DEFAULT_CONDITIONS = (
    "Type 2 Diabetes",
    "Breast Cancer",
    "Hypertension",
    "Asthma",
    "Major Depressive Disorder",
    "Rheumatoid Arthritis",
    "Chronic Kidney Disease",
    "COPD",
)


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer()
            if settings.log_format == "console"
            else structlog.processors.JSONRenderer(),
        ]
    )


@app.command()
def migrate_db(
    dry_run: bool = typer.Option(False, help="List pending migrations without applying them."),
) -> None:
    """Apply pending SQL migrations."""
    _configure_logging()

    async def run() -> None:
        settings = get_settings()
        conn = await asyncpg.connect(settings.asyncpg_dsn)
        try:
            applied = await migrate(conn, dry_run=dry_run)
        finally:
            await conn.close()

        verb = "would apply" if dry_run else "applied"
        if applied:
            console.print(
                f"[green]{verb} {len(applied)} migration(s)[/]: "
                f"{', '.join(f'{m.version:04d}_{m.name}' for m in applied)}"
            )
        else:
            console.print("[dim]nothing to do -- schema is up to date[/]")

    try:
        asyncio.run(run())
    except MigrationError as exc:
        console.print(f"[red]migration error:[/] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def ingest(
    limit: int | None = typer.Option(
        None, help="Stop after this many studies (smoke runs; omit for a full sync)."
    ),
    conditions: list[str] = typer.Option(  # noqa: B008 - Typer's documented pattern
        list(DEFAULT_CONDITIONS), help="Condition areas to pull (repeatable)."
    ),
    phases: list[str] = typer.Option(  # noqa: B008
        ["PHASE2", "PHASE3", "PHASE4"], help="Study phases to include."
    ),
    statuses: list[str] = typer.Option(  # noqa: B008
        ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"], help="Overall statuses to include."
    ),
    manifest_path: Path = typer.Option(  # noqa: B008
        DEFAULT_MANIFEST, help="Where to write corpus_manifest.json."
    ),
) -> None:
    """Run the full pipeline: fetch -> parse -> chunk -> embed -> load.

    Incremental by construction: studies whose ``lastUpdatePostDate`` matches
    what is already stored are skipped before parsing or embedding, so a
    re-run after the first full sync costs almost nothing.
    """
    _configure_logging()
    settings = get_settings()

    async def run() -> None:
        db = await Database(settings).connect()
        counter = get_counter(settings.embed_model)
        embedder = Embedder(
            model=settings.embed_model,
            dim=settings.embed_dim,
            batch_size=settings.embed_batch_size,
            api_key=settings.voyage_api_key.get_secret_value() or None,
            rate_limit_rpm=settings.voyage_rate_limit_rpm,
        )
        study_filter = StudyFilter(conditions=conditions, phases=phases, statuses=statuses)

        known_versions = {
            row["nct_id"]: row["last_update_posted"].isoformat()
            for row in await db.fetch(
                "SELECT nct_id, last_update_posted FROM studies "
                "WHERE last_update_posted IS NOT NULL"
            )
        }

        # Seeded with real vectors, not a presence-only placeholder: a chunk
        # whose hash is cached but whose vector is missing from this map would
        # be silently dropped from its OWN study's write by upsert_chunks (it
        # treats "no vector" as "no embedding", not "already embedded
        # elsewhere"). Boilerplate criteria ("Signed informed consent...")
        # recur verbatim across thousands of protocols, so this is the exact
        # case the cache exists for -- and getting it wrong here means those
        # chunks vanish from every study after the first one that used them.
        # One-time cost at ~45K chunks x 512 floats (~90MB); a batch job, not
        # the request path.
        existing_hashes = {
            # The pool registers the pgvector codec on every connection (see
            # trialrag.db.pool._init_connection), so `embedding` decodes
            # straight to a HalfVector here -- no manual binary handling needed.
            row["content_hash"]: row["embedding"].to_list()
            for row in await db.fetch("SELECT content_hash, embedding FROM chunks")
        }

        studies_seen = studies_written = chunks_written = parse_errors = 0
        chunking_config = ChunkingConfig()

        try:
            async with CtGovClient(
                rate_limit_rpm=settings.ctgov_rate_limit_rpm,
                page_size=settings.ctgov_page_size,
                base_url=settings.ctgov_base_url,
            ) as client:
                total = await client.count(study_filter)
                console.print(f"[bold]{total}[/] studies match the corpus filter")

                async for record in client.iter_studies(
                    study_filter, limit=limit, known_versions=known_versions
                ):
                    studies_seen += 1
                    try:
                        study = parse_study(record)
                    except ParseError as exc:
                        parse_errors += 1
                        logger.warning("parse_error", error=str(exc))
                        continue

                    candidates = chunk_study(
                        iter_sections(record, study), study, counter, chunking_config
                    )
                    vectors = await embedder.embed_chunks(candidates, cache=existing_hashes)
                    for digest in vectors:
                        existing_hashes[digest] = [0.0]

                    written = await upsert_studies(db, [study])
                    studies_written += written
                    chunk_stats = await upsert_chunks(
                        db,
                        study.nct_id,
                        candidates,
                        vectors,
                        model=settings.embed_model,
                        dim=settings.embed_dim,
                    )
                    chunks_written += chunk_stats.chunks_written

                    if chunk_stats.embeddings_missing:
                        logger.warning(
                            "chunks_missing_embeddings",
                            nct_id=study.nct_id,
                            count=len(chunk_stats.embeddings_missing),
                        )

                    if studies_seen % 50 == 0:
                        console.print(
                            f"  {studies_seen} seen | {studies_written} written | "
                            f"{chunks_written} chunks | embed hit-rate "
                            f"{embedder.stats.hit_rate:.0%}"
                        )
        except CtGovError as exc:
            console.print(f"[red]registry error:[/] {exc}")
            raise typer.Exit(1) from exc
        finally:
            await embedder.aclose()

        manifest = await write_manifest(db, manifest_path)
        await db.close()

        table = Table(title="Ingest summary")
        table.add_column("metric")
        table.add_column("value", justify="right")
        for label, value in (
            ("studies seen", studies_seen),
            ("studies written", studies_written),
            ("parse errors", parse_errors),
            ("chunks written", chunks_written),
            ("embed requests", embedder.stats.requests),
            ("embed cache hit-rate", f"{embedder.stats.hit_rate:.0%}"),
            ("corpus studies (total)", manifest["study_count"]),
            ("corpus chunks (total)", manifest["chunk_count"]),
        ):
            table.add_row(label, str(value))
        console.print(table)

    asyncio.run(run())


@app.command()
def manifest(
    manifest_path: Path = typer.Option(DEFAULT_MANIFEST),  # noqa: B008
) -> None:
    """Regenerate ``corpus_manifest.json`` from the current database state."""
    _configure_logging()
    settings = get_settings()

    async def run() -> dict[str, object]:
        db = await Database(settings).connect()
        try:
            return await write_manifest(db, manifest_path)
        finally:
            await db.close()

    result = asyncio.run(run())
    console.print(f"wrote [bold]{manifest_path}[/]")
    console.print(result)


if __name__ == "__main__":
    app()
