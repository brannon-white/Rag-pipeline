"""Run the golden dataset through hybrid_search and score it.

Deliberately bypasses the LLM query-parser (retrieval/filters.py): each
question already carries the filter it was constructed under (see
``evals/dataset.py``), so this measures the search mechanism -- embedding +
fusion + metadata pre-filter -- in isolation from query-parsing quality. The
two are evaluated separately on purpose; conflating them would leave "was
retrieval bad" and "was the query misread" indistinguishable in one number.

The unanswerable category is scored separately from Recall/MRR/nDCG: its gold
set is empty by construction, so those metrics are undefined for it (see
metrics.py). What's measured instead is whether the system stays unconfident
-- a low top-1 fusion score -- when there is nothing correct to return.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from evals.dataset import EvalQuestion, load_dataset
from evals.metrics import aggregate, score_one
from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.config import get_settings
from trialrag.db.pool import Database
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.filters import QueryFilters
from trialrag.retrieval.service import DEFAULT_CONFIG, RetrievalConfig, hybrid_search

DEFAULT_DATASET = Path("evals/datasets/golden.json")


def _filters_from_dict(raw: dict[str, object]) -> QueryFilters:
    return QueryFilters.model_validate(raw)


def _dedup_nct_ids(nct_ids: list[str]) -> list[str]:
    """Rank-ordered nct_ids with duplicates removed, first occurrence kept.

    A study contributes several chunks to one result list; recall/MRR/nDCG
    need the *first* rank at which the right study appears, not every
    occurrence, or a study with many matching chunks would look artificially
    well-ranked by repetition alone.
    """
    seen: dict[str, None] = {}
    for nct_id in nct_ids:
        seen.setdefault(nct_id, None)
    return list(seen)


async def evaluate_one(
    db: Database,
    embedder: Embedder,
    question: EvalQuestion,
    config: RetrievalConfig,
) -> dict[str, Any]:
    vector = await embedder.embed_query(question.query_text)
    results = await hybrid_search(
        db,
        query_vector=vector,
        search_query=question.query_text,
        filters=_filters_from_dict(question.filters),
        config=config,
    )
    ranked_nct_ids = _dedup_nct_ids([r.nct_id for r in results])

    row: dict[str, Any] = {
        "id": question.id,
        "query_type": question.query_type,
        "top1_rrf_score": results[0].rrf_score if results else 0.0,
        "n_results": len(results),
    }
    if question.query_type != "unanswerable":
        row.update(score_one(ranked_nct_ids, question.gold_nct_ids))
    return row


async def run_eval(
    db: Database,
    embedder: Embedder,
    questions: list[EvalQuestion],
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> list[dict[str, Any]]:
    return [await evaluate_one(db, embedder, q, config) for q in questions]


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Metrics overall and sliced by query_type.

    A single blended number hides exactly the failure mode that matters --
    e.g. fact lookups scoring well while multi-hop questions quietly don't.
    """
    scored_rows = [r for r in rows if r["query_type"] != "unanswerable"]
    unanswerable_rows = [r for r in rows if r["query_type"] == "unanswerable"]

    summary: dict[str, dict[str, float]] = {
        "overall": aggregate(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("id", "query_type", "top1_rrf_score", "n_results")
                }
                for r in scored_rows
            ]
        )
    }
    for query_type in sorted({r["query_type"] for r in scored_rows}):
        subset = [
            {
                k: v
                for k, v in r.items()
                if k not in ("id", "query_type", "top1_rrf_score", "n_results")
            }
            for r in scored_rows
            if r["query_type"] == query_type
        ]
        summary[query_type] = aggregate(subset)

    if unanswerable_rows:
        summary["unanswerable"] = {
            "mean_top1_rrf_score": sum(r["top1_rrf_score"] for r in unanswerable_rows)
            / len(unanswerable_rows),
            "n_returned_any_result": sum(1 for r in unanswerable_rows if r["n_results"] > 0)
            / len(unanswerable_rows),
        }
    return summary


def _git_sha() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        # Fully static argv, no untrusted input -- resolved binary path plus
        # two literal arguments.
        return subprocess.check_output(  # noqa: S603
            [git, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


async def _main(dataset_path: Path, output_path: Path, *, write_run: bool) -> None:
    settings = get_settings()
    db = await Database(settings).connect()
    embedder = Embedder(
        model=settings.embed_model,
        dim=settings.embed_dim,
        api_key=settings.voyage_api_key.get_secret_value() or None,
        rate_limit_rpm=settings.voyage_rate_limit_rpm,
    )

    questions = load_dataset(dataset_path)
    rows = await run_eval(db, embedder, questions)
    summary = summarise(rows)

    text = json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n"
    await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(output_path.write_text, text, encoding="utf-8")

    if write_run:
        await db.execute(
            "INSERT INTO eval_runs (git_sha, config, metrics) VALUES ($1, $2, $3)",
            _git_sha(),
            json.dumps({"n_questions": len(questions)}),
            json.dumps(summary),
        )

    print(f"{len(questions)} questions evaluated -> {output_path}\n")
    for slice_name, metrics in summary.items():
        print(f"[{slice_name}]")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.3f}")
        print()

    await db.close()
    await embedder.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.json"))
    parser.add_argument("--write-run", action="store_true", help="Also record to eval_runs")
    args = parser.parse_args()
    asyncio.run(_main(args.dataset, args.output, write_run=args.write_run))
