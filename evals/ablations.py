"""Ablation grid: retrieval mode x reranking, against the golden dataset.

Scope, stated honestly: this sweeps retrieval-time parameters only --
``dense_weight`` (dense-only / sparse-only / hybrid) and whether reranking
runs. Two dimensions named in the original design are deliberately *not*
swept here, for reasons worth recording rather than hiding:

* **HNSW ``ef_search``** -- checked directly with ``EXPLAIN (ANALYZE,
  BUFFERS)`` against the live corpus (432 chunks): Postgres's planner chooses
  a sequential scan over the HNSW index at this table size, because a full
  scan is cheaper than index overhead for a table this small. ``ef_search``
  only affects an actual index scan, so sweeping it right now would produce a
  flat, meaningless line -- not a finding, just noise. Revisit once the
  corpus is large enough that the planner prefers the index (a `SET
  enable_seqscan = off` probe, or simply watching `EXPLAIN` as the corpus
  grows, will show when that crossover happens).
* **Embedding dimension (1024/512/256) and chunking strategy** -- both require
  re-embedding or re-chunking the *entire* corpus per grid point. At the
  current Voyage rate-limit tier that is a multi-run, multi-hour undertaking
  (see ``docs/adr/`` for the observed ceiling), not a five-minute sweep --
  deferred to a dedicated ablation pass rather than rushed here.

Cost design: query embeddings are computed *once* per question and cached --
retrieval-time parameters never change the query embedding, only how it's
used, so re-embedding per grid point would be pure waste (and pure API
spend). Reranking, where swept, does cost one Voyage rerank call per question
per reranked config; everything else is Postgres-side and free.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.dataset import EvalQuestion, load_dataset
from evals.metrics import aggregate, score_one
from evals.retrieval_eval import DEFAULT_DATASET, _dedup_nct_ids, _filters_from_dict
from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.config import get_settings
from trialrag.db.pool import Database
from trialrag.ingest.embed import Embedder
from trialrag.retrieval.rerank import Reranker
from trialrag.retrieval.service import RetrievalConfig, hybrid_search

DEFAULT_OUTPUT = Path("docs/EVALUATION.md")


@dataclass(frozen=True)
class GridPoint:
    name: str
    retrieval_mode: str  # "dense_only" | "sparse_only" | "hybrid"
    dense_weight: float
    rerank: bool


GRID: tuple[GridPoint, ...] = (
    GridPoint("dense_only", "dense_only", dense_weight=1.0, rerank=False),
    GridPoint("sparse_only", "sparse_only", dense_weight=0.0, rerank=False),
    GridPoint("hybrid", "hybrid", dense_weight=0.5, rerank=False),
    GridPoint("hybrid+rerank", "hybrid", dense_weight=0.5, rerank=True),
)


async def _embed_all_queries(
    embedder: Embedder, questions: list[EvalQuestion]
) -> dict[str, list[float]]:
    """One embedding per question, reused across every grid point."""
    return {q.id: await embedder.embed_query(q.query_text) for q in questions}


async def _run_point(
    db: Database,
    reranker: Reranker | None,
    questions: list[EvalQuestion],
    query_vectors: dict[str, list[float]],
    point: GridPoint,
) -> dict[str, Any]:
    config = RetrievalConfig(
        dense_weight=point.dense_weight,
        # sparse_k=0 disables the lexical arm entirely (see service.py);
        # dense_k=0 does the same for the dense arm.
        sparse_k=0 if point.retrieval_mode == "dense_only" else 50,
        dense_k=0 if point.retrieval_mode == "sparse_only" else 50,
    )

    rows: list[dict[str, Any]] = []
    for question in questions:
        results = await hybrid_search(
            db,
            query_vector=query_vectors[question.id],
            search_query=question.query_text,
            filters=_filters_from_dict(question.filters),
            config=config,
        )
        if point.rerank and reranker is not None:
            results = await reranker.rerank(
                question.query_text, results, top_n=20, max_per_study=20
            )

        ranked = _dedup_nct_ids([r.nct_id for r in results])
        row: dict[str, Any] = {"id": question.id, "query_type": question.query_type}
        if question.query_type != "unanswerable":
            row.update(score_one(ranked, question.gold_nct_ids))
        rows.append(row)

    scored = [
        {k: v for k, v in r.items() if k not in ("id", "query_type")}
        for r in rows
        if r["query_type"] != "unanswerable"
    ]
    return {"point": point.name, **aggregate(scored)}


def _markdown_table(results: list[dict[str, Any]]) -> str:
    metric_keys = [k for k in results[0] if k != "point"]
    header = "| config | " + " | ".join(metric_keys) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    rows = [
        "| " + row["point"] + " | " + " | ".join(f"{row[k]:.3f}" for k in metric_keys) + " |"
        for row in results
    ]
    return "\n".join([header, sep, *rows])


_MARKDOWN_HEADER = """# Evaluation

## Retrieval ablation

Sweep of retrieval-time parameters against the golden dataset
(`evals/datasets/golden.json`), evaluated on the live corpus. See
`evals/ablations.py`'s module docstring for what is and isn't swept here, and
why.

"""

_MARKDOWN_FOOTER = """

**Reading this table:** `dense_only` and `sparse_only` isolate each arm;
`hybrid` fuses them via Reciprocal Rank Fusion at equal weight; `hybrid+rerank`
adds the Voyage `rerank-2.5-lite` cross-encoder pass on top. `unanswerable`
questions are excluded from these metrics by construction (see
`evals/metrics.py`) -- their gold set is empty, so recall/MRR/nDCG are
undefined for them; `retrieval_eval.py` reports them separately (mean top-1
fused score, a confidence check rather than a recall check).
"""


async def run_ablations(dataset_path: Path) -> list[dict[str, Any]]:
    settings = get_settings()
    db = await Database(settings).connect()
    embedder = Embedder(
        model=settings.embed_model,
        dim=settings.embed_dim,
        api_key=settings.voyage_api_key.get_secret_value() or None,
        rate_limit_rpm=settings.voyage_rate_limit_rpm,
    )
    reranker = Reranker(
        api_key=settings.voyage_api_key.get_secret_value() or None,
        rate_limit_rpm=settings.voyage_rate_limit_rpm,
    )

    try:
        questions = load_dataset(dataset_path)
        query_vectors = await _embed_all_queries(embedder, questions)
        return [await _run_point(db, reranker, questions, query_vectors, point) for point in GRID]
    finally:
        await db.close()
        await embedder.aclose()
        await reranker.aclose()


async def _main(dataset_path: Path, output_path: Path) -> None:
    results = await run_ablations(dataset_path)
    table = _markdown_table(results)

    text = _MARKDOWN_HEADER + table + _MARKDOWN_FOOTER
    await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(output_path.write_text, text, encoding="utf-8")

    print(table)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asyncio.run(_main(args.dataset, args.output))
