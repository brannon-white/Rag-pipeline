"""CI regression gate: fail if retrieval quality drops vs the committed baseline.

Runs the same golden-dataset harness as ``retrieval_eval.py`` against whatever
database ``TRIALRAG_DATABASE_URL`` currently points at (an ephemeral Neon
branch in CI, seeded from a parent branch that holds the real corpus), then
compares ``summary.overall[metric]`` against ``evals/results/baseline.json``.
Exits non-zero -- failing the PR -- if the metric drops by more than
``threshold``.

Deliberately reuses ``run_eval``/``summarise`` rather than re-implementing
scoring: this must measure exactly what ``retrieval_eval.py`` measures, or a
gate and a baseline computed by different code paths would drift apart
silently.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evals.dataset import load_dataset
from evals.retrieval_eval import DEFAULT_DATASET, run_eval, summarise
from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.config import get_settings
from trialrag.db.pool import Database
from trialrag.ingest.embed import Embedder

DEFAULT_BASELINE = Path("evals/results/baseline.json")
DEFAULT_METRIC = "recall@10"
DEFAULT_THRESHOLD = 0.02


async def _current_overall(dataset_path: Path) -> dict[str, float]:
    settings = get_settings()
    db = await Database(settings).connect()
    embedder = Embedder(
        model=settings.embed_model,
        dim=settings.embed_dim,
        api_key=settings.voyage_api_key.get_secret_value() or None,
        rate_limit_rpm=settings.voyage_rate_limit_rpm,
    )
    try:
        questions = load_dataset(dataset_path)
        rows = await run_eval(db, embedder, questions)
        return summarise(rows)["overall"]
    finally:
        await db.close()
        await embedder.aclose()


def _main(dataset_path: Path, baseline_path: Path, metric: str, threshold: float) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_value = baseline["summary"]["overall"][metric]

    current = asyncio.run(_current_overall(dataset_path))
    current_value = current[metric]

    drop = baseline_value - current_value
    print(f"{metric}: baseline={baseline_value:.4f} current={current_value:.4f} drop={drop:.4f}")

    if drop > threshold:
        print(
            f"REGRESSION: {metric} dropped by {drop:.4f}, exceeding the "
            f"{threshold:.4f} threshold ({baseline_path})",
            file=sys.stderr,
        )
        return 1

    print(f"OK: within the {threshold:.4f} threshold")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    sys.exit(_main(args.dataset, args.baseline, args.metric, args.threshold))
