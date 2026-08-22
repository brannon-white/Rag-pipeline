"""Generate the golden retrieval-eval dataset from structured DB fields.

Every question is built by construction, not hand-labeled or LLM-generated:
the gold answer is read directly from the same ``studies`` columns the
question asks about, so "is this gold label correct" is a non-question. That
is the entire point of picking a corpus with a machine-verifiable structured
half (see README.md's "why this corpus" section).

Four question shapes:

* **fact_phase / fact_sponsor / fact_condition** -- "what is true about the
  trial titled X", answerable purely from the study's own metadata plus
  whatever chunk text lets it be *found*. Gold is the one study.
* **multi_hop** -- "which {phase} trials study {condition}", built from a real
  phase+condition co-occurrence already present in the corpus, so the gold set
  is guaranteed non-empty and the filter carried alongside the question is the
  literal filter that produced it.
* **unanswerable** -- fabricated conditions guaranteed absent from any real
  registry record. Correct retrieval behaviour is returning nothing confident,
  which is a different measurement than recall (there is nothing to recall);
  ``retrieval_eval.py`` scores this category separately.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evals.dataset import EvalQuestion, save_dataset
from trialrag import bootstrap  # noqa: F401 - import side effect: SSL trust store fix
from trialrag.config import get_settings
from trialrag.db.pool import Database

DEFAULT_OUTPUT = Path("evals/datasets/golden.json")

# Guaranteed-absent from any real ClinicalTrials.gov record -- invented terms,
# not real conditions with unusual spelling (a real-but-rare condition risks a
# false "unanswerable" if the corpus happens to contain it).
UNANSWERABLE_CONDITIONS = (
    "Xyloquinamine-9 Deficiency Syndrome",
    "Zorbitant-Resistant Fictivemia",
    "Quenlar's Paradoxical Neurocytopathy",
)


async def _fact_questions(db: Database) -> list[EvalQuestion]:
    rows = await db.fetch(
        "SELECT nct_id, brief_title, lead_sponsor, phases, conditions "
        "FROM studies WHERE brief_title IS NOT NULL"
    )
    questions: list[EvalQuestion] = []

    for row in rows:
        title = row["brief_title"]

        if row["phases"]:
            questions.append(
                EvalQuestion(
                    id=f"fact_phase:{row['nct_id']}",
                    query_type="fact_phase",
                    query_text=f'What phase is the clinical trial titled "{title}"?',
                    gold_nct_ids=[row["nct_id"]],
                    source_fields={"phases": row["phases"]},
                )
            )

        if row["lead_sponsor"]:
            questions.append(
                EvalQuestion(
                    id=f"fact_sponsor:{row['nct_id']}",
                    query_type="fact_sponsor",
                    query_text=f'Who is the lead sponsor of the trial titled "{title}"?',
                    gold_nct_ids=[row["nct_id"]],
                    source_fields={"lead_sponsor": row["lead_sponsor"]},
                )
            )

        if row["conditions"]:
            questions.append(
                EvalQuestion(
                    id=f"fact_condition:{row['nct_id']}",
                    query_type="fact_condition",
                    query_text=f'What medical condition does the trial titled "{title}" study?',
                    gold_nct_ids=[row["nct_id"]],
                    source_fields={"conditions": row["conditions"]},
                )
            )

    return questions


async def _multi_hop_questions(db: Database, *, max_pairs: int = 15) -> list[EvalQuestion]:
    # Real (phase, condition) co-occurrences already in the corpus, most
    # common first, so sampling `max_pairs` favours pairs with enough
    # supporting studies to make "recall" a meaningful signal.
    rows = await db.fetch(
        """
        SELECT phase, condition, array_agg(nct_id ORDER BY nct_id) AS nct_ids
        FROM studies, unnest(phases) AS phase, unnest(conditions) AS condition
        GROUP BY phase, condition
        ORDER BY array_length(array_agg(nct_id), 1) DESC, phase, condition
        LIMIT $1
        """,
        max_pairs,
    )

    return [
        EvalQuestion(
            id=f"multi_hop:{row['phase']}:{row['condition']}",
            query_type="multi_hop",
            query_text=f"Which {_phase_label(row['phase'])} trials are studying {row['condition']}?",
            gold_nct_ids=list(row["nct_ids"]),
            filters={"phases": [row["phase"]]},
            source_fields={"phase": row["phase"], "condition": row["condition"]},
        )
        for row in rows
    ]


def _phase_label(phase: str) -> str:
    return "Phase " + phase.removeprefix("PHASE") if phase.startswith("PHASE") else phase


def _unanswerable_questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            id=f"unanswerable:{i}",
            query_type="unanswerable",
            query_text=f"What trials are studying {condition}?",
            gold_nct_ids=[],
            source_fields={"fabricated_condition": condition},
        )
        for i, condition in enumerate(UNANSWERABLE_CONDITIONS)
    ]


async def build_dataset(db: Database) -> list[EvalQuestion]:
    fact = await _fact_questions(db)
    multi_hop = await _multi_hop_questions(db)
    unanswerable = _unanswerable_questions()
    return [*fact, *multi_hop, *unanswerable]


async def _main(output: Path) -> None:
    db = await Database(get_settings()).connect()
    try:
        questions = await build_dataset(db)
    finally:
        await db.close()

    save_dataset(questions, output)

    by_type: dict[str, int] = {}
    for q in questions:
        by_type[q.query_type] = by_type.get(q.query_type, 0) + 1
    print(f"wrote {len(questions)} questions to {output}")
    for query_type, count in sorted(by_type.items()):
        print(f"  {query_type}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    asyncio.run(_main(args.output))
