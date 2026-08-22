"""The golden-dataset item schema, shared by the builder and the scorers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

QueryType = Literal["fact_phase", "fact_sponsor", "fact_condition", "multi_hop", "unanswerable"]


class EvalQuestion(BaseModel):
    """One golden Q&A item, built by construction from structured DB fields.

    ``gold_nct_ids`` is deliberately study-level, not chunk-level: several of
    the source fields (sponsor, phase, condition) live in every chunk's
    deterministic context header rather than in any one chunk's body text, so
    "the correct chunk" isn't a well-defined single answer -- "the correct
    study" is. Document-level recall is a standard, honest retrieval metric
    that doesn't require guessing which chunk a fact "belongs to".

    ``filters`` records the filter the question was *constructed* to be
    answered under (e.g. a multi-hop question built from a real phase+condition
    co-occurrence). The retrieval eval applies these explicitly rather than
    routing through the LLM query-parser, so this harness measures the search
    mechanism in isolation from query-parsing quality -- the two are evaluated
    separately on purpose.
    """

    id: str
    query_type: QueryType
    query_text: str
    gold_nct_ids: list[str] = Field(default_factory=list)
    filters: dict[str, object] = Field(default_factory=dict)
    source_fields: dict[str, object] = Field(default_factory=dict)


def save_dataset(questions: list[EvalQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump() for q in questions]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> list[EvalQuestion]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion.model_validate(item) for item in data]
