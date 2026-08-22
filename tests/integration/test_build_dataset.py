"""Golden-dataset construction tests against a real Postgres.

Verifies the generator's SQL directly: fact questions per available field,
the phase+condition co-occurrence query behind multi-hop questions, and
partial-data handling (a study missing a field must not produce a question
that field would answer -- generating one would put an unfindable gold answer
into the dataset).
"""

from __future__ import annotations

import pytest
from evals.build_dataset import UNANSWERABLE_CONDITIONS, build_dataset

from trialrag.db.pool import Database
from trialrag.domain.models import Study

pytestmark = pytest.mark.integration


async def _seed(db: Database, **overrides: object) -> Study:
    from trialrag.ingest.load import upsert_studies

    defaults: dict[str, object] = {
        "nct_id": "NCT00000001",
        "brief_title": "A study of things",
        "source_hash": "h1",
        "phases": ("PHASE3",),
        "conditions": ("Diabetes",),
        "lead_sponsor": "Example Sponsor",
    }
    defaults.update(overrides)
    study = Study(**defaults)  # type: ignore[arg-type]
    await upsert_studies(db, [study])
    return study


_FACT_TYPES = {"fact_phase", "fact_sponsor", "fact_condition"}


async def test_fact_questions_generated_for_a_fully_populated_study(db: Database) -> None:
    await _seed(db)
    questions = await build_dataset(db)

    # A single-study corpus legitimately also produces one multi_hop question
    # with this same singleton gold set (its phase+condition pair still
    # "co-occurs", just with one study) -- filtering by query_type, not by
    # gold-set equality, is what isolates the fact_* questions specifically.
    by_type = {q.query_type: q for q in questions if q.query_type in _FACT_TYPES}
    assert set(by_type) == _FACT_TYPES
    assert "A study of things" in by_type["fact_phase"].query_text


async def test_missing_sponsor_skips_only_that_question_type(db: Database) -> None:
    await _seed(db, lead_sponsor=None)
    questions = await build_dataset(db)

    types = {q.query_type for q in questions if q.gold_nct_ids == ["NCT00000001"]}
    assert "fact_sponsor" not in types
    assert "fact_phase" in types  # unaffected fields still produce questions
    assert "fact_condition" in types


async def test_missing_phases_and_conditions_skip_their_questions(db: Database) -> None:
    await _seed(db, phases=(), conditions=())
    questions = await build_dataset(db)

    types = {q.query_type for q in questions if q.gold_nct_ids == ["NCT00000001"]}
    assert types == {"fact_sponsor"}


async def test_multi_hop_gold_set_matches_real_cooccurrence(db: Database) -> None:
    await _seed(
        db, nct_id="NCT00000001", phases=("PHASE3",), conditions=("Asthma",), source_hash="a"
    )
    await _seed(
        db, nct_id="NCT00000002", phases=("PHASE3",), conditions=("Asthma",), source_hash="b"
    )
    await _seed(
        db, nct_id="NCT00000003", phases=("PHASE1",), conditions=("Asthma",), source_hash="c"
    )

    questions = await build_dataset(db)
    multi_hop = next(
        q
        for q in questions
        if q.query_type == "multi_hop"
        and q.filters.get("phases") == ["PHASE3"]
        and "Asthma" in q.query_text
    )

    assert set(multi_hop.gold_nct_ids) == {"NCT00000001", "NCT00000002"}
    assert "NCT00000003" not in multi_hop.gold_nct_ids  # wrong phase, correctly excluded


async def test_unanswerable_questions_always_present_with_empty_gold(db: Database) -> None:
    await _seed(db)
    questions = await build_dataset(db)

    unanswerable = [q for q in questions if q.query_type == "unanswerable"]
    assert len(unanswerable) == len(UNANSWERABLE_CONDITIONS)
    assert all(q.gold_nct_ids == [] for q in unanswerable)


async def test_empty_corpus_yields_only_unanswerable_questions(db: Database) -> None:
    questions = await build_dataset(db)
    assert {q.query_type for q in questions} == {"unanswerable"}
