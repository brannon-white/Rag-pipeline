"""Tests for the pure retrieval metrics.

These are exactly the tests that gate every PR in CI (see the plan's Tier-1 /
Tier-2 eval split): deterministic, dependency-light, and fast enough to run on
every commit with zero API cost.
"""

from __future__ import annotations

import math

import pytest
from evals.metrics import (
    aggregate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_vector,
    score_one,
)


def test_relevance_vector_marks_gold_positions() -> None:
    assert relevance_vector(["A", "B", "C"], {"B"}) == [0, 1, 0]


def test_relevance_vector_handles_no_matches() -> None:
    assert relevance_vector(["A", "B"], {"Z"}) == [0, 0]


def test_relevance_vector_empty_retrieval() -> None:
    assert relevance_vector([], {"A"}) == []


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_hit_within_window() -> None:
    assert recall_at_k([0, 0, 1, 0], k=3) == 1.0


def test_recall_at_k_hit_outside_window() -> None:
    assert recall_at_k([0, 0, 0, 1], k=3) == 0.0


def test_recall_at_k_empty_relevance_is_zero() -> None:
    assert recall_at_k([], k=5) == 0.0


def test_recall_at_k_is_any_not_fraction() -> None:
    """Study-level recall: one hit anywhere in the window is a full credit,
    not partial credit for missing the study's other chunks."""
    assert recall_at_k([1, 1, 1], k=3) == recall_at_k([1, 0, 0], k=3) == 1.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relevance", "expected"),
    [
        ([1, 0, 0], 1.0),
        ([0, 1, 0], 0.5),
        ([0, 0, 1], pytest.approx(1 / 3)),
        ([0, 0, 0], 0.0),
        ([], 0.0),
    ],
)
def test_reciprocal_rank(relevance: list[int], expected: float) -> None:
    assert reciprocal_rank(relevance) == expected


def test_reciprocal_rank_uses_first_hit_only() -> None:
    assert reciprocal_rank([0, 1, 1]) == 0.5


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ordering_is_one() -> None:
    assert ndcg_at_k([1, 1, 0, 0], k=4) == pytest.approx(1.0)


def test_ndcg_no_relevant_items_is_zero() -> None:
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_worst_ordering_is_penalised() -> None:
    """The same single relevant item, ranked last vs. first, must score
    strictly lower -- otherwise nDCG isn't measuring rank position at all."""
    best = ndcg_at_k([1, 0, 0], k=3)
    worst = ndcg_at_k([0, 0, 1], k=3)
    assert best == pytest.approx(1.0)
    assert worst < best


def test_ndcg_matches_hand_computed_value() -> None:
    # relevance [1, 0, 1] at k=3: DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG (2 relevant, ideal = [1,1,0]) = 1/log2(2) + 1/log2(3) = 1 + 0.6309...
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k([1, 0, 1], k=3) == pytest.approx(dcg / idcg)


def test_ndcg_truncates_at_k() -> None:
    """A relevant item beyond the window must not contribute -- otherwise
    nDCG@10 would be indistinguishable from nDCG@1000."""
    within_window = ndcg_at_k([1, 0, 0], k=2)
    beyond_window = ndcg_at_k([0, 0, 1], k=2)
    assert within_window == pytest.approx(1.0)
    assert beyond_window == 0.0


def test_ndcg_empty_relevance_is_zero() -> None:
    assert ndcg_at_k([], k=10) == 0.0


# ---------------------------------------------------------------------------
# score_one / aggregate
# ---------------------------------------------------------------------------


def test_score_one_produces_all_expected_keys() -> None:
    scores = score_one(["NCT1", "NCT2", "NCT3"], ["NCT2"])
    assert set(scores) == {"recall@1", "recall@5", "recall@10", "recall@20", "mrr", "ndcg@10"}
    assert scores["recall@1"] == 0.0
    assert scores["recall@5"] == 1.0
    assert scores["mrr"] == 0.5


def test_score_one_unanswerable_style_empty_gold_scores_zero_everywhere() -> None:
    """An empty gold set (the unanswerable category) can never be "recalled" --
    this is the exact reason retrieval_eval.py scores that category
    separately rather than averaging it into these metrics."""
    scores = score_one(["NCT1", "NCT2"], [])
    assert all(value == 0.0 for value in scores.values())


def test_aggregate_averages_across_questions() -> None:
    per_question = [
        {"recall@1": 1.0, "mrr": 1.0},
        {"recall@1": 0.0, "mrr": 0.5},
    ]
    result = aggregate(per_question)
    assert result["recall@1"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.75)


def test_aggregate_empty_input_returns_empty_dict() -> None:
    assert aggregate([]) == {}
