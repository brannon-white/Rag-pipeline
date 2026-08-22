"""Pure retrieval metrics: Recall@k, MRR, nDCG@k.

Deterministic, dependency-light (only numpy), and separated from anything
that touches the network or a database -- this module is what runs in CI on
every PR, in milliseconds, with zero API cost. The eval *harness*
(retrieval_eval.py) is what's expensive and non-deterministic (real
embeddings, an LLM query parse); the *scoring*, once you have a ranked list
and a gold set, never should be.

Relevance throughout is binary at the study level: a retrieved chunk is
"relevant" iff its ``nct_id`` is in the question's gold set. See
``evals/dataset.py``'s docstring for why gold is study-level, not chunk-level.
"""

from __future__ import annotations

import numpy as np


def relevance_vector(retrieved_nct_ids: list[str], gold_nct_ids: set[str]) -> list[int]:
    """Binary relevance per retrieved position, in rank order."""
    return [1 if nct_id in gold_nct_ids else 0 for nct_id in retrieved_nct_ids]


def recall_at_k(relevance: list[int], k: int) -> float:
    """Whether *any* relevant document appears in the top ``k`` -- not the
    fraction of the gold set found, since a single retrieved chunk from the
    right study is enough to answer a study-level question."""
    if not relevance:
        return 0.0
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[int]) -> float:
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """Binary-relevance nDCG@k, normalised against the ideal ordering given
    however many relevant items actually appear in the top ``k``."""
    rel_k = np.asarray(relevance[:k], dtype=float)
    if rel_k.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel_k.size + 2))
    dcg = float(np.sum(rel_k * discounts))

    n_relevant = min(int(np.sum(rel_k)), k)
    if n_relevant == 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, n_relevant + 2))
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


K_VALUES: tuple[int, ...] = (1, 5, 10, 20)


def score_one(retrieved_nct_ids: list[str], gold_nct_ids: list[str]) -> dict[str, float]:
    """All metrics for a single question."""
    gold = set(gold_nct_ids)
    relevance = relevance_vector(retrieved_nct_ids, gold)
    scores = {f"recall@{k}": recall_at_k(relevance, k) for k in K_VALUES}
    scores["mrr"] = reciprocal_rank(relevance)
    scores["ndcg@10"] = ndcg_at_k(relevance, 10)
    return scores


def aggregate(per_question: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across questions. Empty input yields an empty dict
    rather than NaNs -- a slice with zero questions has nothing to report."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {key: float(np.mean([q[key] for q in per_question])) for key in keys}
