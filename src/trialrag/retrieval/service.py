"""Hybrid retrieval: one SQL statement, two arms, fused by reciprocal rank.

The whole point of putting the corpus in Postgres rather than a dedicated
vector store (ADR-001) is that the metadata filter, the dense arm and the
lexical arm all run inside the *same* query plan. ``filtered_studies`` below
is a pre-filter both arms join against -- never a post-filter applied after
ANN search -- so a selective filter (a specific phase, a narrow age band)
cannot silently starve the result set the way it would if filtering happened
after a fixed-size top-k vector search.

Reciprocal Rank Fusion, not a weighted sum of raw scores: cosine similarity and
``ts_rank_cd`` live on incomparable scales (roughly [-1, 1] vs. an unbounded,
corpus-frequency-dependent value), and RRF sidesteps that by fusing on rank
position instead of raw magnitude. ``dense_weight`` still lets either arm
dominate the fusion without needing the two scores to be numerically
commensurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from trialrag.db.pool import Database
from trialrag.domain.models import RetrievedChunk, SectionKind
from trialrag.retrieval.filters import QueryFilters

logger = logging.getLogger(__name__)

# One statement, three CTEs. Parameters, in order:
#   $1 statuses (text[] | NULL)      $2  phases (text[] | NULL)
#   $3 query_min_age (numeric|NULL)  $4  query_max_age (numeric|NULL)
#      -- interval-overlap against the study's [min_age_years, max_age_years];
#      -- $3 (a stated lower bound) must not exceed the study's ceiling, and
#      -- $4 (a stated upper bound) must not fall below the study's floor.
#   $5 sex (text|NULL)               $6  healthy_volunteers (bool|NULL)
#   $7 query embedding (halfvec)     $8  dense_k
#   $9 search_query (text)           $10 sparse_k
#   $11 dense_weight (float)         $12 rrf_k
#   $13 final limit (pre-rerank)
_HYBRID_SEARCH_SQL = """
WITH filtered_studies AS (
    SELECT nct_id
    FROM studies st
    WHERE ($1::text[] IS NULL OR st.overall_status = ANY($1))
      AND ($2::text[] IS NULL OR st.phases && $2)
      AND ($3::numeric IS NULL OR st.max_age_years IS NULL OR st.max_age_years >= $3)
      AND ($4::numeric IS NULL OR st.min_age_years IS NULL OR st.min_age_years <= $4)
      AND ($5::text IS NULL OR st.sex = 'ALL' OR st.sex = $5)
      AND ($6::bool IS NULL OR st.healthy_volunteers IS NULL OR st.healthy_volunteers = $6)
),
dense AS (
    SELECT c.id,
           row_number() OVER (ORDER BY c.embedding <=> $7::halfvec(512)) AS rank,
           1 - (c.embedding <=> $7::halfvec(512)) AS score
    FROM chunks c
    JOIN filtered_studies fs ON fs.nct_id = c.nct_id
    ORDER BY c.embedding <=> $7::halfvec(512)
    LIMIT $8
),
sparse AS (
    SELECT c.id,
           row_number() OVER (ORDER BY ts_rank_cd(c.tsv, q.query) DESC) AS rank,
           ts_rank_cd(c.tsv, q.query) AS score
    FROM chunks c
    JOIN filtered_studies fs ON fs.nct_id = c.nct_id
    CROSS JOIN LATERAL websearch_to_tsquery('english', $9) AS q(query)
    WHERE c.tsv @@ q.query
    ORDER BY score DESC
    LIMIT $10
),
fused AS (
    SELECT
        COALESCE(d.id, s.id) AS chunk_id,
        ($11 * COALESCE(1.0 / ($12 + d.rank), 0.0))
            + ((1.0 - $11) * COALESCE(1.0 / ($12 + s.rank), 0.0)) AS rrf_score,
        d.rank AS dense_rank, d.score AS dense_score,
        s.rank AS sparse_rank, s.score AS sparse_score
    FROM dense d
    FULL OUTER JOIN sparse s ON d.id = s.id
)
SELECT
    c.id AS chunk_id, c.nct_id, c.section, c.content, c.context_header,
    st.brief_title AS study_title,
    f.dense_rank, f.dense_score, f.sparse_rank, f.sparse_score, f.rrf_score
FROM fused f
JOIN chunks c ON c.id = f.chunk_id
JOIN studies st ON st.nct_id = c.nct_id
ORDER BY f.rrf_score DESC
LIMIT $13
"""


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable knobs for the fused search, swept by the ablation harness."""

    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    dense_weight: float = 0.5
    limit: int = 50  # candidates returned pre-rerank

    def __post_init__(self) -> None:
        if not 0.0 <= self.dense_weight <= 1.0:
            raise ValueError("dense_weight must be in [0, 1]")


DEFAULT_CONFIG: RetrievalConfig = RetrievalConfig()


def _age_bounds(filters: QueryFilters) -> tuple[float | None, float | None]:
    """Interval-overlap bounds for the age filter.

    ``min_age_years``/``max_age_years`` on a parsed query describe a stated
    person-age constraint ("a 10-year-old", "over 65"), not a range to match
    verbatim against the study's own eligibility window -- so the filter is an
    interval-overlap test, not an equality test.
    """
    return filters.min_age_years, filters.max_age_years


async def hybrid_search(
    db: Database,
    *,
    query_vector: list[float],
    search_query: str,
    filters: QueryFilters,
    config: RetrievalConfig = DEFAULT_CONFIG,
) -> list[RetrievedChunk]:
    """Run the fused dense+lexical search and return ranked chunks.

    Ranks and scores from both arms are preserved on each result (see
    :class:`trialrag.domain.models.RetrievedChunk`) rather than collapsed into
    one number -- the debug UI and the ablation harness both need to see why a
    chunk ranked where it did, not just that it did.
    """
    min_age, max_age = _age_bounds(filters)

    rows = await db.fetch(
        _HYBRID_SEARCH_SQL,
        filters.statuses,
        filters.phases,
        min_age,
        max_age,
        filters.sex,
        filters.healthy_volunteers,
        query_vector,
        config.dense_k,
        search_query,
        config.sparse_k,
        config.dense_weight,
        float(config.rrf_k),
        config.limit,
    )

    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            nct_id=row["nct_id"],
            kind=SectionKind(row["section"]),
            content=row["content"],
            context_header=row["context_header"],
            study_title=row["study_title"],
            dense_rank=row["dense_rank"],
            dense_score=row["dense_score"],
            sparse_rank=row["sparse_rank"],
            sparse_score=row["sparse_score"],
            rrf_score=row["rrf_score"],
            final_rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]
