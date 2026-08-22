# Evaluation

## Retrieval ablation

Sweep of retrieval-time parameters against the golden dataset
(`evals/datasets/golden.json`, 108 questions), evaluated on the live corpus
(30 studies / 432 chunks). See `evals/ablations.py`'s module docstring for
the full rationale on what is and isn't swept here.

| config | recall@1 | recall@5 | recall@10 | recall@20 | mrr | ndcg@10 |
|---|---|---|---|---|---|---|
| dense_only | 0.962 | 1.000 | 1.000 | 1.000 | 0.981 | 0.987 |
| sparse_only | 0.048 | 0.057 | 0.057 | 0.057 | 0.052 | 0.054 |
| hybrid | 0.990 | 1.000 | 1.000 | 1.000 | 0.995 | 0.995 |
| hybrid+rerank | 0.981 | 1.000 | 1.000 | 1.000 | 0.990 | 0.993 |

**Reading this table:** `dense_only` and `sparse_only` isolate each arm;
`hybrid` fuses them via Reciprocal Rank Fusion at equal weight; `hybrid+rerank`
adds the Voyage `rerank-2.5-lite` cross-encoder pass on top. `unanswerable`
questions are excluded from these metrics by construction (see
`evals/metrics.py`) -- their gold set is empty, so recall/MRR/nDCG are
undefined for them; `retrieval_eval.py` reports them separately (mean top-1
fused score, a confidence check rather than a recall check).

### Findings, reported honestly

**Sparse-only is not competitive on this dataset, and that's expected rather
than a bug.** Recall@1 of 0.048 was checked against the SQL and the query
phrasing before writing it here, not taken at face value. The golden
questions are natural-language wrappers around entity names --
`What phase is the clinical trial titled "<title>"?` -- and the lexical arm
runs `websearch_to_tsquery`, which ANDs terms by default. Generic filler
words in the wrapper (`phase`, `clinical`, `trial`, `titled`) compete on equal
footing with the distinctive title terms, so a chunk has to lexically contain
nearly the whole phrase to match at all. This is a real property of plain
full-text search on question-shaped queries, not an artifact of this corpus
or a broken index -- it's the reason hybrid retrieval exists rather than
lexical search alone.

**`hybrid+rerank` does not clearly beat plain `hybrid` here.** Recall@5/10/20
are tied at a perfect 1.000 for both, and `hybrid+rerank` is marginally
*lower* on recall@1 (0.981 vs 0.990), MRR (0.990 vs 0.995), and nDCG@10 (0.993
vs 0.995). At this corpus size (30 studies) hybrid RRF fusion alone is
already close to ceiling, so there isn't enough headroom left for reranking
to show a benefit -- the small gap is noise at this sample size, not evidence
that reranking hurts. The value of the cross-encoder pass (and the per-study
diversity cap it enables) is expected to show up as the corpus grows past the
point where hybrid retrieval alone saturates recall@5; it isn't visible in
this snapshot and shouldn't be oversold as a regression.

**Numbers are clustered near 1.0 for every dense-including config.** With 30
studies and fact-question titles that are individually distinctive, this is
expected -- it demonstrates that retrieval is not currently the bottleneck at
this corpus size, not that the ablation grid found no separation. Re-run this
sweep as the corpus grows to see whether the gap between `hybrid` and
`hybrid+rerank` opens up.

**Unanswerable questions: retrieval alone cannot detect "no good match" by
absence of results.** From the baseline run (`evals/results/baseline.json`):
`n_returned_any_result = 1.000` for unanswerable questions -- ANN search
always returns *something*, even against a fabricated condition. The mean
top-1 fused score for these (0.008) is meaningfully lower than for real
matches (~0.01-0.02), so the *score* carries a signal, but there's no
result-count-based signal. Abstention on unanswerable queries has to be
implemented in generation (M3), by thresholding on retrieval score and/or an
LLM judgment call, not inferred from retrieval returning zero rows.

### Scope: what isn't swept here, and why

- **HNSW `ef_search`** -- checked directly with `EXPLAIN (ANALYZE, BUFFERS)`
  against the live corpus (432 chunks): Postgres's planner chooses a
  sequential scan over the HNSW index at this table size, because a full scan
  is cheaper than index overhead for a table this small. `ef_search` only
  affects an actual index scan, so sweeping it right now would produce a
  flat, meaningless line -- not a finding, just noise. Revisit once the
  corpus is large enough that the planner prefers the index.
- **Embedding dimension (1024/512/256) and chunking strategy** -- both
  require re-embedding or re-chunking the *entire* corpus per grid point. At
  the current Voyage rate-limit tier that is a multi-run, multi-hour
  undertaking, not a five-minute sweep -- deferred to a dedicated ablation
  pass rather than rushed here.

### Baseline reference

The single-configuration baseline run (hybrid + rerank, the production
default) is recorded in `evals/results/baseline.json` and the `eval_runs`
table: overall recall@5 = 1.00, MRR = 0.995, nDCG@10 = 0.995;
`multi_hop` (harder -- needs both a filter and a semantic match) recall@1 =
0.933. This ablation sweep is consistent with that baseline.
