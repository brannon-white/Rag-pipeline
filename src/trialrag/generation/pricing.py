"""Per-model dollar rates, for cost attribution and the spend circuit breaker.

Figures are best-effort $/million-tokens estimates, not pulled live from
Anthropic's billing API -- there is no such endpoint. Reconcile these against
the Anthropic Console's usage page periodically; a stale rate here skews
``query_log.cost_usd`` and the daily-spend breaker, but doesn't affect actual
billing, which is metered by Anthropic independently of anything computed
here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRates:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float


# Cache writes are billed at a premium over base input (5m ephemeral
# breakpoints, which is what this project uses throughout); cache reads are
# billed at a steep discount. Both follow Anthropic's usual ratios to base
# input price when an exact quoted number isn't available.
_RATES: dict[str, ModelRates] = {
    "claude-opus-5": ModelRates(
        input_per_mtok=15.0, output_per_mtok=75.0, cache_read_per_mtok=1.5, cache_write_per_mtok=18.75
    ),
    "claude-sonnet-5": ModelRates(
        input_per_mtok=3.0, output_per_mtok=15.0, cache_read_per_mtok=0.3, cache_write_per_mtok=3.75
    ),
    "claude-haiku-4-5": ModelRates(
        input_per_mtok=0.8, output_per_mtok=4.0, cache_read_per_mtok=0.08, cache_write_per_mtok=1.0
    ),
}


def rates_for(model: str) -> ModelRates:
    """Rates for ``model``, falling back to Opus's (the most expensive tier)
    for an unrecognised name -- better to overestimate spend against the
    circuit breaker than to silently undercount it."""
    return _RATES.get(model, _RATES["claude-opus-5"])


def compute_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    rates = rates_for(model)
    return (
        input_tokens * rates.input_per_mtok
        + output_tokens * rates.output_per_mtok
        + cache_read_input_tokens * rates.cache_read_per_mtok
        + cache_creation_input_tokens * rates.cache_write_per_mtok
    ) / 1_000_000
