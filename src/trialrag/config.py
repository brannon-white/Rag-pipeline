"""Application configuration.

Twelve-factor: every knob is an environment variable, resolved once at process
start into an immutable settings object. Nothing in the codebase reads
``os.environ`` directly -- that keeps the full configuration surface greppable
in one file and makes it trivial to snapshot into an eval run's metadata.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Env(StrEnum):
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PROD = "prod"


class Settings(BaseSettings):
    """Resolved application configuration.

    Field names map to ``TRIALRAG_``-prefixed env vars, except vendor keys which
    keep their conventional names (``ANTHROPIC_API_KEY``, ``VOYAGE_API_KEY``) so
    the vendor SDKs and our config agree on one source of truth.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRIALRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Environment ---------------------------------------------------------
    env: Env = Env.LOCAL
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # --- Database ------------------------------------------------------------
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql://trialrag:trialrag@localhost:5432/trialrag")
    )
    # min_size=0 is deliberate: Neon suspends compute after ~5 minutes idle, and
    # a pool that holds even one connection open keeps the meter running 24/7.
    db_pool_min_size: Annotated[int, Field(ge=0, le=50)] = 0
    db_pool_max_size: Annotated[int, Field(ge=1, le=100)] = 10
    db_idle_timeout_s: Annotated[float, Field(gt=0)] = 240.0
    db_command_timeout_s: Annotated[float, Field(gt=0)] = 15.0

    # --- Anthropic -----------------------------------------------------------
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    answer_model: str = "claude-opus-5"
    answer_effort: Effort = "low"
    answer_max_tokens: Annotated[int, Field(ge=256, le=128_000)] = 4096
    query_parse_model: str = "claude-opus-5"
    query_parse_effort: Effort = "low"
    judge_model: str = "claude-opus-5"

    # --- Voyage --------------------------------------------------------------
    voyage_api_key: SecretStr = Field(default=SecretStr(""), alias="VOYAGE_API_KEY")
    embed_model: str = "voyage-4-lite"
    # Matryoshka truncation target. 512 halved storage vs 1024 at a recall cost
    # we measure rather than assume -- see docs/EVALUATION.md dimension ablation.
    embed_dim: Annotated[int, Field(ge=64, le=2048)] = 512
    embed_batch_size: Annotated[int, Field(ge=1, le=256)] = 64
    rerank_model: str = "rerank-2.5-lite"
    # Conservative default: an unpaid Voyage account is hard-capped at 3 req/min
    # on both embeddings and reranking (verified live -- a real ingest run
    # failed outright with RateLimitError before this limiter existed). Raise
    # this once a payment method is on file; the 200M free-token allowance
    # applies either way, only the RPM ceiling changes.
    voyage_rate_limit_rpm: Annotated[float, Field(gt=0)] = 3.0

    # --- Retrieval -----------------------------------------------------------
    dense_k: Annotated[int, Field(ge=1, le=500)] = 50
    sparse_k: Annotated[int, Field(ge=0, le=500)] = 50
    rrf_k: Annotated[int, Field(ge=1)] = 60
    dense_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    rerank_top_n: Annotated[int, Field(ge=1, le=50)] = 8
    max_chunks_per_study: Annotated[int, Field(ge=1, le=20)] = 3
    hnsw_ef_search: Annotated[int, Field(ge=1, le=1000)] = 100

    # --- Cost guardrails -----------------------------------------------------
    # A public endpoint with an LLM behind it is a wallet-drain vector. These are
    # enforced server-side; they are not advisory.
    max_daily_spend_usd: Annotated[float, Field(ge=0.0)] = 1.50
    per_ip_daily_queries: Annotated[int, Field(ge=1)] = 25
    rate_limit_per_minute: Annotated[int, Field(ge=1)] = 6

    # --- Object storage ------------------------------------------------------
    s3_bucket: str | None = None
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    # --- Observability -------------------------------------------------------
    otel_enabled: bool = False
    otel_endpoint: str = Field(default="http://localhost:4318", alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name: str = "trialrag"

    # --- Ingestion -----------------------------------------------------------
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    # ClinicalTrials.gov documents ~50 req/min. We run under it on purpose: a 429
    # storm mid-ingest costs more wall-clock than the margin ever saves.
    ctgov_rate_limit_rpm: Annotated[int, Field(ge=1, le=50)] = 40
    ctgov_page_size: Annotated[int, Field(ge=1, le=1000)] = 1000

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {v!r}")
        return upper

    @property
    def is_deployed(self) -> bool:
        return self.env in (Env.STAGING, Env.PROD)

    @property
    def asyncpg_dsn(self) -> str:
        """asyncpg wants a bare libpq DSN, not SQLAlchemy's driver-qualified form."""
        return str(self.database_url).replace("postgresql+asyncpg://", "postgresql://")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that import order can never produce two divergent configurations.
    Tests override via ``get_settings.cache_clear()`` plus monkeypatched env.
    """
    return Settings()
