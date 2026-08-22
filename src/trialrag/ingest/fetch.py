"""ClinicalTrials.gov API v2 client.

The registry publishes a soft limit of roughly 50 requests per minute per IP and
does not issue API keys, so there is no way to buy headroom. That single fact
shapes this module:

* A token-bucket limiter runs at a configured margin *under* the cap. Racing to
  the limit and absorbing 429s costs more wall-clock than the margin saves, and
  it is rude to a public good.
* Requests ask for ``pageSize=1000`` and pass an explicit ``fields`` projection,
  so a full corpus pull is hundreds of requests rather than thousands.
* Every payload is archived verbatim before parsing. Re-chunking, re-embedding
  and ablation sweeps then run entirely off local data -- the expensive,
  rate-limited part happens exactly once.
* Sync is incremental on ``lastUpdatePostDate``: unchanged studies are skipped
  before any parsing or embedding spend.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

JsonMap = Mapping[str, Any]

# Fetching whole records would pull megabytes of results tables we never index
# (NCT04368728 alone is 2 MB, 94% of it resultsSection). This projection is the
# difference between a ~40 MB corpus pull and a multi-gigabyte one.
DEFAULT_FIELDS: Final[tuple[str, ...]] = (
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.descriptionModule",
    "protocolSection.conditionsModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.eligibilityModule",
    "protocolSection.contactsLocationsModule",
    "hasResults",
)

_RETRY_STATUSES: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})


_REPO_URL: Final = "https://github.com/brannon-white/Rag-pipeline"


def _user_agent() -> str:
    """Identify ourselves without tripping the registry's bot filter.

    ClinicalTrials.gov sits behind a WAF that cross-checks the User-Agent
    against the client's TLS fingerprint. *Replacing* httpx's default UA with
    anything -- including a plain ``trialrag/0.1``, and even a verbatim
    ``curl/8.7.1`` -- gets a blanket 403 from a Python client, while the stock
    ``python-httpx/x.y.z`` is allowed. Appending satisfies both: the fingerprint
    still matches the declared library, and a registry operator can still see
    who we are and where to complain.

    Verified empirically; see docs/RUNBOOK.md. If corpus sync starts returning
    403 on every request, check here first -- it will not look like a rate
    limit, because it is not one.
    """
    return f"{httpx.Client().headers['user-agent']} trialrag/0.1 (+{_REPO_URL})"


class CtGovError(RuntimeError):
    """The registry returned something unusable after exhausting retries."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUSES
    return isinstance(exc, httpx.TransportError)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TokenBucket:
    """Async token bucket.

    Chosen over a fixed sleep between calls because it permits a short burst
    after an idle stretch while still holding the long-run average under the
    limit -- which is exactly the shape of a resumed ingest.
    """

    def __init__(self, rate_per_minute: float, *, capacity: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._rate = rate_per_minute / 60.0
        self._capacity = capacity if capacity is not None else max(1.0, rate_per_minute / 10.0)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_for = deficit / self._rate
            await asyncio.sleep(wait_for)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class FetchStats:
    requests: int = 0
    studies_seen: int = 0
    retries: int = 0
    rate_limited: int = 0


@dataclass
class StudyFilter:
    """Server-side corpus selection.

    Filtering at the registry rather than locally is what keeps a bounded corpus
    cheap: the rows we do not want never cross the network, never get parsed and
    never get embedded.
    """

    conditions: Sequence[str] = ()
    statuses: Sequence[str] = ()
    study_type: str | None = "INTERVENTIONAL"
    phases: Sequence[str] = ()
    term: str | None = None

    def as_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.conditions:
            # The registry's query syntax ORs bare alternatives inside parens.
            params["query.cond"] = " OR ".join(f'"{c}"' for c in self.conditions)
        if self.term:
            params["query.term"] = self.term

        # filter.advanced takes an Essie expression; it is the only way to
        # constrain study type and phase server-side in one request.
        advanced: list[str] = []
        if self.study_type:
            advanced.append(f"AREA[StudyType]{self.study_type}")
        if self.phases:
            advanced.append("(" + " OR ".join(f"AREA[Phase]{p}" for p in self.phases) + ")")
        if advanced:
            params["filter.advanced"] = " AND ".join(advanced)

        if self.statuses:
            params["filter.overallStatus"] = "|".join(self.statuses)
        return params


@dataclass
class CtGovClient:
    """Rate-limited, retrying async client for the studies endpoints."""

    base_url: str = "https://clinicaltrials.gov/api/v2"
    rate_limit_rpm: int = 40
    page_size: int = 1000
    max_attempts: int = 5
    timeout_s: float = 60.0
    fields: Sequence[str] = DEFAULT_FIELDS
    stats: FetchStats = field(default_factory=FetchStats)

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _bucket: TokenBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bucket = TokenBucket(self.rate_limit_rpm)

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            headers={"Accept": "application/json", "User-Agent": _user_agent()},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: Mapping[str, str | int]) -> JsonMap:
        if self._client is None:
            raise RuntimeError("CtGovClient must be used as an async context manager")

        attempt_number = 0
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                attempt_number += 1
                if attempt_number > 1:
                    self.stats.retries += 1
                await self._bucket.acquire()
                self.stats.requests += 1
                response = await self._client.get(path, params=dict(params))
                if response.status_code == 429:
                    self.stats.rate_limited += 1
                    # Honour Retry-After when offered; our margin should make
                    # this path rare, and if it is not, the margin is wrong.
                    retry_after = float(response.headers.get("Retry-After", 5))
                    logger.warning("rate limited by registry; sleeping %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                response.raise_for_status()
                payload: Any = response.json()
                if not isinstance(payload, Mapping):
                    raise CtGovError(
                        f"{path}: expected a JSON object, got {type(payload).__name__}"
                    )
                return payload
        raise CtGovError(f"{path}: retries exhausted")  # pragma: no cover - reraise fires first

    async def fetch_study(self, nct_id: str) -> JsonMap:
        """Fetch one study record by NCT ID."""
        return await self._get(f"/studies/{nct_id}", {"format": "json"})

    async def count(self, study_filter: StudyFilter) -> int:
        """Total studies matching ``study_filter``, without paging through them."""
        params: dict[str, str | int] = {
            **study_filter.as_params(),
            "format": "json",
            "countTotal": "true",
            "pageSize": 1,
            "fields": "protocolSection.identificationModule.nctId",
        }
        payload = await self._get("/studies", params)
        total = payload.get("totalCount")
        return int(total) if isinstance(total, int | str) else 0

    async def iter_studies(
        self,
        study_filter: StudyFilter,
        *,
        limit: int | None = None,
        known_versions: Mapping[str, str] | None = None,
    ) -> AsyncIterator[JsonMap]:
        """Page through matching studies, yielding raw records.

        Args:
            study_filter: Server-side corpus selection.
            limit: Stop after this many *yielded* records. Useful for smoke runs.
            known_versions: ``nct_id -> lastUpdatePostDate`` already ingested.
                Matching records are skipped without being yielded, so an
                incremental sync does no parsing or embedding work for them.
        """
        known = known_versions or {}
        params: dict[str, str | int] = {
            **study_filter.as_params(),
            "format": "json",
            "pageSize": self.page_size,
            "fields": "|".join(self.fields),
        }

        yielded = 0
        page_token: str | None = None

        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token

            payload = await self._get("/studies", page_params)
            studies = payload.get("studies")
            if not isinstance(studies, Sequence):
                raise CtGovError("response missing a 'studies' array")

            for record in studies:
                if not isinstance(record, Mapping):
                    continue
                self.stats.studies_seen += 1

                nct_id, updated = _identity(record)
                if nct_id and updated and known.get(nct_id) == updated:
                    continue

                yield record
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

            next_token = payload.get("nextPageToken")
            if not isinstance(next_token, str) or not next_token:
                return
            page_token = next_token


def _identity(record: JsonMap) -> tuple[str | None, str | None]:
    """Extract ``(nct_id, last_update_posted)`` for the incremental-skip check."""
    protocol = record.get("protocolSection")
    if not isinstance(protocol, Mapping):
        return None, None

    identification = protocol.get("identificationModule")
    nct_id = identification.get("nctId") if isinstance(identification, Mapping) else None

    status = protocol.get("statusModule")
    posted = status.get("lastUpdatePostDateStruct") if isinstance(status, Mapping) else None
    updated = posted.get("date") if isinstance(posted, Mapping) else None

    return (
        nct_id if isinstance(nct_id, str) else None,
        updated if isinstance(updated, str) else None,
    )


async def collect(
    study_filter: StudyFilter,
    *,
    limit: int | None = None,
    known_versions: Mapping[str, str] | None = None,
    **client_kwargs: Any,
) -> list[JsonMap]:
    """Convenience wrapper: fetch matching studies into a list.

    Intended for tests and smoke runs. The production pipeline consumes
    :meth:`CtGovClient.iter_studies` as a stream so that memory stays flat
    regardless of corpus size.
    """
    async with CtGovClient(**client_kwargs) as client:
        return [
            record
            async for record in client.iter_studies(
                study_filter, limit=limit, known_versions=known_versions
            )
        ]


def versions_of(records: Iterable[JsonMap]) -> dict[str, str]:
    """Build a ``known_versions`` map from already-ingested raw records."""
    out: dict[str, str] = {}
    for record in records:
        nct_id, updated = _identity(record)
        if nct_id and updated:
            out[nct_id] = updated
    return out
