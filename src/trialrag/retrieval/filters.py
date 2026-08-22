"""Natural-language query -> structured filters + semantic search string.

One Claude call, using ``client.messages.parse()`` with a Pydantic
``output_format`` -- the SDK-recommended structured-output path, not a hand-
rolled JSON schema or a strict tool call (both would work, but ``parse()``
already validates the response against the model and hands back a typed
object, which is strictly less code for the same guarantee).

Only genuinely structured, controlled-vocabulary fields become SQL filters
(phase, status, age, sex). "Condition" is deliberately *not* one of them: the
registry's ``conditions`` array is free text with no fixed vocabulary
("Type 2 Diabetes Mellitus" vs. "T2DM" vs. "Diabetes Mellitus, Type 2" all
appear in real records), so an exact-match filter on it would silently drop
results that mean the same thing but are spelled differently. Condition intent
flows into ``search_query`` instead, where the hybrid search's dense and
lexical arms both handle the variation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from trialrag.domain.models import StudyStatus

logger = logging.getLogger(__name__)

# The registry's own phase vocabulary (verified against live API responses;
# see tests/fixtures/*.json). Not modelled as a domain enum because, unlike
# status, a study can carry more than one phase ("PHASE2", "PHASE3").
KNOWN_PHASES: frozenset[str] = frozenset(
    {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"}
)
KNOWN_STATUSES: frozenset[str] = frozenset(s.value for s in StudyStatus) - {StudyStatus.UNKNOWN}

Sex = Literal["ALL", "MALE", "FEMALE"]


class QueryFilters(BaseModel):
    """Structured filters extracted from a user's question.

    Every field is optional and defaults to "no constraint" -- an under-
    specified query (the common case) should filter nothing, not silently
    exclude studies because a field came back empty.
    """

    phases: list[str] | None = Field(
        default=None, description="Trial phases mentioned, e.g. ['PHASE3']. Omit if unspecified."
    )
    statuses: list[str] | None = Field(
        default=None,
        description="Recruitment statuses mentioned, e.g. ['RECRUITING']. Omit if unspecified.",
    )
    min_age_years: float | None = Field(
        default=None, description="A stated minimum/lower age bound, converted to years."
    )
    max_age_years: float | None = Field(
        default=None, description="A stated maximum/upper age bound, converted to years."
    )
    sex: Sex | None = Field(
        default=None, description="Only if the query explicitly restricts to one sex."
    )
    healthy_volunteers: bool | None = Field(
        default=None, description="True/False only if explicitly discussed; otherwise omit."
    )

    def normalised(self) -> QueryFilters:
        """Drop any value the registry vocabulary doesn't recognise.

        The model is asked to use registry tokens, but nothing stops it from
        writing "Active" instead of "ACTIVE_NOT_RECRUITING". Silently dropping
        an unrecognised filter (rather than passing it through to SQL, where it
        would just match nothing) keeps a near-miss from turning into a
        confusing zero-result query.
        """
        phases = [p for p in (self.phases or []) if p in KNOWN_PHASES] or None
        statuses = [s for s in (self.statuses or []) if s in KNOWN_STATUSES] or None
        return self.model_copy(update={"phases": phases, "statuses": statuses})


class ParsedQuery(BaseModel):
    """What one query-parse call produces."""

    filters: QueryFilters
    search_query: str = Field(
        description="The query rewritten for semantic + lexical search: condition, "
        "intervention and topic language preserved; filter phrasing "
        "('phase 3', 'recruiting', 'adults only') removed since it is "
        "captured structurally instead."
    )


_SYSTEM_PROMPT = f"""You turn a user's natural-language question about clinical \
trials into structured filters plus a clean search string.

Valid phase tokens: {sorted(KNOWN_PHASES)}
Valid status tokens: {sorted(KNOWN_STATUSES)}

Only set a filter field when the query actually states that constraint. Leave \
everything else null -- an unstated constraint must not narrow the search. \
Convert stated ages to years (e.g. "6 months old" -> 0.5, "over 65" -> \
min_age_years 65). For search_query, keep the medical/topical content (\
conditions, drugs, interventions, comparisons) and drop filter phrasing that \
is already captured structurally, since it would otherwise double-count \
against the same signal in lexical search."""


class QueryParseError(RuntimeError):
    """The parse call failed after retries, or returned something unusable."""


@dataclass
class QueryParser:
    """Wraps the Claude call that turns a query into :class:`ParsedQuery`.

    Takes an explicit ``api_key`` rather than relying on
    ``AsyncAnthropic()``'s implicit ``ANTHROPIC_API_KEY`` environment lookup --
    this app's convention is that secrets flow through ``config.Settings``
    (loaded from ``.env`` via pydantic-settings), which does *not* export them
    back into ``os.environ`` for other libraries to auto-discover. The same
    convention is why :class:`trialrag.ingest.embed.Embedder` takes an
    explicit ``api_key`` instead of relying on Voyage's client to find one.
    """

    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_attempts: int = 4
    api_key: str | None = None
    client: anthropic.AsyncAnthropic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def parse(self, query: str) -> ParsedQuery:
        if not query.strip():
            raise QueryParseError("query is empty")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            retry=retry_if_exception_type(
                (
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError,
                )
            ),
            reraise=True,
        ):
            with attempt:
                response = await self.client.messages.parse(
                    model=self.model,
                    max_tokens=1024,
                    output_config={"effort": self.effort},
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": query}],
                    output_format=ParsedQuery,
                )

        if response.stop_reason == "refusal":
            # A protocol-search query refusing is unexpected; fail loudly
            # rather than silently searching with an empty filter set.
            raise QueryParseError(f"query parse refused: {response.stop_reason!r}")

        parsed = response.parsed_output
        if parsed is None:
            raise QueryParseError("response carried no parsed_output")

        return ParsedQuery(filters=parsed.filters.normalised(), search_query=parsed.search_query)

    async def aclose(self) -> None:
        await self.client.close()
