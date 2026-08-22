"""Canonical types for studies, sections, chunks and retrieval results.

Field names and enum members mirror the ClinicalTrials.gov API v2 vocabulary
where a shared vocabulary is genuinely useful (statuses, phases, sexes are
registry-defined controlled terms) and diverge where the registry's encoding is
inconvenient -- notably ages, which the API ships as human strings like
``"12 Years"`` and we normalise to float years.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NCT_ID_RE = re.compile(r"^NCT\d{8}$")
NctId = Annotated[str, Field(pattern=NCT_ID_RE.pattern)]


class Frozen(BaseModel):
    """Immutable base. Domain objects are values; mutation is always a bug here."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------


class LenientEnum(StrEnum):
    """A controlled vocabulary that never rejects an unrecognised token.

    ClinicalTrials.gov adds and renames vocabulary members without notice. A
    strict enum would abort a 5,000-study ingest over one new status string, so
    every member set here carries a designated fallback and unknown values
    degrade onto it. Subclasses set ``__fallback__`` -- a dunder, because
    ``Enum`` reserves ``_sunder_`` names and turns any plain class attribute
    into a member.

    Use :meth:`coerce` rather than the constructor at parse boundaries -- it
    accepts ``None`` for absent fields, which the registry uses constantly.
    """

    __fallback__: ClassVar[str] = ""

    @classmethod
    def _missing_(cls, value: object) -> Self:
        return cls(cls.__fallback__)

    @classmethod
    def coerce(cls, value: str | None) -> Self:
        """Map a raw registry token, or its absence, onto a member."""
        return cls(cls.__fallback__) if value is None else cls(value)


class StudyStatus(LenientEnum):
    __fallback__ = "UNKNOWN"

    RECRUITING = "RECRUITING"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class StudyType(LenientEnum):
    __fallback__ = "UNKNOWN"

    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"
    UNKNOWN = "UNKNOWN"


class Sex(LenientEnum):
    # "ALL" is the safe default: a study whose sex eligibility we cannot read
    # must not be filtered out of results for either sex.
    __fallback__ = "ALL"

    ALL = "ALL"
    FEMALE = "FEMALE"
    MALE = "MALE"


class SectionKind(StrEnum):
    """Which part of a protocol a piece of text came from.

    Retrieval quality is reported per-section, because an aggregate Recall@10
    hides the case where eligibility criteria retrieve well and outcome
    measures do not -- which is exactly the failure that matters to users.
    """

    BRIEF_SUMMARY = "brief_summary"
    DETAILED_DESCRIPTION = "detailed_description"
    ELIGIBILITY_INCLUSION = "eligibility_inclusion"
    ELIGIBILITY_EXCLUSION = "eligibility_exclusion"
    ELIGIBILITY_OTHER = "eligibility_other"
    ARM_GROUP = "arm_group"
    INTERVENTION = "intervention"
    PRIMARY_OUTCOME = "primary_outcome"
    SECONDARY_OUTCOME = "secondary_outcome"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

_AGE_RE = re.compile(
    r"^\s*(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>year|month|week|day|hour|minute)s?\s*$", re.I
)
_UNIT_YEARS: dict[str, float] = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 1 / 52.1775,
    "day": 1 / 365.25,
    "hour": 1 / 8766.0,
    "minute": 1 / 525_960.0,
}


class AgeRange(Frozen):
    """Eligible age window, normalised to years.

    The registry encodes ages as free text with units (``"12 Years"``,
    ``"6 Months"``, ``"N/A"``). Comparing those as strings is how you end up
    matching "9 Years" > "12 Years"; normalising once at ingest is the fix.
    """

    min_years: float | None = None
    max_years: float | None = None
    raw_min: str | None = None
    raw_max: str | None = None

    @staticmethod
    def parse_age(raw: str | None) -> float | None:
        """``"18 Years"`` -> ``18.0``; ``"6 Months"`` -> ``0.5``; junk -> ``None``."""
        if not raw:
            return None
        match = _AGE_RE.match(raw)
        if match is None:
            return None
        return float(match["n"]) * _UNIT_YEARS[match["unit"].lower()]

    @classmethod
    def from_raw(cls, raw_min: str | None, raw_max: str | None) -> Self:
        return cls(
            min_years=cls.parse_age(raw_min),
            max_years=cls.parse_age(raw_max),
            raw_min=raw_min,
            raw_max=raw_max,
        )

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        lo, hi = self.min_years, self.max_years
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"min age {lo} exceeds max age {hi}")
        return self

    def includes(self, age_years: float) -> bool:
        if self.min_years is not None and age_years < self.min_years:
            return False
        return self.max_years is None or age_years <= self.max_years

    def human(self) -> str:
        lo = self.raw_min or "any"
        hi = self.raw_max or "no upper limit"
        return f"{lo} to {hi}"


class Intervention(Frozen):
    type: str
    name: str
    description: str | None = None
    other_names: tuple[str, ...] = ()


class Outcome(Frozen):
    measure: str
    description: str | None = None
    time_frame: str | None = None


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


class Study(Frozen):
    """A protocol record: the structured half of a ClinicalTrials.gov study.

    Everything here is filterable metadata *and* verifiable ground truth. The
    eval harness builds golden Q&A pairs directly off these fields, which is
    what makes retrieval scoring deterministic instead of LLM-judged.
    """

    nct_id: NctId
    brief_title: str
    official_title: str | None = None

    overall_status: StudyStatus = StudyStatus.UNKNOWN
    study_type: StudyType = StudyType.UNKNOWN
    phases: tuple[str, ...] = ()

    conditions: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    interventions: tuple[Intervention, ...] = ()

    lead_sponsor: str | None = None
    sponsor_class: str | None = None
    enrollment: int | None = None
    enrollment_type: str | None = None

    age_range: AgeRange = AgeRange()
    sex: Sex = Sex.ALL
    healthy_volunteers: bool | None = None
    std_ages: tuple[str, ...] = ()

    allocation: str | None = None
    intervention_model: str | None = None
    primary_purpose: str | None = None
    masking: str | None = None

    start_date: dt.date | None = None
    completion_date: dt.date | None = None
    last_update_posted: dt.date | None = None

    countries: tuple[str, ...] = ()
    location_count: int = 0
    has_results: bool = False

    source_hash: str = ""
    raw_s3_key: str | None = None

    @property
    def phase_label(self) -> str:
        """``('PHASE2','PHASE3')`` -> ``"Phase 2/3"``; empty -> ``"unphased"``."""
        if not self.phases:
            return "unphased"
        nums = [p.removeprefix("PHASE") for p in self.phases if p.startswith("PHASE")]
        if not nums:
            return "/".join(p.replace("_", " ").title() for p in self.phases)
        return "Phase " + "/".join(nums)

    def context_header(self, section: SectionKind) -> str:
        """Deterministic contextual-retrieval prefix for a chunk of this study.

        Anthropic's contextual retrieval normally costs one LLM call per chunk.
        Here the structured fields already carry the disambiguating context, so
        we synthesise the same signal for $0 -- and it is reproducible, which an
        LLM-generated header would not be.
        """
        bits = [f"{self.nct_id}:", self.brief_title.rstrip(".") + "."]
        design = self.phase_label
        if self.study_type is not StudyType.UNKNOWN:
            design += f" {self.study_type.value.lower()}"
        if self.conditions:
            design += " study of " + ", ".join(self.conditions[:3])
        bits.append(design + ".")
        if self.lead_sponsor:
            bits.append(f"Sponsor: {self.lead_sponsor}.")
        bits.append(f"Status: {self.overall_status.value.replace('_', ' ').lower()}.")
        bits.append(f"Section: {section.value.replace('_', ' ')}.")
        return " ".join(bits)


class Section(Frozen):
    """A named free-text region of a protocol, before chunking."""

    nct_id: NctId
    kind: SectionKind
    text: str
    ordinal: int = 0
    label: str | None = None


class ChunkCandidate(Frozen):
    """Chunker output: text plus provenance, not yet embedded or persisted."""

    nct_id: NctId
    kind: SectionKind
    ordinal: int
    content: str
    context_header: str
    token_count: int
    label: str | None = None

    @property
    def embedding_input(self) -> str:
        """What actually gets embedded: header + body.

        Both the dense vector and the tsvector are built from this same string,
        so the two retrieval arms see identical text. Divergence there produces
        fusion bugs that are invisible until recall quietly drops.
        """
        return f"{self.context_header}\n\n{self.content}"


class Chunk(ChunkCandidate):
    """A persisted chunk."""

    id: int
    content_hash: str


class RetrievedChunk(Frozen):
    """A chunk plus the full score trail that selected it.

    Every arm's contribution is kept rather than collapsed into one number: the
    debug UI renders it, the ablation harness reads it, and "why did this rank
    here" stops being guesswork.
    """

    chunk_id: int
    nct_id: NctId
    kind: SectionKind
    content: str
    context_header: str
    study_title: str

    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int = 0

    @property
    def cited_text(self) -> str:
        return self.content

    @property
    def source_url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"
