"""ClinicalTrials.gov API v2 JSON -> canonical domain objects.

This is the *only* module in the codebase that knows the registry's wire format.
Everything downstream consumes :class:`Study` and :class:`Section`. That
containment is deliberate: the registry adds and renames modules without
notice, and when it does we want a loud failure in one parser test rather than
a quiet hole in the corpus.

Two properties the API has that bite naive parsers:

* **Absent, not null.** Optional modules and fields are *omitted* entirely. A
  ``.get(...)`` chain that assumes presence raises; one that assumes ``None``
  silently yields empty studies. Every access here goes through :func:`_dig`.
* **Human-formatted scalars.** Ages arrive as ``"12 Years"``, booleans as the
  strings ``"True"``/``"False"``, dates sometimes as ``"2026-03"`` with no day.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final

from trialrag.domain.models import (
    AgeRange,
    Intervention,
    Outcome,
    Section,
    SectionKind,
    Sex,
    Study,
    StudyStatus,
    StudyType,
)

JsonMap = Mapping[str, Any]

# Registry-side sentinels that mean "no value" but arrive as real strings.
_NULLISH: Final[frozenset[str]] = frozenset({"", "n/a", "na", "none", "null", "-"})


class ParseError(ValueError):
    """A record could not be parsed into the domain model.

    Raised only for defects that make a record unusable (no NCT ID, no title).
    Missing optional data is represented as ``None``, never as an exception --
    otherwise one incomplete registry entry aborts a 5,000-study ingest.
    """


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------


def _dig(obj: Any, *path: str) -> Any:
    """Walk a nested mapping, returning ``None`` at the first missing key.

    ``_dig(study, "protocolSection", "designModule", "phases")`` rather than a
    chain of ``.get("x", {})`` calls, which quietly returns ``{}`` on a type
    mismatch and hides schema drift.
    """
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _text(obj: Any, *path: str) -> str | None:
    """Dig for a string, normalising registry null-sentinels to ``None``."""
    val = _dig(obj, *path)
    if not isinstance(val, str):
        return None
    stripped = val.strip()
    return None if stripped.lower() in _NULLISH else stripped


def _tuple(obj: Any, *path: str) -> tuple[str, ...]:
    """Dig for a list of strings; anything else becomes an empty tuple."""
    val = _dig(obj, *path)
    if not isinstance(val, Sequence) or isinstance(val, str):
        return ()
    return tuple(item.strip() for item in val if isinstance(item, str) and item.strip())


def _int(obj: Any, *path: str) -> int | None:
    val = _dig(obj, *path)
    if isinstance(val, bool):  # bool is an int subclass; never an enrollment count
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val.strip())
        except ValueError:
            return None
    return None


def _bool(obj: Any, *path: str) -> bool | None:
    """The API ships booleans as the *strings* ``"True"`` / ``"False"``."""
    val = _dig(obj, *path)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        lowered = val.strip().lower()
        if lowered in ("true", "yes"):
            return True
        if lowered in ("false", "no"):
            return False
    return None


def _date(obj: Any, *path: str) -> dt.date | None:
    """Parse ``"2020-04-29"`` or the month-only ``"2026-03"`` form.

    Month-only values are anchored to the first of the month. That is a lossy
    but monotonic choice: it keeps date ordering correct, which is all the
    filters need, and the exact string survives in the archived raw payload.
    """
    raw = _text(obj, *path)
    if raw is None:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return dt.datetime.strptime(raw, fmt).replace(tzinfo=dt.UTC).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def source_hash(raw: JsonMap) -> str:
    """Stable content hash of a raw record.

    Keys are sorted so that a re-serialisation with different ordering does not
    look like a content change and trigger a needless re-embed.
    """
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_study(raw: JsonMap, *, raw_s3_key: str | None = None) -> Study:
    """Build a :class:`Study` from one API v2 record.

    Args:
        raw: A single element of ``studies[]``, or the body of ``/studies/{id}``.
        raw_s3_key: Where the untouched payload was archived, for provenance.

    Raises:
        ParseError: The record lacks an NCT ID or a title.
    """
    protocol = _dig(raw, "protocolSection")
    if not isinstance(protocol, Mapping):
        raise ParseError("record has no protocolSection")

    nct_id = _text(protocol, "identificationModule", "nctId")
    if nct_id is None:
        raise ParseError("record has no nctId")

    brief_title = _text(protocol, "identificationModule", "briefTitle")
    if brief_title is None:
        # Rare, but real: withdrawn records occasionally carry only an official
        # title. Fall back before giving up -- a title is required downstream
        # for the context header.
        brief_title = _text(protocol, "identificationModule", "officialTitle")
    if brief_title is None:
        raise ParseError(f"{nct_id}: record has no title")

    countries = tuple(
        dict.fromkeys(  # preserve order, drop duplicates across 100s of sites
            country
            for loc in _dig(protocol, "contactsLocationsModule", "locations") or ()
            if isinstance(loc, Mapping) and (country := _text(loc, "country"))
        )
    )

    return Study(
        nct_id=nct_id,
        brief_title=brief_title,
        official_title=_text(protocol, "identificationModule", "officialTitle"),
        overall_status=StudyStatus.coerce(_text(protocol, "statusModule", "overallStatus")),
        study_type=StudyType.coerce(_text(protocol, "designModule", "studyType")),
        phases=_tuple(protocol, "designModule", "phases"),
        conditions=_tuple(protocol, "conditionsModule", "conditions"),
        keywords=_tuple(protocol, "conditionsModule", "keywords"),
        interventions=_parse_interventions(protocol),
        lead_sponsor=_text(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        sponsor_class=_text(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class"),
        enrollment=_int(protocol, "designModule", "enrollmentInfo", "count"),
        enrollment_type=_text(protocol, "designModule", "enrollmentInfo", "type"),
        age_range=AgeRange.from_raw(
            _text(protocol, "eligibilityModule", "minimumAge"),
            _text(protocol, "eligibilityModule", "maximumAge"),
        ),
        sex=Sex.coerce(_text(protocol, "eligibilityModule", "sex")),
        healthy_volunteers=_bool(protocol, "eligibilityModule", "healthyVolunteers"),
        std_ages=_tuple(protocol, "eligibilityModule", "stdAges"),
        allocation=_text(protocol, "designModule", "designInfo", "allocation"),
        intervention_model=_text(protocol, "designModule", "designInfo", "interventionModel"),
        primary_purpose=_text(protocol, "designModule", "designInfo", "primaryPurpose"),
        masking=_text(protocol, "designModule", "designInfo", "maskingInfo", "masking"),
        start_date=_date(protocol, "statusModule", "startDateStruct", "date"),
        completion_date=_date(protocol, "statusModule", "completionDateStruct", "date"),
        last_update_posted=_date(protocol, "statusModule", "lastUpdatePostDateStruct", "date"),
        countries=countries,
        location_count=len(_dig(protocol, "contactsLocationsModule", "locations") or ()),
        has_results=bool(_dig(raw, "hasResults")),
        source_hash=source_hash(raw),
        raw_s3_key=raw_s3_key,
    )


def _parse_interventions(protocol: JsonMap) -> tuple[Intervention, ...]:
    items = _dig(protocol, "armsInterventionsModule", "interventions") or ()
    out: list[Intervention] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = _text(item, "name")
        if name is None:
            continue
        out.append(
            Intervention(
                type=_text(item, "type") or "OTHER",
                name=name,
                description=_text(item, "description"),
                other_names=_tuple(item, "otherNames"),
            )
        )
    return tuple(out)


def _parse_outcomes(protocol: JsonMap, key: str) -> tuple[Outcome, ...]:
    items = _dig(protocol, "outcomesModule", key) or ()
    out: list[Outcome] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        measure = _text(item, "measure")
        if measure is None:
            continue
        out.append(
            Outcome(
                measure=measure,
                description=_text(item, "description"),
                time_frame=_text(item, "timeFrame"),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

# Eligibility criteria are semi-structured prose. The registry has no schema for
# them, but submitters overwhelmingly follow one of these header conventions.
_INCLUSION_RE: Final = re.compile(
    r"^\s*[*\-•#]*\s*(key\s+)?inclusion(\s+criteria)?\s*:?\s*$", re.I | re.M
)
_EXCLUSION_RE: Final = re.compile(
    r"^\s*[*\-•#]*\s*(key\s+)?exclusion(\s+criteria)?\s*:?\s*$", re.I | re.M
)


def split_eligibility(text: str) -> list[tuple[SectionKind, str]]:
    """Split a raw ``eligibilityCriteria`` blob into inclusion/exclusion parts.

    Returning ``ELIGIBILITY_OTHER`` for unheaded text is intentional. Roughly a
    fifth of records write criteria as undifferentiated prose, and guessing a
    polarity for them would inject false negations into the index -- a far worse
    failure than an honestly-unlabelled section.
    """
    inc = _INCLUSION_RE.search(text)
    exc = _EXCLUSION_RE.search(text)

    if inc is None and exc is None:
        body = text.strip()
        return [(SectionKind.ELIGIBILITY_OTHER, body)] if body else []

    # Boundaries are the header matches, in document order.
    marks: list[tuple[int, int, SectionKind]] = []
    if inc is not None:
        marks.append((inc.start(), inc.end(), SectionKind.ELIGIBILITY_INCLUSION))
    if exc is not None:
        marks.append((exc.start(), exc.end(), SectionKind.ELIGIBILITY_EXCLUSION))
    marks.sort()

    out: list[tuple[SectionKind, str]] = []

    preamble = text[: marks[0][0]].strip()
    if preamble:
        out.append((SectionKind.ELIGIBILITY_OTHER, preamble))

    for idx, (_, header_end, kind) in enumerate(marks):
        body_end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        body = text[header_end:body_end].strip()
        if body:
            out.append((kind, body))
    return out


def iter_sections(raw: JsonMap, study: Study) -> Iterator[Section]:
    """Yield every free-text region of a protocol worth indexing.

    Structured scalars (phase, enrollment, sponsor) are deliberately *not*
    emitted as sections. They live in ``studies`` columns where they can be
    filtered on exactly, and they reach the model through each chunk's context
    header -- indexing them as prose would only add near-duplicate noise to the
    retrieval pool.
    """
    protocol = _dig(raw, "protocolSection")
    if not isinstance(protocol, Mapping):
        return

    nct_id = study.nct_id
    ordinal = 0

    def emit(kind: SectionKind, text: str | None, label: str | None = None) -> Iterator[Section]:
        nonlocal ordinal
        if text and text.strip():
            yield Section(nct_id=nct_id, kind=kind, text=text.strip(), ordinal=ordinal, label=label)
            ordinal += 1

    yield from emit(SectionKind.BRIEF_SUMMARY, _text(protocol, "descriptionModule", "briefSummary"))
    yield from emit(
        SectionKind.DETAILED_DESCRIPTION,
        _text(protocol, "descriptionModule", "detailedDescription"),
    )

    criteria = _text(protocol, "eligibilityModule", "eligibilityCriteria")
    if criteria:
        for kind, body in split_eligibility(criteria):
            yield from emit(kind, body)

    for arm in _dig(protocol, "armsInterventionsModule", "armGroups") or ():
        if not isinstance(arm, Mapping):
            continue
        label = _text(arm, "label")
        description = _text(arm, "description")
        if description:
            arm_type = _text(arm, "type") or "arm"
            yield from emit(
                SectionKind.ARM_GROUP,
                f"{label or 'Study arm'} ({arm_type.lower().replace('_', ' ')}): {description}",
                label,
            )

    for intervention in study.interventions:
        if intervention.description:
            yield from emit(
                SectionKind.INTERVENTION,
                f"{intervention.type.title()}: {intervention.name}. {intervention.description}",
                intervention.name,
            )

    for key, kind in (
        ("primaryOutcomes", SectionKind.PRIMARY_OUTCOME),
        ("secondaryOutcomes", SectionKind.SECONDARY_OUTCOME),
    ):
        for outcome in _parse_outcomes(protocol, key):
            parts = [outcome.measure]
            if outcome.description:
                parts.append(outcome.description)
            if outcome.time_frame:
                parts.append(f"Time frame: {outcome.time_frame}")
            yield from emit(kind, ". ".join(parts), outcome.measure)
