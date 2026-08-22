"""Parser tests, run against unmodified API v2 payloads.

These double as schema-drift detection: if ClinicalTrials.gov renames a module,
re-recording the fixtures makes these fail loudly instead of letting the corpus
quietly lose a field.
"""

from __future__ import annotations

from typing import Any

import pytest

from trialrag.domain.models import SectionKind, Sex, StudyStatus, StudyType
from trialrag.ingest.parse import (
    ParseError,
    iter_sections,
    parse_study,
    source_hash,
    split_eligibility,
)

# ---------------------------------------------------------------------------
# Study-level parsing
# ---------------------------------------------------------------------------


def test_parses_headline_fields(covid_vaccine_study: dict[str, Any]) -> None:
    study = parse_study(covid_vaccine_study)

    assert study.nct_id == "NCT04368728"
    assert study.overall_status is StudyStatus.COMPLETED
    assert study.study_type is StudyType.INTERVENTIONAL
    assert study.phases == ("PHASE2", "PHASE3")
    assert study.lead_sponsor == "BioNTech SE"
    assert study.enrollment == 46_969
    assert study.sex is Sex.ALL
    assert study.healthy_volunteers is True
    assert study.masking == "TRIPLE"
    assert study.allocation == "RANDOMIZED"


def test_age_strings_normalise_to_years(
    covid_vaccine_study: dict[str, Any],
    bounded_age_study: dict[str, Any],
    observational_study: dict[str, Any],
) -> None:
    # "12 Years" with no maximum -> open-ended upper bound.
    covid = parse_study(covid_vaccine_study).age_range
    assert covid.min_years == 12.0
    assert covid.max_years is None
    assert covid.includes(80.0)

    bounded = parse_study(bounded_age_study).age_range
    assert (bounded.min_years, bounded.max_years) == (14.0, 35.0)
    assert not bounded.includes(13.0)
    assert bounded.includes(14.0)
    assert not bounded.includes(36.0)

    # Ceiling only: everyone below it qualifies, including infants.
    observational = parse_study(observational_study).age_range
    assert observational.min_years is None
    assert observational.max_years == 13.0
    assert observational.includes(0.5)
    assert not observational.includes(14.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("18 Years", 18.0),
        ("6 Months", 0.5),
        ("1 Year", 1.0),
        ("30 Days", pytest.approx(30 / 365.25)),
        ("N/A", None),
        ("", None),
        (None, None),
        ("adult", None),
    ],
)
def test_parse_age_units(raw: str | None, expected: float | None) -> None:
    from trialrag.domain.models import AgeRange

    assert AgeRange.parse_age(raw) == expected


def test_string_booleans_are_coerced(covid_vaccine_study: dict[str, Any]) -> None:
    # The API sends "True"/"False" as strings; a truthiness check on the raw
    # value would make "False" read as True.
    assert covid_vaccine_study["protocolSection"]["eligibilityModule"]["healthyVolunteers"] in (
        "True",
        True,
    )
    assert parse_study(covid_vaccine_study).healthy_volunteers is True


def test_missing_phases_do_not_crash_observational(observational_study: dict[str, Any]) -> None:
    study = parse_study(observational_study)
    assert study.study_type is StudyType.OBSERVATIONAL
    assert study.phases == ()
    assert study.phase_label == "unphased"


def test_countries_deduplicated_in_order(covid_vaccine_study: dict[str, Any]) -> None:
    study = parse_study(covid_vaccine_study)
    assert study.location_count == 175
    assert len(study.countries) == len(set(study.countries))
    assert "United States" in study.countries


def test_unknown_enum_values_degrade_not_raise() -> None:
    record = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "T"},
            "statusModule": {"overallStatus": "SOME_NEW_STATUS_2027"},
            "designModule": {"studyType": "SOMETHING_ELSE"},
        }
    }
    study = parse_study(record)
    assert study.overall_status is StudyStatus.UNKNOWN
    assert study.study_type is StudyType.UNKNOWN


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"protocolSection": {}},
        {"protocolSection": {"identificationModule": {"briefTitle": "No ID"}}},
        {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}},
    ],
)
def test_unusable_records_raise_parse_error(record: dict[str, Any]) -> None:
    with pytest.raises(ParseError):
        parse_study(record)


def test_source_hash_is_key_order_independent() -> None:
    a = {"protocolSection": {"x": 1, "y": 2}}
    b = {"protocolSection": {"y": 2, "x": 1}}
    assert source_hash(a) == source_hash(b)
    assert source_hash(a) != source_hash({"protocolSection": {"x": 1, "y": 3}})


def test_phase_label_formatting(
    covid_vaccine_study: dict[str, Any], bounded_age_study: dict[str, Any]
) -> None:
    assert parse_study(covid_vaccine_study).phase_label == "Phase 2/3"
    assert parse_study(bounded_age_study).phase_label == "Phase 1/2"


# ---------------------------------------------------------------------------
# Eligibility splitting
# ---------------------------------------------------------------------------


def test_split_eligibility_separates_polarity() -> None:
    text = (
        "Inclusion Criteria:\n\n"
        "• Age 18 or older\n"
        "• Confirmed diagnosis\n\n"
        "Exclusion Criteria:\n\n"
        "• Pregnancy\n"
        "• Prior therapy\n"
    )
    parts = dict(split_eligibility(text))

    assert "Age 18 or older" in parts[SectionKind.ELIGIBILITY_INCLUSION]
    assert "Pregnancy" in parts[SectionKind.ELIGIBILITY_EXCLUSION]
    # The single most damaging failure mode: exclusion text leaking into the
    # inclusion chunk inverts the meaning of every criterion in it.
    assert "Pregnancy" not in parts[SectionKind.ELIGIBILITY_INCLUSION]
    assert "Age 18 or older" not in parts[SectionKind.ELIGIBILITY_EXCLUSION]


@pytest.mark.parametrize(
    "header",
    [
        "Inclusion Criteria:",
        "INCLUSION CRITERIA",
        "  inclusion criteria :  ",
        "* Inclusion Criteria:",
        "- Key Inclusion Criteria:",
    ],
)
def test_inclusion_header_variants(header: str) -> None:
    parts = dict(split_eligibility(f"{header}\n\nSome criterion here\n"))
    assert SectionKind.ELIGIBILITY_INCLUSION in parts


def test_unheaded_criteria_are_labelled_other_not_guessed() -> None:
    text = "Patients must be ambulatory and have adequate organ function."
    parts = split_eligibility(text)
    assert [kind for kind, _ in parts] == [SectionKind.ELIGIBILITY_OTHER]


def test_preamble_before_first_header_is_preserved() -> None:
    text = "This study enrolls two cohorts.\n\nInclusion Criteria:\n\n• Cohort A only\n"
    parts = dict(split_eligibility(text))
    assert parts[SectionKind.ELIGIBILITY_OTHER] == "This study enrolls two cohorts."
    assert "Cohort A only" in parts[SectionKind.ELIGIBILITY_INCLUSION]


def test_split_eligibility_loses_no_content() -> None:
    text = "Inclusion Criteria:\n\n• A\n• B\n\nExclusion Criteria:\n\n• C\n"
    recovered = " ".join(body for _, body in split_eligibility(text))
    for token in ("• A", "• B", "• C"):
        assert token in recovered


def test_empty_criteria_yields_nothing() -> None:
    assert split_eligibility("   \n\n  ") == []


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


def test_sections_cover_expected_kinds(covid_vaccine_study: dict[str, Any]) -> None:
    study = parse_study(covid_vaccine_study)
    sections = list(iter_sections(covid_vaccine_study, study))
    kinds = {section.kind for section in sections}

    assert SectionKind.BRIEF_SUMMARY in kinds
    assert SectionKind.ELIGIBILITY_INCLUSION in kinds
    assert SectionKind.ELIGIBILITY_EXCLUSION in kinds
    assert SectionKind.PRIMARY_OUTCOME in kinds
    # No detailedDescription on this record -- absence must not fabricate one.
    assert SectionKind.DETAILED_DESCRIPTION not in kinds


def test_section_ordinals_are_unique_and_contiguous(all_studies: list[dict[str, Any]]) -> None:
    for raw in all_studies:
        sections = list(iter_sections(raw, parse_study(raw)))
        ordinals = [section.ordinal for section in sections]
        assert ordinals == list(range(len(sections)))


def test_sections_are_never_blank(all_studies: list[dict[str, Any]]) -> None:
    for raw in all_studies:
        for section in iter_sections(raw, parse_study(raw)):
            assert section.text.strip()
            assert section.nct_id == parse_study(raw).nct_id


def test_context_header_carries_filterable_identity(
    covid_vaccine_study: dict[str, Any],
) -> None:
    study = parse_study(covid_vaccine_study)
    header = study.context_header(SectionKind.ELIGIBILITY_INCLUSION)

    # This header is the entire contextual-retrieval mechanism, and it is what
    # makes a bare chunk like "• Pregnancy" attributable and searchable.
    assert "NCT04368728" in header
    assert "Phase 2/3" in header
    assert "BioNTech SE" in header
    assert "eligibility inclusion" in header


def test_context_header_is_deterministic(covid_vaccine_study: dict[str, Any]) -> None:
    a = parse_study(covid_vaccine_study).context_header(SectionKind.BRIEF_SUMMARY)
    b = parse_study(covid_vaccine_study).context_header(SectionKind.BRIEF_SUMMARY)
    assert a == b
