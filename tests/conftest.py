"""Shared fixtures.

The study fixtures are unmodified ClinicalTrials.gov API v2 responses (bar the
2 MB ``resultsSection`` of NCT04368728, which no parser path touches). Testing
against synthetic JSON would only prove the parser agrees with our imagination
of the schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="session")
def covid_vaccine_study() -> dict[str, Any]:
    """NCT04368728 — Pfizer/BioNTech COVID-19 vaccine trial.

    Stresses scale and partial data: Phase 2/3, 175 locations, 40+ outcomes,
    a minimum age but *no* maximum, and no ``detailedDescription``.
    """
    return _load("study_nct04368728.json")


@pytest.fixture(scope="session")
def bounded_age_study() -> dict[str, Any]:
    """NCT00000102 — small Phase 1/2 study with both an age floor and ceiling."""
    return _load("study_nct00000102.json")


@pytest.fixture(scope="session")
def observational_study() -> dict[str, Any]:
    """NCT03840798 — observational, so no ``phases``, and a maximum age only."""
    return _load("study_nct03840798.json")


@pytest.fixture(scope="session")
def all_studies(
    covid_vaccine_study: dict[str, Any],
    bounded_age_study: dict[str, Any],
    observational_study: dict[str, Any],
) -> list[dict[str, Any]]:
    return [covid_vaccine_study, bounded_age_study, observational_study]
