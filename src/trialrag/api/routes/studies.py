"""Study record lookup -- the inverse of ``ingest/load.py``'s ``upsert_studies``."""

from __future__ import annotations

import json

import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from trialrag.api.deps import get_db
from trialrag.db.pool import Database
from trialrag.domain.models import AgeRange, Intervention, Sex, Study, StudyStatus, StudyType

router = APIRouter(prefix="/v1", tags=["studies"])


def _row_to_study(row: asyncpg.Record) -> Study:
    return Study(
        nct_id=row["nct_id"],
        brief_title=row["brief_title"],
        official_title=row["official_title"],
        overall_status=StudyStatus.coerce(row["overall_status"]),
        study_type=StudyType.coerce(row["study_type"]),
        phases=tuple(row["phases"]),
        conditions=tuple(row["conditions"]),
        keywords=tuple(row["keywords"]),
        interventions=tuple(Intervention(**i) for i in json.loads(row["interventions"])),
        lead_sponsor=row["lead_sponsor"],
        sponsor_class=row["sponsor_class"],
        enrollment=row["enrollment"],
        enrollment_type=row["enrollment_type"],
        age_range=AgeRange(
            min_years=float(row["min_age_years"]) if row["min_age_years"] is not None else None,
            max_years=float(row["max_age_years"]) if row["max_age_years"] is not None else None,
        ),
        sex=Sex.coerce(row["sex"]),
        healthy_volunteers=row["healthy_volunteers"],
        std_ages=tuple(row["std_ages"]),
        allocation=row["allocation"],
        intervention_model=row["intervention_model"],
        primary_purpose=row["primary_purpose"],
        masking=row["masking"],
        start_date=row["start_date"],
        completion_date=row["completion_date"],
        last_update_posted=row["last_update_posted"],
        countries=tuple(row["countries"]),
        location_count=row["location_count"],
        has_results=row["has_results"],
        source_hash=row["source_hash"],
        raw_s3_key=row["raw_s3_key"],
    )


@router.get("/studies/{nct_id}", response_model=Study)
async def get_study(nct_id: str, db: Database = Depends(get_db)) -> Study:  # noqa: B008
    row = await db.fetchrow("SELECT * FROM studies WHERE nct_id = $1", nct_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"{nct_id} not found")
    return _row_to_study(row)
