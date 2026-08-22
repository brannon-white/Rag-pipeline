"""Thumbs up/down on a past query."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from trialrag.api.deps import get_db
from trialrag.api.schemas import FeedbackRequest, FeedbackResponse
from trialrag.db.pool import Database

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    body: FeedbackRequest, db: Database = Depends(get_db)  # noqa: B008 - FastAPI's documented pattern
) -> FeedbackResponse:
    row = await db.fetchrow(
        "SELECT id FROM query_log WHERE id = $1", body.query_log_id
    )
    if row is None:
        raise HTTPException(status_code=404, detail="query_log_id not found")

    feedback_id = await db.fetchval(
        "INSERT INTO feedback (query_log_id, rating, comment) VALUES ($1, $2, $3) RETURNING id",
        body.query_log_id,
        body.rating,
        body.comment,
    )
    return FeedbackResponse(id=feedback_id)
