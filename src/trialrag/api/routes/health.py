"""Liveness/readiness.

``/healthz`` is deliberately shallow -- no DB touch -- because a deep check
here would pin Neon's compute awake against its own scale-to-zero suspend
window (see ``db/pool.py``'s module docstring on the same cost model).
``/readyz`` is the deep one, gated on ``Database.healthy()``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from trialrag.api.deps import get_db
from trialrag.api.schemas import ReadyStatus
from trialrag.db.pool import Database

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> JSONResponse:
    # TEMPORARY: deliberately broken to verify deploy.yml's rollback path
    # fires for real. Will be reverted immediately after confirming it.
    return JSONResponse(content={"status": "broken on purpose"}, status_code=500)


@router.get("/readyz", response_model=ReadyStatus)
async def readyz(db: Database = Depends(get_db)) -> JSONResponse:  # noqa: B008
    healthy = await db.healthy()
    status = ReadyStatus(database=healthy)
    return JSONResponse(content=status.model_dump(), status_code=200 if healthy else 503)
