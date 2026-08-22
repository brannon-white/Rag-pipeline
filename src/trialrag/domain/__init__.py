"""Canonical domain model.

The rest of the system talks in these types, never in raw ClinicalTrials.gov
JSON. That boundary is the whole point: the registry's schema is not ours to
control, so exactly one module (``ingest.parse``) knows its shape, and a
breaking upstream change surfaces as a parser test failure rather than as
silently-missing data three layers down.
"""

from trialrag.domain.models import (
    AgeRange,
    Chunk,
    ChunkCandidate,
    Intervention,
    Outcome,
    RetrievedChunk,
    Section,
    SectionKind,
    Sex,
    Study,
    StudyStatus,
    StudyType,
)

__all__ = [
    "AgeRange",
    "Chunk",
    "ChunkCandidate",
    "Intervention",
    "Outcome",
    "RetrievedChunk",
    "Section",
    "SectionKind",
    "Sex",
    "Study",
    "StudyStatus",
    "StudyType",
]
