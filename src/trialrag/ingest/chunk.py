"""Section -> chunk conversion.

Two strategies, chosen by section kind, because clinical protocols contain two
genuinely different kinds of text and treating them alike costs measurable
recall:

**Enumerated criteria** (inclusion/exclusion lists) are a set of independent
propositions. A criterion split across two chunks produces a fragment that is
not merely less useful but *wrong* -- half of "no history of cardiovascular
disease requiring intervention within 6 months" retrieves as a different claim
than the whole. So criteria are packed greedily and never split, and only a
criterion that alone exceeds the budget is broken down further.

Notably these chunks carry **no overlap**. Overlap is a hedge against splitting
mid-thought, which packing already prevents; applying it to a list instead
duplicates whole criteria into adjacent chunks, so the same proposition
competes with itself for top-k slots and crowds out genuinely distinct ones.

**Prose** (summaries, descriptions, outcome definitions) is continuous argument
where a hard split really can sever a referent, so it gets recursive splitting
on a separator hierarchy with overlap.

Both paths emit chunks carrying the study's deterministic context header; see
:meth:`trialrag.domain.models.Study.context_header`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from trialrag.domain.models import ChunkCandidate, Section, SectionKind, Study
from trialrag.ingest.tokens import TokenCounter

# Bullets seen in the wild: the registry's own renderer emits "*", submitters
# paste en/em dashes, middots and unicode bullets, plus numbered and lettered
# lists. The "ambiguous unicode" lint is suppressed deliberately -- these exact
# codepoints appear in real records, and normalising them away loses splits.
_BULLET_RE: Final = re.compile(
    r"^[ \t]*(?:[*\-–—•·]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+",  # noqa: RUF001
    re.M,
)

# Prose separator hierarchy, most-preferred first.
_SEPARATORS: Final[tuple[str, ...]] = ("\n\n", "\n", ". ", "; ", ", ", " ")

_SENTENCE_RE: Final = re.compile(r"(?<=[.!?])\s+")

ELIGIBILITY_KINDS: Final[frozenset[SectionKind]] = frozenset(
    {
        SectionKind.ELIGIBILITY_INCLUSION,
        SectionKind.ELIGIBILITY_EXCLUSION,
        SectionKind.ELIGIBILITY_OTHER,
    }
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking knobs.

    Exposed as a value object rather than settings so the ablation harness can
    sweep it (``evals/ablations.py``) without mutating process configuration.
    """

    max_tokens: int = 400
    min_tokens: int = 32
    overlap_tokens: int = 60
    strategy: str = "structural"  # "structural" | "flat" -- flat is the ablation baseline

    def __post_init__(self) -> None:
        if self.max_tokens <= self.min_tokens:
            raise ValueError("max_tokens must exceed min_tokens")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")


DEFAULT_CONFIG: Final = ChunkingConfig()


# ---------------------------------------------------------------------------
# Criterion splitting
# ---------------------------------------------------------------------------


def split_criteria(text: str) -> list[str]:
    """Break an eligibility block into individual criteria.

    Falls back to blank-line paragraphs, then to lines, when no bullet markers
    are present -- about a fifth of records write criteria as plain prose.
    """
    if _BULLET_RE.search(text):
        pieces = _BULLET_RE.split(text)
    elif "\n\n" in text:
        pieces = text.split("\n\n")
    else:
        pieces = text.split("\n")

    return [cleaned for piece in pieces if (cleaned := " ".join(piece.split()))]


# ---------------------------------------------------------------------------
# Prose splitting
# ---------------------------------------------------------------------------


def _split_recursive(text: str, budget: int, count: TokenCounter) -> list[str]:
    """Split ``text`` into pieces under ``budget`` tokens, preferring big seams.

    Tries each separator in turn and only descends to a finer one when a piece
    is still oversized, so a paragraph is never broken at a comma when a
    sentence break would have done.
    """
    if count(text) <= budget:
        return [text]

    for separator in _SEPARATORS:
        if separator not in text:
            continue
        parts = [p for p in text.split(separator) if p.strip()]
        if len(parts) < 2:
            continue

        out: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if count(candidate) <= budget:
                buffer = candidate
                continue
            if buffer:
                out.append(buffer)
            # A single part over budget needs the next, finer separator.
            out.extend(_split_recursive(part, budget, count) if count(part) > budget else [part])
            buffer = "" if count(part) > budget else part
        if buffer:
            out.append(buffer)
        return [piece for piece in out if piece.strip()]

    # No separator left: hard-cut on whitespace-free text (long identifiers,
    # tables pasted without spaces). Rare, but must terminate.
    approx_chars = max(1, len(text) * budget // max(1, count(text)))
    return [text[i : i + approx_chars] for i in range(0, len(text), approx_chars)]


def _overlap_tail(text: str, overlap: int, count: TokenCounter) -> str:
    """Last ``overlap`` tokens of ``text``, snapped to a sentence boundary.

    Snapping matters: a raw token-count tail starts mid-clause, and that
    fragment gets embedded as though it were a statement.
    """
    if overlap <= 0 or count(text) <= overlap:
        return ""
    sentences = _SENTENCE_RE.split(text)
    tail: list[str] = []
    for sentence in reversed(sentences):
        tail.insert(0, sentence)
        if count(" ".join(tail)) >= overlap:
            break
    return " ".join(tail).strip()


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _pack(
    units: Sequence[str],
    config: ChunkingConfig,
    count: TokenCounter,
    *,
    joiner: str,
    overlap: bool,
) -> Iterator[str]:
    """Greedily fill chunks from atomic ``units`` without splitting any unit.

    A unit larger than ``max_tokens`` on its own is recursively split first, so
    the no-split guarantee holds for every unit that *can* fit.
    """
    buffer: list[str] = []

    for unit in units:
        if count(unit) > config.max_tokens:
            if buffer:
                yield joiner.join(buffer)
                buffer = []
            yield from _split_recursive(unit, config.max_tokens, count)
            continue

        if buffer and count(joiner.join([*buffer, unit])) > config.max_tokens:
            emitted = joiner.join(buffer)
            yield emitted
            # The carry is a hedge, never a budget overrun: if prepending it
            # would push this unit over on its own, the overlap is dropped.
            # Silently keeping it is how oversized chunks reach the embedding
            # API and get truncated without an error.
            carry = _overlap_tail(emitted, config.overlap_tokens, count) if overlap else ""
            buffer = (
                [carry, unit]
                if carry and count(joiner.join([carry, unit])) <= config.max_tokens
                else [unit]
            )
        else:
            buffer.append(unit)

    if buffer:
        yield joiner.join(buffer)


def _merge_slivers(chunks: list[str], config: ChunkingConfig, count: TokenCounter) -> list[str]:
    """Fold sub-``min_tokens`` chunks into a neighbour.

    A three-token chunk ("* None") embeds to near-noise and pollutes top-k for
    every query. NCT03840798's exclusion section is literally that.
    """
    if len(chunks) <= 1:
        return chunks
    out: list[str] = []
    for chunk in chunks:
        if out and count(chunk) < config.min_tokens:
            merged = f"{out[-1]}\n{chunk}"
            if count(merged) <= config.max_tokens:
                out[-1] = merged
                continue
        out.append(chunk)
    # A leading sliver has no predecessor to merge into; push it into its successor.
    if len(out) > 1 and count(out[0]) < config.min_tokens:
        merged = f"{out[0]}\n{out[1]}"
        if count(merged) <= config.max_tokens:
            out = [merged, *out[2:]]
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def chunk_section(
    section: Section,
    study: Study,
    count: TokenCounter,
    config: ChunkingConfig = DEFAULT_CONFIG,
    *,
    start_ordinal: int = 0,
) -> list[ChunkCandidate]:
    """Convert one section into chunk candidates."""
    text = section.text.strip()
    if not text:
        return []

    structural = config.strategy == "structural"
    if structural and section.kind in ELIGIBILITY_KINDS:
        units = split_criteria(text)
        pieces = list(_pack(units, config, count, joiner="\n", overlap=False))
    else:
        units = [p for p in text.split("\n\n") if p.strip()] or [text]
        pieces = list(_pack(units, config, count, joiner="\n\n", overlap=True))

    pieces = _merge_slivers(pieces, config, count)
    header = study.context_header(section.kind)

    return [
        ChunkCandidate(
            nct_id=section.nct_id,
            kind=section.kind,
            ordinal=start_ordinal + offset,
            content=piece.strip(),
            context_header=header,
            token_count=count(piece),
            label=section.label,
        )
        for offset, piece in enumerate(pieces)
        if piece.strip()
    ]


def chunk_study(
    sections: Iterable[Section],
    study: Study,
    count: TokenCounter,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> list[ChunkCandidate]:
    """Chunk every section of a study, assigning study-wide ordinals.

    Ordinals are unique per study rather than per section so that ``(nct_id,
    ordinal)`` is a stable primary key for upserts across re-ingests.
    """
    out: list[ChunkCandidate] = []
    for section in sections:
        out.extend(chunk_section(section, study, count, config, start_ordinal=len(out)))
    return out
