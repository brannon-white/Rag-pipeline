"""Provider-agnostic generation interface.

``AnthropicProvider`` is the only implementation today, but nothing in the API
layer imports the ``anthropic`` package directly -- it only sees
:class:`GenerationEvent` and :class:`TokenUsage`. Swapping in a Bedrock-backed
provider later (same model family, different transport) is a config change,
not a rewrite of every call site.

Deliberately smaller than the interface sketched in the original project plan:
there is no ``parse_structured`` method here. Structured query-parsing already
has its own standalone implementation in
:class:`trialrag.retrieval.filters.QueryParser`, which nothing here calls
through this protocol -- adding a method no caller uses would be speculative.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from trialrag.config import Effort
from trialrag.domain.models import RetrievedChunk


class TokenUsage(BaseModel):
    """Provider-agnostic token accounting, for cost computation and logging."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class TextDelta(BaseModel):
    """A chunk of streamed answer text."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class CitationDelta(BaseModel):
    """A citation the model attached to the text just streamed.

    ``chunk_id`` is resolved from the citation's ``document_index`` back to the
    :class:`RetrievedChunk` that was passed as the corresponding ``document``
    content block -- see :func:`trialrag.generation.anthropic_provider.build_document_blocks`,
    which is what fixes that ordering.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["citation"] = "citation"
    cited_text: str
    chunk_id: int
    nct_id: str
    source_url: str
    start_char: int
    end_char: int


class Done(BaseModel):
    """Terminal event: the stream is finished, with usage/cost/outcome."""

    model_config = ConfigDict(frozen=True)

    type: Literal["done"] = "done"
    stop_reason: str
    abstained: bool
    usage: TokenUsage
    cost_usd: float


GenerationEvent = TextDelta | CitationDelta | Done


class LLMProvider(Protocol):
    """What the API layer needs from a generation backend."""

    def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        model: str,
        effort: Effort,
        max_tokens: int,
    ) -> AsyncIterator[GenerationEvent]:
        """Stream an answer to ``query`` grounded in ``chunks``.

        ``chunks`` are the only source of truth the model may cite from --
        implementations must pass them as isolated document content, never
        interpolated into an instruction string, so retrieved text can never
        be read as a command (prompt-injection containment falls out of the
        content-block boundary rather than needing a separate filter).
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Best-effort token count, for pre-flight budget checks."""
        ...

    def price(self, usage: TokenUsage, model: str) -> float:
        """Dollar cost of ``usage`` against ``model``'s published rates."""
        ...
