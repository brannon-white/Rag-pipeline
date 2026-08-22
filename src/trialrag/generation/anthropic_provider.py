"""The only :class:`LLMProvider` implementation: streaming Claude with native
document citations.

Retrieved chunks are passed as isolated ``document`` content blocks, never
string-interpolated into a prompt -- this is the structural prompt-injection
containment ``guardrails.py`` describes, not a detection filter. Citation
spans come back as ``document_index``-addressed events, resolved here to the
originating :class:`RetrievedChunk` via list position, so the API layer never
has to reason about Anthropic's citation shape at all.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import anthropic

from trialrag.config import Effort
from trialrag.domain.models import RetrievedChunk
from trialrag.generation.guardrails import SYSTEM_PROMPT
from trialrag.generation.pricing import compute_cost_usd
from trialrag.generation.provider import CitationDelta, Done, GenerationEvent, TextDelta, TokenUsage

logger = logging.getLogger(__name__)


def build_document_blocks(
    chunks: list[RetrievedChunk],
) -> list[anthropic.types.DocumentBlockParam]:
    """One ``document`` content block per chunk, in list order.

    Order matters: a citation event's ``document_index`` is the position of
    its source block in this exact list, which is how
    :meth:`AnthropicProvider.generate_stream` maps a citation back to a
    ``RetrievedChunk``. Only the last block carries a cache breakpoint --
    caching needs a stable, unchanging prefix, and the retrieved-context
    blocks (unlike the system prompt) change on every query, so caching
    anything but the full run of them would never hit.
    """
    blocks: list[anthropic.types.DocumentBlockParam] = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.content},
            "title": chunk.nct_id,
            "context": f"{chunk.study_title} ({chunk.nct_id}) -- {chunk.kind}",
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


@dataclass
class AnthropicProvider:
    """Wraps ``client.messages.stream()`` for the answer-generation path.

    Same explicit-``api_key`` convention as :class:`QueryParser` and
    :class:`Embedder`: secrets flow through ``config.Settings``, never through
    the SDK's implicit environment lookup.
    """

    api_key: str | None = None
    client: anthropic.AsyncAnthropic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate_stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        model: str,
        effort: Effort,
        max_tokens: int,
    ) -> AsyncIterator[GenerationEvent]:
        documents = build_document_blocks(chunks)
        content: list[anthropic.types.TextBlockParam | anthropic.types.DocumentBlockParam] = [
            *documents,
            {"type": "text", "text": query},
        ]
        messages: list[anthropic.types.MessageParam] = [{"role": "user", "content": content}]
        system: list[anthropic.types.TextBlockParam] = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ]

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            output_config={"effort": effort},
            thinking={"type": "adaptive", "display": "omitted"},
            system=system,
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue
                if event.delta.type == "text_delta":
                    yield TextDelta(text=event.delta.text)
                elif event.delta.type == "citations_delta":
                    citation = event.delta.citation
                    if citation.type != "char_location":
                        continue
                    source_chunk = chunks[citation.document_index]
                    yield CitationDelta(
                        cited_text=citation.cited_text,
                        chunk_id=source_chunk.chunk_id,
                        nct_id=source_chunk.nct_id,
                        source_url=source_chunk.source_url,
                        start_char=citation.start_char_index,
                        end_char=citation.end_char_index,
                    )
            message = await stream.get_final_message()

        usage = TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
        )
        cost = compute_cost_usd(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
        )
        stop_reason = message.stop_reason or "unknown"
        # Only an API-level refusal counts as "abstained" here -- detecting a
        # text-level "the documents don't say" via string matching would be
        # exactly the brittle heuristic already rejected for query screening.
        # Whether the model abstained appropriately in *language* rather than
        # via a hard refusal is what the Tier-2 generation eval measures.
        yield Done(
            stop_reason=stop_reason,
            abstained=stop_reason == "refusal",
            usage=usage,
            cost_usd=cost,
        )

    def count_tokens(self, text: str) -> int:
        # A stable, cheap proxy rather than a live /count_tokens API call on
        # every pre-flight check -- Claude models average close to 4
        # characters/token on English text, which is precise enough for a
        # budget *estimate* (the circuit breaker's authority is the real
        # metered `usage` from each response, not this).
        return max(1, len(text) // 4)

    def price(self, usage: TokenUsage, model: str) -> float:
        return compute_cost_usd(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
        )

    async def aclose(self) -> None:
        await self.client.close()
