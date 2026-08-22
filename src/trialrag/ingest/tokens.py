"""Token counting for chunk sizing.

Chunk boundaries are an *embedding-model* concern, so they are measured with
the embedding model's own tokenizer -- not Anthropic's, and not ``tiktoken``,
which belongs to a third vendor entirely and would be wrong for both. Anthropic's
``messages.count_tokens`` is used elsewhere, for prompt assembly and cost
accounting, where it is the correct instrument.

Voyage's tokenizer is a Hugging Face tokenizer fetched once and cached on disk;
after that it runs entirely offline. Because that first fetch needs a network,
and hermetic CI does not have one, this module degrades to a deterministic
heuristic rather than failing. The degradation is explicit and observable --
:func:`get_counter` reports which implementation it returned -- because a
silently-wrong token count shifts every chunk boundary in the corpus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    """Counts tokens in a string under some model's tokenizer."""

    name: str

    def __call__(self, text: str) -> int: ...


@dataclass
class VoyageTokenCounter:
    """Exact counts from Voyage's published tokenizer."""

    model: str
    name: str = "voyage"

    def __post_init__(self) -> None:
        import voyageai

        # The tokenizer is local; the key is never used for tokenisation, but
        # the client requires one to construct.
        self._client = voyageai.Client(api_key="tokenizer-only")
        self._encode = self._client.tokenizer(self.model).encode

    def __call__(self, text: str) -> int:
        return len(self._encode(text).ids)


_WORD_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class HeuristicTokenCounter:
    """Offline fallback: word/punctuation count scaled by an empirical factor.

    Subword tokenizers emit roughly 1.3 tokens per whitespace-delimited word on
    clinical prose, which is denser than general English because of drug names
    and dosage strings. The factor errs high on purpose: over-counting yields
    slightly smaller chunks, which is a benign failure, while under-counting can
    push a chunk past the embedding model's limit and truncate it silently.
    """

    factor: float = 1.35
    name: str = "heuristic"

    def __call__(self, text: str) -> int:
        return int(len(_WORD_RE.findall(text)) * self.factor) + 1


@lru_cache(maxsize=4)
def get_counter(model: str, *, allow_download: bool = True) -> TokenCounter:
    """Return the best available counter for ``model``.

    Cached, because constructing the Voyage tokenizer costs ~1s and chunking
    calls this once per section across tens of thousands of records.
    """
    if allow_download:
        try:
            counter = VoyageTokenCounter(model=model)
            counter("warmup")  # force the lazy fetch here, not mid-corpus
        except Exception as exc:  # any failure at all means fall back
            logger.warning(
                "Voyage tokenizer unavailable for %s (%s: %s); using heuristic counts. "
                "Chunk boundaries will differ from a networked run.",
                model,
                type(exc).__name__,
                exc,
            )
        else:
            return counter
    return HeuristicTokenCounter()
