# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sliding-window fallback chunker for Mnemosyne.

Used for any language that the :class:`~mnemosyne.chunkers.code_chunker.CodeChunker`
and :class:`~mnemosyne.chunkers.text_chunker.TextChunker` do not handle
specifically (e.g. JSON, YAML, TOML, HTML, CSS, shell, SQL when no regex
boundaries fire).

Produces fixed-size chunks with configurable line-level overlap so that
context is not lost at chunk boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemosyne.models import estimate_tokens
from mnemosyne.chunkers.code_chunker import ChunkCandidate

if TYPE_CHECKING:
    from mnemosyne.config import Config


class GenericChunker:
    """
    Fixed-size sliding-window chunker with line-level overlap.

    Algorithm:
    1. Split source into lines.
    2. Walk forward, accumulating lines until the token budget is reached.
    3. Emit the current window as a chunk.
    4. Step back by ``overlap_lines`` so the next chunk shares context.
    5. Repeat until all lines are consumed.

    If the final fragment is smaller than ``min_tokens``, it is merged into
    the previous chunk (unless that would exceed ``max_tokens``).
    """

    def __init__(self, config: "Config") -> None:
        self.max_tokens: int = config.chunking.max_chunk_tokens
        self.min_tokens: int = config.chunking.min_chunk_tokens
        self.overlap_lines: int = config.chunking.overlap_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, language: str = "unknown") -> list[ChunkCandidate]:
        """
        Split *source* into overlapping fixed-size :class:`ChunkCandidate` objects.

        Args:
            source:   Full source text.
            language: Language tag (informational only; does not affect behavior).

        Returns:
            Ordered list of chunk candidates with line-level overlap.
        """
        if not source.strip():
            return []

        lines = source.splitlines(keepends=True)

        # If the whole file fits in one chunk, return it directly.
        if estimate_tokens(source) <= self.max_tokens:
            return [
                ChunkCandidate(
                    content=source,
                    chunk_type="generic",
                    line_start=1,
                    line_end=len(lines),
                )
            ]

        return self._sliding_window(lines)

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _sliding_window(self, lines: list[str]) -> list[ChunkCandidate]:
        """
        Walk *lines* with a sliding window, emitting chunks at the token budget.

        Returns:
            List of :class:`ChunkCandidate` instances.
        """
        candidates: list[ChunkCandidate] = []
        n = len(lines)
        pos = 0  # current start index (0-based)

        while pos < n:
            window: list[str] = []
            end = pos  # exclusive end index (0-based)

            # Accumulate lines until we hit the token budget
            while end < n:
                window.append(lines[end])
                end += 1
                if estimate_tokens("".join(window)) >= self.max_tokens:
                    break

            # 'end' is now the exclusive end of this chunk (0-based)
            content = "".join(window)
            line_start = pos + 1           # 1-based
            line_end = end                 # inclusive 1-based (= exclusive 0-based)

            candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type="generic",
                    line_start=line_start,
                    line_end=line_end,
                )
            )

            if end >= n:
                # Consumed all remaining lines
                break

            # Step forward, stepping back by overlap so next chunk shares context.
            advance = max(1, len(window) - self.overlap_lines)
            pos += advance

        # Merge a final tiny fragment into its predecessor when possible
        if len(candidates) >= 2:
            last = candidates[-1]
            if estimate_tokens(last.content) < self.min_tokens:
                prev = candidates[-2]
                combined = prev.content + last.content
                if estimate_tokens(combined) <= self.max_tokens:
                    candidates[-2] = ChunkCandidate(
                        content=combined,
                        chunk_type=prev.chunk_type,
                        line_start=prev.line_start,
                        line_end=last.line_end,
                    )
                    candidates.pop()

        return candidates
