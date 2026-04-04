# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Markdown and plain-text chunker for Mnemosyne.

Splits prose documents at structural boundaries (headings for Markdown,
paragraph breaks for plain text) and applies token-budget enforcement by
further splitting at sentence boundaries when a section is too large.

Small fragments are merged with their predecessor to avoid noise chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mnemosyne.models import estimate_tokens
from mnemosyne.chunkers.code_chunker import ChunkCandidate

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Heading pattern: H1 - H3 at the start of a line
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)

# Sentence boundary: period/question/exclamation followed by space + capital,
# or double newline (paragraph break inside a section).
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class TextChunker:
    """
    Heading-aware chunker for Markdown and plain-text prose.
    """

    def __init__(self, config: "Config") -> None:
        self.max_tokens: int = config.chunking.max_chunk_tokens
        self.min_tokens: int = config.chunking.min_chunk_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, language: str = "markdown") -> list[ChunkCandidate]:
        """
        Split *source* into :class:`ChunkCandidate` objects.

        Args:
            source:   Full text of the document.
            language: ``"markdown"`` or ``"text"`` (default ``"markdown"``).

        Returns:
            Ordered list of chunk candidates.
        """
        if not source.strip():
            return []
        if language == "markdown":
            return self._chunk_markdown(source)
        return self._chunk_plaintext(source)

    # ------------------------------------------------------------------
    # Markdown chunker
    # ------------------------------------------------------------------

    def _chunk_markdown(self, source: str) -> list[ChunkCandidate]:
        """
        Split source at H1-H3 headings, then enforce token budget within each
        section by splitting at paragraph or sentence boundaries.
        """
        lines = source.splitlines(keepends=True)
        total_lines = len(lines)

        # Find 0-based line indices of heading lines.
        heading_indices: list[int] = [
            i for i, line in enumerate(lines) if _HEADING_RE.match(line)
        ]

        if not heading_indices:
            # No headings -- treat as plain text
            return self._chunk_plaintext(source)

        # Build sections: each section = heading line(s) + body until next heading
        section_boundaries: list[int] = heading_indices + [total_lines]

        # Also include everything before the first heading as a preamble section.
        sections: list[tuple[int, int]] = []  # (start_line_0based, end_line_exclusive)
        if heading_indices[0] > 0:
            sections.append((0, heading_indices[0]))
        for i in range(len(section_boundaries) - 1):
            sections.append((section_boundaries[i], section_boundaries[i + 1]))

        raw_candidates: list[ChunkCandidate] = []
        for seg_start, seg_end in sections:
            seg_lines = lines[seg_start:seg_end]
            content = "".join(seg_lines)
            if not content.strip():
                continue
            raw_candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type="paragraph",
                    line_start=seg_start + 1,  # 1-based
                    line_end=seg_end,
                )
            )

        # Enforce token budget, then merge small fragments.
        split_candidates = self._split_large_candidates(raw_candidates)
        merged = self._merge_small_candidates(split_candidates)
        return merged

    # ------------------------------------------------------------------
    # Plain-text chunker
    # ------------------------------------------------------------------

    def _chunk_plaintext(self, source: str) -> list[ChunkCandidate]:
        """
        Split by double-newline paragraphs, merge small, split large.
        """
        lines = source.splitlines(keepends=True)

        # Split into paragraph groups (double newline = blank line between groups)
        paragraphs: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if line.strip() == "":
                if current:
                    paragraphs.append(current)
                    current = []
                # Attach blank separator to the preceding paragraph
                if paragraphs:
                    paragraphs[-1].append(line)
            else:
                current.append(line)
        if current:
            paragraphs.append(current)

        if not paragraphs:
            return []

        raw_candidates: list[ChunkCandidate] = []
        line_cursor = 1  # 1-based running line number
        for para in paragraphs:
            content = "".join(para)
            end_line = line_cursor + len(para) - 1
            if content.strip():
                raw_candidates.append(
                    ChunkCandidate(
                        content=content,
                        chunk_type="paragraph",
                        line_start=line_cursor,
                        line_end=end_line,
                    )
                )
            line_cursor = end_line + 1

        split_candidates = self._split_large_candidates(raw_candidates)
        merged = self._merge_small_candidates(split_candidates)
        return merged

    # ------------------------------------------------------------------
    # Budget enforcement helpers
    # ------------------------------------------------------------------

    def _split_large_candidates(
        self, candidates: list[ChunkCandidate]
    ) -> list[ChunkCandidate]:
        """Expand any candidate exceeding ``max_tokens`` into smaller pieces."""
        result: list[ChunkCandidate] = []
        for cand in candidates:
            if estimate_tokens(cand.content) <= self.max_tokens:
                result.append(cand)
            else:
                result.extend(self._split_candidate(cand))
        return result

    def _split_candidate(self, cand: ChunkCandidate) -> list[ChunkCandidate]:
        """
        Split an oversized candidate first at paragraph breaks, then at
        sentence boundaries.  Falls back to line-midpoint split if necessary.
        """
        # Try paragraph splits first (double newline)
        para_parts = re.split(r"\n\n+", cand.content)
        if len(para_parts) > 1:
            sub_chunks = self._reassemble_parts(para_parts, cand)
            result: list[ChunkCandidate] = []
            for sc in sub_chunks:
                if estimate_tokens(sc.content) <= self.max_tokens:
                    result.append(sc)
                else:
                    result.extend(self._split_by_sentences(sc))
            return result

        # Try sentence splits
        return self._split_by_sentences(cand)

    def _split_by_sentences(self, cand: ChunkCandidate) -> list[ChunkCandidate]:
        """Split a candidate at sentence boundaries."""
        sentences = _SENTENCE_END_RE.split(cand.content)
        if len(sentences) <= 1:
            # Cannot split further -- return as-is, even if over budget
            return [cand]
        return self._reassemble_parts(sentences, cand)

    def _reassemble_parts(
        self,
        parts: list[str],
        original: ChunkCandidate,
    ) -> list[ChunkCandidate]:
        """
        Greedily pack *parts* into new candidates that stay within ``max_tokens``.
        Line numbers are approximate: we track character offsets to compute them.
        """
        candidates: list[ChunkCandidate] = []
        current_parts: list[str] = []
        separator = " "

        # Estimate the starting line by counting prior newlines in original.content
        # We will track offset into content to approximate line_start/line_end.
        content_so_far_chars = 0
        original_lines_before = original.line_start - 1  # 0-based offset

        def _flush(parts_buf: list[str], chars_so_far: int) -> None:
            text = separator.join(parts_buf)
            # Compute approximate line start/end based on newline counts in original
            lines_before = original.content[:chars_so_far].count("\n")
            start = original_lines_before + lines_before + 1  # 1-based
            end = start + text.count("\n")
            candidates.append(
                ChunkCandidate(
                    content=text,
                    chunk_type=original.chunk_type,
                    line_start=start,
                    line_end=end,
                    symbol_name=original.symbol_name,
                    parent_symbol=original.parent_symbol,
                )
            )

        offset = 0
        flush_offset = 0
        for part in parts:
            test_parts = current_parts + [part]
            test_text = separator.join(test_parts)
            if (
                estimate_tokens(test_text) > self.max_tokens
                and current_parts
            ):
                _flush(current_parts, flush_offset)
                flush_offset = offset
                current_parts = [part]
            else:
                current_parts = test_parts
            offset += len(part) + len(separator)

        if current_parts:
            _flush(current_parts, flush_offset)

        return candidates

    def _merge_small_candidates(
        self, candidates: list[ChunkCandidate]
    ) -> list[ChunkCandidate]:
        """
        Merge candidates below ``min_tokens`` into their predecessor.

        A leading small candidate is merged into the following candidate instead.
        """
        if not candidates:
            return []

        result: list[ChunkCandidate] = [candidates[0]]

        for cand in candidates[1:]:
            prev = result[-1]
            combined_tokens = estimate_tokens(prev.content) + estimate_tokens(cand.content)
            if (
                estimate_tokens(cand.content) < self.min_tokens
                and combined_tokens <= self.max_tokens
            ):
                # Merge cand into prev
                joined = prev.content.rstrip("\n") + "\n\n" + cand.content
                result[-1] = ChunkCandidate(
                    content=joined,
                    chunk_type=prev.chunk_type,
                    line_start=prev.line_start,
                    line_end=cand.line_end,
                    symbol_name=prev.symbol_name,
                    parent_symbol=prev.parent_symbol,
                )
            elif (
                estimate_tokens(prev.content) < self.min_tokens
                and combined_tokens <= self.max_tokens
            ):
                # Merge prev (which was just added) into cand
                joined = prev.content.rstrip("\n") + "\n\n" + cand.content
                result[-1] = ChunkCandidate(
                    content=joined,
                    chunk_type=cand.chunk_type,
                    line_start=prev.line_start,
                    line_end=cand.line_end,
                    symbol_name=cand.symbol_name,
                    parent_symbol=cand.parent_symbol,
                )
            else:
                result.append(cand)

        return result
