# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document chunker for Mnemosyne.

Chunks extracted document text (from PDFs, DOCX, CSV, etc.) into
:class:`ChunkCandidate` objects.  Delegates to the existing
:class:`TextChunker` for heading/paragraph-aware splitting, adding
page-number tracking on top.

The document chunker is designed to work with :class:`ExtractedContent`
from the extractors package.  Each page is chunked independently (to
preserve page boundaries), and chunks carry ``page_number`` metadata
via a new ``extra`` dict on :class:`ChunkCandidate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemosyne.chunkers.code_chunker import ChunkCandidate
from mnemosyne.chunkers.text_chunker import TextChunker

if TYPE_CHECKING:
    from mnemosyne.config import Config
    from mnemosyne.extractors.base import ExtractedContent


class DocumentChunker:
    """Chunker for extracted document content.

    Wraps :class:`TextChunker` and adds page-number tracking.
    Multi-page documents are chunked per-page so that retrieval
    results can reference the source page.

    Args:
        config: Mnemosyne :class:`~mnemosyne.config.Config` instance.
    """

    def __init__(self, config: "Config") -> None:
        self._text_chunker = TextChunker(config)

    def chunk(self, source: str, language: str = "document") -> list[ChunkCandidate]:
        """Chunk a plain text string (for compatibility with chunker protocol).

        When called directly with a string, behaves like TextChunker
        with ``"text"`` language and page_number=0 on all chunks.

        Args:
            source:   Full text to chunk.
            language: Ignored (always treated as prose).

        Returns:
            List of :class:`ChunkCandidate` objects.
        """
        candidates = self._text_chunker.chunk(source, "text")
        for cand in candidates:
            cand.page_number = 0
        return candidates

    def chunk_extracted(self, content: "ExtractedContent") -> list[ChunkCandidate]:
        """Chunk an :class:`ExtractedContent` with page-number tracking.

        Each page is chunked independently.  Chunks from later pages
        have their ``line_start`` / ``line_end`` offset to account for
        preceding pages, so they remain globally unique within the file.

        Args:
            content: Extracted document content from an extractor.

        Returns:
            Ordered list of :class:`ChunkCandidate` objects with
            ``page_number`` set.
        """
        all_candidates: list[ChunkCandidate] = []
        line_offset = 0

        for page in content.pages:
            if not page.text.strip():
                # Count lines in empty pages for offset tracking
                line_offset += page.text.count("\n") + 1
                continue

            # Chunk this page's text
            page_candidates = self._text_chunker.chunk(page.text, "text")

            # Apply page number and line offset
            for cand in page_candidates:
                cand.page_number = page.page_number
                cand.line_start += line_offset
                cand.line_end += line_offset

            all_candidates.extend(page_candidates)
            line_offset += page.text.count("\n") + 1

        return all_candidates
