# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Base types and protocol for document extractors.

Every extractor converts a binary or non-code file into
:class:`ExtractedContent` -- a sequence of text pages with confidence
metadata.  The ingestion pipeline then feeds these pages into the
document chunker for indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ExtractedPage:
    """One page (or logical section) of extracted text.

    Attributes:
        page_number:  1-based page index.  Use 0 for non-paginated
                      formats (CSV, plaintext, DOCX without page breaks).
        text:         Extracted text content.
        confidence:   Extraction confidence 0-100.  Direct digital
                      extraction = 100.0; OCR results vary.
        method:       How the text was obtained.  One of ``'direct'``,
                      ``'ocr_tesseract'``, ``'ocr_doctr'``, ``'metadata'``.
    """

    page_number: int
    text: str
    confidence: float = 100.0
    method: str = "direct"


@dataclass
class ExtractedContent:
    """Result of extracting text from a document file.

    Attributes:
        pages:              Ordered list of extracted pages.
        extraction_method:  Best method used across all pages.
        extraction_quality: Overall quality verdict: ``'good'``,
                            ``'poor'``, or ``'failed'``.
        page_count:         Total number of pages in the source document
                            (may differ from ``len(pages)`` if some pages
                            were empty or skipped).
        metadata:           Optional key-value pairs (title, author,
                            created date, etc.).
    """

    pages: list[ExtractedPage]
    extraction_method: str = "direct"
    extraction_quality: str = "good"
    page_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all page texts with double-newline separators."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def mean_confidence(self) -> float:
        """Average extraction confidence across all pages."""
        if not self.pages:
            return 0.0
        return sum(p.confidence for p in self.pages) / len(self.pages)


def classify_quality(confidence: float, text_length: int) -> str:
    """Classify extraction quality from confidence and text length.

    Thresholds derived from production OCR pipeline testing:
    - < 50 confidence -> ``'failed'``
    - 50-65 confidence -> ``'poor'``
    - >= 65 confidence AND >= 200 chars -> ``'good'``
    - >= 65 confidence AND < 200 chars -> ``'poor'``
    """
    if confidence < 50.0:
        return "failed"
    if confidence < 65.0:
        return "poor"
    if text_length < 200:
        return "poor"
    return "good"


@runtime_checkable
class BaseExtractor(Protocol):
    """Protocol that all extractors implement."""

    def extract(self, file_path: str) -> ExtractedContent | None:
        """Extract text content from *file_path*.

        Returns:
            :class:`ExtractedContent` on success, or ``None`` if this
            extractor cannot handle the file.
        """
        ...

    def supported_extensions(self) -> frozenset[str]:
        """Return the set of file extensions this extractor handles."""
        ...
