# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PDF text extractor for Mnemosyne.

Extracts text from digital (text-layer) PDFs using pypdf.  Does NOT
perform OCR -- scanned PDFs will yield empty or minimal text and get
a ``'poor'`` or ``'failed'`` quality rating, signalling that an OCR
extractor (Tier 1+) should be installed.

pypdf is an optional dependency: ``pip install mnemosyne-engine[pdf]``
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.extractors.base import (
    ExtractedContent,
    ExtractedPage,
    classify_quality,
)

if TYPE_CHECKING:
    pass

_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

# Minimum chars per page before we consider the extraction useful.
# From HCCValidator lesson: ~30% of "scanned" PDFs actually have
# embedded text layers.  100 chars is the threshold where direct
# extraction is worth keeping.
_MIN_PAGE_CHARS: int = 100

# Collapse 3+ consecutive newlines to 2 (paragraph break)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _available() -> bool:
    """Return True if pypdf is importable."""
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def extract(file_path: str) -> ExtractedContent | None:
    """Extract text from a PDF using pypdf.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        :class:`ExtractedContent` or ``None`` if pypdf is not installed.
    """
    if not _available():
        return None

    import pypdf

    try:
        reader = pypdf.PdfReader(file_path)
    except Exception:
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
            metadata={"error": "Could not open PDF"},
        )

    page_count = len(reader.pages)
    pages: list[ExtractedPage] = []
    total_chars = 0

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        # Normalize whitespace
        text = _MULTI_NEWLINE_RE.sub("\n\n", text).strip()
        char_count = len(text)
        total_chars += char_count

        # Confidence: 100 for substantial text, scaled down for thin pages
        if char_count >= _MIN_PAGE_CHARS:
            confidence = 100.0
        elif char_count > 0:
            confidence = min(100.0, (char_count / _MIN_PAGE_CHARS) * 65.0)
        else:
            confidence = 0.0

        pages.append(ExtractedPage(
            page_number=i + 1,
            text=text,
            confidence=confidence,
            method="direct",
        ))

    # Collect metadata
    meta: dict[str, str] = {}
    if reader.metadata:
        for key in ("title", "author", "subject", "creator"):
            val = getattr(reader.metadata, key, None)
            if val:
                meta[key] = str(val)
    meta["page_count"] = str(page_count)

    # Overall quality
    mean_conf = sum(p.confidence for p in pages) / max(len(pages), 1)
    quality = classify_quality(mean_conf, total_chars)

    return ExtractedContent(
        pages=pages,
        extraction_method="direct",
        extraction_quality=quality,
        page_count=page_count,
        metadata=meta,
    )


def supported_extensions() -> frozenset[str]:
    return _EXTENSIONS
