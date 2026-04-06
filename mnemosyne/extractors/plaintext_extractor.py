# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Plaintext extractor for Mnemosyne.

Handles non-code text files that are not Markdown -- config files,
logs, READMEs without .md extension, etc.  This is a thin wrapper
that reads the file as UTF-8 and packages it as ExtractedContent.

No external dependencies required.
"""

from __future__ import annotations

import os

from mnemosyne.extractors.base import (
    ExtractedContent,
    ExtractedPage,
    classify_quality,
)

_EXTENSIONS: frozenset[str] = frozenset({
    ".log", ".cfg", ".ini", ".conf", ".env.example",
    ".rst", ".textile", ".adoc", ".org",
    ".xml", ".svg",
})


def extract(file_path: str) -> ExtractedContent | None:
    """Read a plaintext file and return it as ExtractedContent.

    Args:
        file_path: Absolute path to the file.

    Returns:
        :class:`ExtractedContent`.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
            metadata={"error": "Could not read file"},
        )

    if not text.strip():
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
        )

    quality = classify_quality(100.0, len(text))

    _, ext = os.path.splitext(file_path)
    meta: dict[str, str] = {"format": ext.lstrip(".")}

    return ExtractedContent(
        pages=[ExtractedPage(
            page_number=0,
            text=text,
            confidence=100.0,
            method="direct",
        )],
        extraction_method="direct",
        extraction_quality=quality,
        page_count=1,
        metadata=meta,
    )


def supported_extensions() -> frozenset[str]:
    return _EXTENSIONS
