# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
CSV/TSV text extractor for Mnemosyne.

Converts tabular data into a text representation suitable for chunking
and full-text search.  Each row becomes a line with column headers
prepended for context.  Uses only the Python standard library.

Large files are capped at ``_MAX_ROWS`` to prevent memory exhaustion.
"""

from __future__ import annotations

import csv
import os

from mnemosyne.extractors.base import (
    ExtractedContent,
    ExtractedPage,
    classify_quality,
)

_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv"})

# Cap row count to prevent unbounded memory usage on huge CSVs.
_MAX_ROWS: int = 10_000


def extract(file_path: str) -> ExtractedContent | None:
    """Extract text from a CSV or TSV file.

    Each row is rendered as ``Header1: value1 | Header2: value2 | ...``
    for maximum BM25/TF-IDF searchability.  If the file has no header
    row (detected by csv.Sniffer), columns are labeled ``col_1``, etc.

    Args:
        file_path: Absolute path to the CSV/TSV file.

    Returns:
        :class:`ExtractedContent`.
    """
    _, ext = os.path.splitext(file_path)
    delimiter = "\t" if ext.lower() == ".tsv" else ","

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            # Sniff for header
            sample = fh.read(8192)
            fh.seek(0)

            sniffer = csv.Sniffer()
            try:
                has_header = sniffer.has_header(sample)
            except csv.Error:
                has_header = True  # assume header if sniffing fails

            reader = csv.reader(fh, delimiter=delimiter)

            # Read header
            try:
                first_row = next(reader)
            except StopIteration:
                return ExtractedContent(
                    pages=[],
                    extraction_method="direct",
                    extraction_quality="failed",
                    page_count=0,
                )

            if has_header:
                headers = [h.strip() or f"col_{i+1}" for i, h in enumerate(first_row)]
                data_rows: list[list[str]] = []
            else:
                headers = [f"col_{i+1}" for i in range(len(first_row))]
                data_rows = [first_row]

            # Read data rows
            for row_num, row in enumerate(reader):
                if row_num >= _MAX_ROWS:
                    break
                data_rows.append(row)

    except (OSError, csv.Error):
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
            metadata={"error": "Could not parse CSV/TSV"},
        )

    if not data_rows:
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
        )

    # Render rows as searchable text
    lines: list[str] = []
    lines.append("# Columns: " + " | ".join(headers))
    lines.append("")

    for row in data_rows:
        parts: list[str] = []
        for i, val in enumerate(row):
            header = headers[i] if i < len(headers) else f"col_{i+1}"
            parts.append(f"{header}: {val.strip()}")
        lines.append(" | ".join(parts))

    text = "\n".join(lines)
    quality = classify_quality(100.0, len(text))

    meta: dict[str, str] = {
        "row_count": str(len(data_rows)),
        "column_count": str(len(headers)),
        "columns": ", ".join(headers),
    }

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
