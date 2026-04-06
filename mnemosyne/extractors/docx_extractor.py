# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
DOCX text extractor for Mnemosyne.

Extracts text from .docx files using only the Python standard library
(zipfile + xml.etree).  DOCX is a ZIP archive containing XML parts;
the main document body is at ``word/document.xml``.

No external dependencies required.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile

from mnemosyne.extractors.base import (
    ExtractedContent,
    ExtractedPage,
    classify_quality,
)

_EXTENSIONS: frozenset[str] = frozenset({".docx"})

# Word XML namespace
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Match w:t elements (text runs)
_TEXT_TAG = f"{{{_W_NS}}}t"
_PARA_TAG = f"{{{_W_NS}}}p"
_BREAK_TAG = f"{{{_W_NS}}}br"

# Core properties namespace (for metadata)
_CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_DC_NS = "http://purl.org/dc/elements/1.1/"


def extract(file_path: str) -> ExtractedContent | None:
    """Extract text from a DOCX file.

    Args:
        file_path: Absolute path to the .docx file.

    Returns:
        :class:`ExtractedContent`, or ``None`` if the file is not a
        valid ZIP/DOCX archive.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return None

            # Extract main document text
            doc_xml = zf.read("word/document.xml")
            text = _parse_document_xml(doc_xml)

            # Extract metadata
            meta = _parse_metadata(zf)
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
            metadata={"error": "Invalid or corrupt DOCX file"},
        )

    if not text.strip():
        return ExtractedContent(
            pages=[],
            extraction_method="direct",
            extraction_quality="failed",
            page_count=0,
            metadata=meta,
        )

    # DOCX doesn't have reliable page boundaries without rendering,
    # so we return everything as page 0 (non-paginated).
    quality = classify_quality(100.0, len(text))

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


def _parse_document_xml(xml_bytes: bytes) -> str:
    """Parse word/document.xml and return concatenated paragraph text."""
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []

    for para in root.iter(_PARA_TAG):
        runs: list[str] = []
        for elem in para.iter():
            if elem.tag == _TEXT_TAG and elem.text:
                runs.append(elem.text)
            elif elem.tag == _BREAK_TAG:
                runs.append("\n")
        if runs:
            paragraphs.append("".join(runs))

    return "\n\n".join(paragraphs)


def _parse_metadata(zf: zipfile.ZipFile) -> dict[str, str]:
    """Extract core properties (title, author, etc.) from the DOCX."""
    meta: dict[str, str] = {}
    try:
        if "docProps/core.xml" not in zf.namelist():
            return meta
        core_xml = zf.read("docProps/core.xml")
        root = ET.fromstring(core_xml)

        tag_map = {
            f"{{{_DC_NS}}}title": "title",
            f"{{{_DC_NS}}}creator": "author",
            f"{{{_DC_NS}}}subject": "subject",
            f"{{{_DC_NS}}}description": "description",
        }
        for tag, key in tag_map.items():
            elem = root.find(tag)
            if elem is not None and elem.text:
                meta[key] = elem.text.strip()
    except (ET.ParseError, KeyError):
        pass

    return meta


def supported_extensions() -> frozenset[str]:
    return _EXTENSIONS
