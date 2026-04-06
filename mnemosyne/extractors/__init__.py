# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document extractor registry and dispatch for Mnemosyne.

Routes non-code files to the appropriate text extractor based on file
extension.  Each extractor is a module with ``extract(file_path)`` and
``supported_extensions()`` functions.  Optional extractors (PDF, OCR)
gracefully degrade when their dependencies are missing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mnemosyne.extractors.base import ExtractedContent

if TYPE_CHECKING:
    from mnemosyne.config import Config

# Import all built-in extractors.  Each module exposes:
#   extract(file_path) -> ExtractedContent | None
#   supported_extensions() -> frozenset[str]
from mnemosyne.extractors import (
    csv_extractor,
    docx_extractor,
    plaintext_extractor,
)

# Build a static extension -> extractor-module map from always-available
# extractors.
_EXTRACTOR_MAP: dict[str, object] = {}

for _mod in (csv_extractor, docx_extractor, plaintext_extractor):
    for _ext in _mod.supported_extensions():
        _EXTRACTOR_MAP[_ext] = _mod


def get_extractor(file_path: str, config: "Config" = None) -> object | None:
    """Return the best available extractor module for *file_path*.

    The returned object has an ``extract(file_path)`` callable.
    Returns ``None`` if no extractor handles this file type.

    Extractor priority for PDF:
      1. pypdf (direct text) -- available with ``mnemosyne-engine[pdf]``
      2. None (no OCR in Tier 0)

    Args:
        file_path: Path to the file to extract.
        config:    Optional Mnemosyne config (reserved for future
                   engine selection in Tier 1+).

    Returns:
        An extractor module or ``None``.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # PDF gets special handling -- optional dep
    if ext == ".pdf":
        from mnemosyne.extractors import pdf_extractor
        if pdf_extractor._available():
            return pdf_extractor
        return None

    return _EXTRACTOR_MAP.get(ext)


def extract_file(file_path: str, config: "Config" = None) -> ExtractedContent | None:
    """Convenience: find the right extractor and run it.

    Args:
        file_path: Absolute path to the file.
        config:    Optional Mnemosyne config.

    Returns:
        :class:`ExtractedContent` or ``None`` if no extractor is available
        or extraction fails.
    """
    extractor = get_extractor(file_path, config)
    if extractor is None:
        return None
    return extractor.extract(file_path)


def supported_extensions() -> frozenset[str]:
    """Return all extensions handled by available extractors."""
    exts = set(_EXTRACTOR_MAP.keys())
    # PDF is always registered as a known extension even if pypdf is
    # not installed -- the extractor will return None at runtime.
    exts.add(".pdf")
    return frozenset(exts)


__all__ = [
    "get_extractor",
    "extract_file",
    "supported_extensions",
    "ExtractedContent",
]
