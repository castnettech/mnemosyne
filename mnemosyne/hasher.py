# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Content hashing utilities for Mnemosyne.

Provides normalised SHA-256 digests for both in-memory text and on-disk files,
plus a fast binary-file detector based on null-byte scanning.

Normalisation rules applied before hashing:
  - CRLF -> LF (Windows line endings)
  - Trailing whitespace stripped from each line
  - No trailing newline added / removed (content structure is preserved)

These rules ensure that a file edited on Windows and one on Linux produce the
same hash when their logical content is identical, avoiding spurious re-index
events in cross-platform workflows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Number of bytes read for binary detection.
_BINARY_PROBE_BYTES: int = 8192


def _normalise(text: str) -> str:
    """
    Apply hash-normalisation to *text*.

    Steps:
      1. Unify line endings to LF.
      2. Strip trailing whitespace from every line.

    The resulting string is joined back with LF separators but no final
    newline is appended (we preserve the original presence / absence of a
    trailing newline to avoid false hash matches between files that differ
    only in that regard).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines)


def content_hash(text: str) -> str:
    """
    Return the SHA-256 hex digest of whitespace-normalised *text*.

    Args:
        text: Any string -- source code, prose, configuration, etc.

    Returns:
        64-character lowercase hex string.

    Example::

        >>> content_hash("hello world\\n")
        '...'   # deterministic 64-char hex
    """
    normalised = _normalise(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """
    Compute the SHA-256 hash of the file at *path* using the same normalisation
    as :func:`content_hash`.

    Text is read with ``errors='replace'`` so that files with encoding
    irregularities (e.g. mixed-encoding source) still produce a stable hash
    rather than raising an exception.

    Args:
        path: Filesystem path to the file (str or :class:`pathlib.Path`).

    Returns:
        64-character lowercase hex string.

    Raises:
        OSError: If the file cannot be opened (does not exist, permission denied,
                 etc.).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return content_hash(fh.read())


def file_hash_incremental(path: str | Path) -> str:
    """
    Compute the SHA-256 hash of the file at *path* using incremental I/O.

    Unlike :func:`file_hash`, this variant processes the file in 64 KiB chunks
    so it does not require loading the entire file into memory.  It still
    applies line-ending normalisation.  Suitable for large files.

    Args:
        path: Filesystem path to the file.

    Returns:
        64-character lowercase hex string.
    """
    hasher = hashlib.sha256()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_chunk in iter(lambda: fh.read(65536), ""):
            normalised = _normalise(raw_chunk)
            hasher.update(normalised.encode("utf-8"))
    return hasher.hexdigest()


def is_binary(path: str | Path) -> bool:
    """
    Return True if *path* is likely a binary file.

    Detection strategy: read up to :data:`_BINARY_PROBE_BYTES` bytes in binary
    mode and check for the presence of a null byte (``\\x00``).  This catches
    executables, compiled objects, images, PDFs, and most archive formats while
    correctly identifying UTF-8 and Latin-1 text files.

    False positives are possible for unusual binary formats that happen to avoid
    null bytes in their first 8 KiB; false negatives are unlikely for common
    binary types.

    Args:
        path: Filesystem path to probe.

    Returns:
        True if a null byte is found in the first 8 KiB; False otherwise.

    Raises:
        OSError: If the file cannot be opened.
    """
    with open(path, "rb") as fh:
        probe = fh.read(_BINARY_PROBE_BYTES)
    return b"\x00" in probe


def file_hash_binary(path: str | Path) -> str:
    """
    Compute the SHA-256 hash of a binary file at *path* using raw bytes.

    Unlike :func:`file_hash`, this variant reads the file in binary mode
    without text normalisation.  Suitable for PDFs, images, and other
    non-text files where byte-level identity matters.

    Args:
        path: Filesystem path to the file.

    Returns:
        64-character lowercase hex string.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_document(path: str | Path) -> bool:
    """
    Return True if *path* has a known document extension.

    Document files are binary (PDFs, DOCX) or structured text (CSV, XML)
    that need special extraction rather than direct source-code chunking.

    Args:
        path: Filesystem path or filename to check.

    Returns:
        True if the file extension is a known document type.
    """
    import os
    _, ext = os.path.splitext(str(path))
    return ext.lower() in _DOCUMENT_EXTENSIONS


_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx",
    ".csv", ".tsv",
    ".md", ".txt",
    ".log", ".cfg", ".ini", ".conf",
    ".rst", ".xml", ".svg",
    ".adoc", ".org", ".textile",
})


def bytes_hash(data: bytes) -> str:
    """
    Return the SHA-256 hex digest of raw bytes *data* without normalisation.

    Useful for hashing non-text artefacts (compiled ASTs, serialised embeddings,
    cache state blobs).

    Args:
        data: Raw bytes to hash.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(data).hexdigest()
