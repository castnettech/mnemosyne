# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Domain model dataclasses for Mnemosyne.

All objects are pure-Python dataclasses -- no ORM, no external deps.
The ``estimate_tokens`` utility provides a lightweight word-count proxy for
token estimation, sufficient for budget gating before a real tokeniser is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


@dataclass
class FileRecord:
    """
    Represents a file tracked in the index.

    Attributes:
        file_id:       Primary key assigned by the database (None before insert).
        rel_path:      Path relative to the project root (forward slashes).
        content_hash:  SHA-256 hex digest of the normalised file content.
        size_bytes:    Raw byte size of the file on disk.
        language:      Detected language tag (e.g. ``"python"``, ``"markdown"``),
                       or None if unknown.
        last_modified: ``os.path.getmtime()`` float timestamp.
        last_indexed:  ISO-8601 UTC timestamp of the most recent index run,
                       or None if never indexed.
        is_deleted:    True when the file has been removed from disk but its
                       record is retained for delta / history purposes.
    """

    file_id: int | None
    rel_path: str
    content_hash: str
    size_bytes: int
    language: str | None
    last_modified: float
    last_indexed: str | None = None
    is_deleted: bool = False


@dataclass
class Chunk:
    """
    A contiguous, semantically meaningful slice of a file.

    Attributes:
        chunk_id:          Database primary key (None before insert).
        file_id:           FK -> FileRecord.file_id.
        content_hash:      SHA-256 of the chunk's normalised content.
        chunk_type:        Structural category of the chunk content.
                           One of: ``'function'``, ``'class'``, ``'paragraph'``,
                           ``'block'``, ``'imports'``, ``'generic'``.
        line_start:        1-based inclusive start line in the source file.
        line_end:          1-based inclusive end line in the source file.
        token_count:       Approximate token count (from ``estimate_tokens``).
        content:           Raw source text of this chunk.
        compressed:        Compressed representation (may be None if not yet
                           compressed or if compression was skipped).
        compression_ratio: ``len(compressed) / len(content)`` approximation,
                           or None when ``compressed`` is None.
        symbol_name:       For ``function`` / ``class`` chunks, the qualified name
                           of the symbol (e.g. ``"MyClass.my_method"``).
        parent_chunk_id:   For nested chunks (e.g. a method inside a class body),
                           the chunk_id of the enclosing chunk.
    """

    chunk_id: int | None
    file_id: int
    content_hash: str
    chunk_type: str  # 'function' | 'class' | 'paragraph' | 'block' | 'imports' | 'generic'
    line_start: int
    line_end: int
    token_count: int
    content: str
    compressed: str | None = None
    compression_ratio: float | None = None
    symbol_name: str | None = None
    parent_chunk_id: int | None = None


@dataclass
class Summary:
    """
    A human/LLM-readable summary at a configurable scope level.

    Attributes:
        summary_id: Database primary key.
        scope_type: Granularity of what is summarised.
                    One of: ``'chunk'``, ``'file'``, ``'directory'``, ``'project'``.
        scope_path: The path (or chunk hash) that identifies the summarised unit.
        content:    The summary text itself.
        token_count: Approximate token count of *content*.
        parent_id:  FK -> Summary.summary_id of the enclosing scope, or None for
                    top-level summaries.
        version:    Monotonically increasing integer; incremented on each re-summary.
    """

    summary_id: int | None
    scope_type: str  # 'chunk' | 'file' | 'directory' | 'project'
    scope_path: str
    content: str
    token_count: int
    parent_id: int | None = None
    version: int = 1


@dataclass
class QueryResult:
    """
    A single ranked result returned from the retrieval pipeline.

    Attributes:
        chunk:      The matched :class:`Chunk` object.
        file_path:  Relative path of the file that owns the chunk.
        scores:     Per-signal and composite scores::

                        {
                            "bm25":   float,  # BM25 full-text score (normalised 0-1)
                            "vector": float,  # Cosine / TF-IDF similarity (0-1)
                            "usage":  float,  # Decayed usage frequency score (0-1)
                            "rrf":    float,  # Reciprocal Rank Fusion composite
                        }

        is_delta:   True when the result represents a *changed* chunk and the
                    delta text is provided rather than full content.
        delta_text: Unified diff or summarised change text (only when
                    ``is_delta`` is True).
        is_stale:   True when the underlying file on disk has been modified
                    or deleted since the last index run.
        stale_reason: Human-readable explanation when ``is_stale`` is True
                      (e.g. ``"file modified since last index"`` or
                      ``"file no longer exists on disk"``).
    """

    chunk: Chunk
    file_path: str
    scores: dict[str, float]
    is_delta: bool = False
    delta_text: str | None = None
    is_stale: bool = False
    stale_reason: str | None = None


@dataclass
class CacheEntry:
    """
    Represents a chunk's position within the ARC (Adaptive Replacement Cache).

    Attributes:
        chunk_id:      FK -> Chunk.chunk_id.
        tier:          ARC tier name.
                       One of: ``'T1'`` (recent, not recurrent),
                       ``'T2'`` (recent and recurrent),
                       ``'B1'`` (ghost -- evicted from T1),
                       ``'B2'`` (ghost -- evicted from T2).
        access_count:  Total number of cache hits for this entry in the current
                       session.
        last_accessed: ISO-8601 UTC timestamp of the most recent access.
    """

    chunk_id: int
    tier: str  # 'T1' | 'T2' | 'B1' | 'B2'
    access_count: int
    last_accessed: str


@dataclass
class UsageEvent:
    """
    Records a single interaction between a query and a chunk.

    Attributes:
        event_id:   Database primary key (None before insert).
        chunk_id:   FK -> Chunk.chunk_id.
        query_text: The raw query string that surfaced this chunk, or None if
                    the event was triggered by a non-query action.
        session_id: Opaque session identifier for grouping related events.
        event_type: What happened to the chunk in this interaction.
                    One of: ``'retrieved'`` (included in results),
                    ``'selected'`` (user/agent clicked / cited),
                    ``'used'`` (chunk was injected into a prompt),
                    ``'discarded'`` (present in results but not used).
        timestamp:  ISO-8601 UTC timestamp; auto-set by the store if None.
    """

    event_id: int | None
    chunk_id: int
    query_text: str | None
    session_id: str | None
    event_type: str  # 'retrieved' | 'selected' | 'used' | 'discarded'
    timestamp: str | None = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """
    Approximate the token count of *text* using a word-split heuristic.

    This is intentionally cheap -- it is used for budget gating and chunk
    sizing, not for precise billing or model-specific tokenisation.

    Rule: ``max(1, len(text.split()))`` (whitespace-delimited words as a proxy
    for subword tokens).  Punctuation-heavy code will be under-counted slightly,
    which is acceptable for the use-case here.

    Args:
        text: Any string to estimate.

    Returns:
        An integer >= 1.
    """
    if not text:
        return 1
    return max(1, len(text.split()))
