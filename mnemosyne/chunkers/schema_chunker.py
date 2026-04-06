# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SQL DDL-aware structural chunker for Mnemosyne.

Extracts CREATE TABLE, CREATE INDEX, CREATE VIEW, and ALTER TABLE statements
from SQL source files.  Each statement becomes a chunk with the table/index/view
name set as ``symbol_name`` and an appropriate ``chunk_type``
(``'table_ddl'``, ``'index_ddl'``, ``'view_ddl'``).

Handles PostgreSQL, MySQL, and SQLite DDL dialects.  Comments (``--`` line
comments and ``/* ... */`` block comments) are preserved in chunk content but
skipped during boundary detection.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.chunkers.code_chunker import ChunkCandidate
from mnemosyne.models import estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# DDL boundary patterns
# ---------------------------------------------------------------------------
# Each pattern captures the object name in a named group ``name``.
# Patterns are case-insensitive and anchored to line start (after optional
# whitespace).  Schema-qualified names (schema.table) capture both parts.

# Identifier pattern: matches unquoted (\w+), double-quoted ("..."),
# backtick-quoted (`...`), or bracket-quoted ([...]) identifiers.
_IDENT = r'(?P<name>"[^"]+"|`[^`]+`|\[[^\]]+\]|\w+)'
_IDENT_SCHEMA = r'(?:(?P<schema>\w+)\.)?' + _IDENT

_DDL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # CREATE [OR REPLACE] [TEMP|TEMPORARY] TABLE [IF NOT EXISTS] [schema.]name
    (
        re.compile(
            r"^[ \t]*CREATE\s+"
            r"(?:OR\s+REPLACE\s+)?"
            r"(?:(?:TEMP|TEMPORARY|UNLOGGED|FOREIGN)\s+)?"
            r"TABLE\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            + _IDENT_SCHEMA,
            re.M | re.I,
        ),
        "table_ddl",
    ),
    # CREATE [UNIQUE] INDEX [CONCURRENTLY] [IF NOT EXISTS] name ON table
    (
        re.compile(
            r"^[ \t]*CREATE\s+"
            r"(?:UNIQUE\s+)?"
            r"INDEX\s+"
            r"(?:CONCURRENTLY\s+)?"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            + _IDENT,
            re.M | re.I,
        ),
        "index_ddl",
    ),
    # CREATE [OR REPLACE] [MATERIALIZED] VIEW [IF NOT EXISTS] [schema.]name
    (
        re.compile(
            r"^[ \t]*CREATE\s+"
            r"(?:OR\s+REPLACE\s+)?"
            r"(?:MATERIALIZED\s+)?"
            r"VIEW\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            + _IDENT_SCHEMA,
            re.M | re.I,
        ),
        "view_ddl",
    ),
    # ALTER TABLE [IF EXISTS] [schema.]name
    (
        re.compile(
            r"^[ \t]*ALTER\s+TABLE\s+"
            r"(?:IF\s+EXISTS\s+)?"
            r"(?:ONLY\s+)?"
            + _IDENT_SCHEMA,
            re.M | re.I,
        ),
        "table_ddl",
    ),
    # CREATE [OR REPLACE] FUNCTION/PROCEDURE [schema.]name
    (
        re.compile(
            r"^[ \t]*CREATE\s+"
            r"(?:OR\s+REPLACE\s+)?"
            r"(?:FUNCTION|PROCEDURE)\s+"
            + _IDENT_SCHEMA,
            re.M | re.I,
        ),
        "function",
    ),
    # CREATE TYPE [schema.]name
    (
        re.compile(
            r"^[ \t]*CREATE\s+TYPE\s+"
            + _IDENT_SCHEMA,
            re.M | re.I,
        ),
        "table_ddl",
    ),
    # CREATE TRIGGER name
    (
        re.compile(
            r"^[ \t]*CREATE\s+"
            r"(?:OR\s+REPLACE\s+)?"
            r"(?:CONSTRAINT\s+)?"
            r"TRIGGER\s+"
            + _IDENT,
            re.M | re.I,
        ),
        "table_ddl",
    ),
]


# ---------------------------------------------------------------------------
# Statement-end detection
# ---------------------------------------------------------------------------

def _find_statement_end(source: str, start: int) -> int:
    """Find the end of a SQL statement starting at *start*.

    Scans forward looking for a semicolon that is not inside a string literal
    or comment.  For statements containing parenthesized blocks (CREATE TABLE
    with column definitions), tracks paren depth so that semicolons inside
    default values or check constraints are not misidentified as terminators.

    Also handles dollar-quoted strings (PostgreSQL ``$tag$...$tag$`` or
    ``$$...$$``) and the ``BEGIN...END`` blocks used in function/trigger bodies.

    Returns the inclusive index of the semicolon, or ``len(source) - 1`` if
    no terminator is found (last statement in file, common for dumps without
    trailing semicolons).
    """
    i = start
    n = len(source)
    paren_depth = 0
    begin_depth = 0

    while i < n:
        ch = source[i]

        # -- line comment ---------------------------------------------------
        if ch == "-" and i + 1 < n and source[i + 1] == "-":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue

        # -- block comment --------------------------------------------------
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                i = n
            continue

        # -- single-quoted string literal -----------------------------------
        if ch == "'":
            i += 1
            while i < n:
                if source[i] == "'" and i + 1 < n and source[i + 1] == "'":
                    i += 2  # escaped quote ''
                    continue
                if source[i] == "'":
                    i += 1
                    break
                i += 1
            continue

        # -- dollar-quoted string (PostgreSQL) ------------------------------
        if ch == "$":
            tag_match = re.match(r"\$(\w*)\$", source[i:])
            if tag_match:
                tag = tag_match.group(0)
                end_tag = source.find(tag, i + len(tag))
                if end_tag != -1:
                    i = end_tag + len(tag)
                else:
                    i = n
                continue

        # -- double-quoted identifier ---------------------------------------
        if ch == '"':
            i += 1
            while i < n and source[i] != '"':
                i += 1
            if i < n:
                i += 1
            continue

        # -- paren tracking -------------------------------------------------
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)

        # -- BEGIN/END tracking (functions, triggers) -----------------------
        if paren_depth == 0:
            upper_slice = source[i:i + 6].upper()
            if upper_slice.startswith("BEGIN") and (
                i + 5 >= n or not source[i + 5].isalnum()
            ):
                begin_depth += 1
                i += 5
                continue
            if upper_slice[:3] == "END" and (
                i + 3 >= n or not source[i + 3].isalnum()
            ):
                begin_depth = max(0, begin_depth - 1)
                i += 3
                continue

        # -- semicolon (statement end) --------------------------------------
        if ch == ";" and paren_depth == 0 and begin_depth == 0:
            return i

        i += 1

    return n - 1


# ---------------------------------------------------------------------------
# SchemaChunker
# ---------------------------------------------------------------------------


class SchemaChunker:
    """
    SQL DDL-aware structural chunker.

    Extracts CREATE TABLE, CREATE INDEX, CREATE VIEW, ALTER TABLE, and other
    DDL statements from SQL source.  Each statement becomes a
    :class:`~mnemosyne.chunkers.code_chunker.ChunkCandidate` with the object
    name as ``symbol_name`` and a DDL-specific ``chunk_type``.

    Text between DDL statements (comments, SET commands, transaction markers)
    is emitted as ``block`` chunks.

    Oversized statements are split with a sliding window, preserving
    ``symbol_name`` on the first sub-chunk.

    Args:
        config: Mnemosyne :class:`~mnemosyne.config.Config` instance.
    """

    def __init__(self, config: "Config") -> None:
        self.max_tokens: int = config.chunking.max_chunk_tokens
        self.min_tokens: int = config.chunking.min_chunk_tokens
        self.overlap: int = config.chunking.overlap_lines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, language: str = "sql") -> list[ChunkCandidate]:
        """
        Split *source* into :class:`ChunkCandidate` objects.

        Args:
            source:   Full SQL source text.
            language: Language tag (informational).

        Returns:
            Ordered list of chunk candidates.  Never empty for non-empty source.
        """
        if not source.strip():
            return []

        lines = source.splitlines(keepends=True)
        boundaries = self._find_boundaries(source)

        if not boundaries:
            # No DDL statements found -- emit entire content as a single block.
            return [
                ChunkCandidate(
                    content=source,
                    chunk_type="block",
                    line_start=1,
                    line_end=len(lines),
                )
            ]

        candidates: list[ChunkCandidate] = []
        consumed_char = 0

        for match_start, chunk_type, symbol_name in boundaries:
            # Emit interstitial text before this statement.
            if match_start > consumed_char:
                interstitial = source[consumed_char:match_start]
                if interstitial.strip():
                    seg_start = source[:consumed_char].count("\n") + 1
                    seg_end = source[:match_start].count("\n")
                    seg_end = max(seg_start, seg_end)
                    candidates.append(
                        ChunkCandidate(
                            content=interstitial,
                            chunk_type="block",
                            line_start=seg_start,
                            line_end=seg_end,
                        )
                    )

            # Find end of this DDL statement.
            stmt_end = _find_statement_end(source, match_start)

            # Include trailing newline if present.
            if stmt_end + 1 < len(source) and source[stmt_end + 1] == "\n":
                stmt_end += 1

            stmt_text = source[match_start:stmt_end + 1]
            line_start = source[:match_start].count("\n") + 1
            line_end = source[:stmt_end + 1].count("\n")
            line_end = max(line_start, line_end)

            # Strip surrounding quotes from symbol name.
            clean_name = symbol_name.strip('"').strip('`').strip('[').strip(']')

            candidates.append(
                ChunkCandidate(
                    content=stmt_text,
                    chunk_type=chunk_type,
                    line_start=line_start,
                    line_end=line_end,
                    symbol_name=clean_name,
                )
            )
            consumed_char = stmt_end + 1

        # Emit trailing text after last statement.
        if consumed_char < len(source):
            tail = source[consumed_char:]
            if tail.strip():
                tail_start = source[:consumed_char].count("\n") + 1
                tail_end = tail_start + tail.count("\n") - 1
                tail_end = max(tail_start, tail_end)
                candidates.append(
                    ChunkCandidate(
                        content=tail,
                        chunk_type="block",
                        line_start=tail_start,
                        line_end=tail_end,
                    )
                )

        # Split oversized chunks.
        result: list[ChunkCandidate] = []
        for cand in candidates:
            result.extend(self._maybe_split(cand))

        return result

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    def _find_boundaries(
        self, source: str
    ) -> list[tuple[int, str, str]]:
        """Find all DDL statement boundaries in *source*.

        Returns a sorted list of ``(char_offset, chunk_type, symbol_name)``
        tuples, deduplicated by character offset (first match wins).
        """
        raw: list[tuple[int, str, str]] = []

        for pattern, chunk_type in _DDL_PATTERNS:
            for m in pattern.finditer(source):
                name = m.group("name")
                if name:
                    raw.append((m.start(), chunk_type, name))

        # Sort by position, deduplicate by offset.
        raw.sort(key=lambda x: x[0])
        seen: set[int] = set()
        unique: list[tuple[int, str, str]] = []
        for item in raw:
            if item[0] not in seen:
                seen.add(item[0])
                unique.append(item)

        return unique

    # ------------------------------------------------------------------
    # Sliding-window split for oversized chunks
    # ------------------------------------------------------------------

    def _maybe_split(self, cand: ChunkCandidate) -> list[ChunkCandidate]:
        """Split *cand* with a sliding window if it exceeds max_tokens."""
        if estimate_tokens(cand.content) <= self.max_tokens:
            return [cand]

        sub_lines = cand.content.splitlines(keepends=True)
        result: list[ChunkCandidate] = []
        n = len(sub_lines)
        pos = 0
        is_first = True

        while pos < n:
            window: list[str] = []
            end = pos

            while end < n:
                window.append(sub_lines[end])
                end += 1
                if estimate_tokens("".join(window)) >= self.max_tokens:
                    break

            content = "".join(window)
            abs_start = cand.line_start + pos
            abs_end = cand.line_start + end - 1

            result.append(
                ChunkCandidate(
                    content=content,
                    chunk_type=cand.chunk_type,
                    line_start=abs_start,
                    line_end=abs_end,
                    symbol_name=cand.symbol_name if is_first else None,
                )
            )
            is_first = False

            if end >= n:
                break
            advance = max(1, len(window) - self.overlap)
            pos += advance

        # Merge tiny trailing fragment.
        if len(result) >= 2:
            last = result[-1]
            if estimate_tokens(last.content) < self.min_tokens:
                prev = result[-2]
                combined = prev.content + last.content
                if estimate_tokens(combined) <= self.max_tokens:
                    result[-2] = ChunkCandidate(
                        content=combined,
                        chunk_type=prev.chunk_type,
                        line_start=prev.line_start,
                        line_end=last.line_end,
                        symbol_name=prev.symbol_name,
                    )
                    result.pop()

        return result or [cand]
