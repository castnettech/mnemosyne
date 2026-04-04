# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared brace-depth scanner and base chunker for brace-delimited languages.

Provides :class:`BraceDepthScanner` for finding matching closing braces while
skipping strings, comments, and other non-code constructs.  :class:`BraceChunker`
is the base class for all ``{...}``-delimited language chunkers (Go, C#, Rust,
Java, etc.).  Language-specific subclasses provide :class:`DeclPattern` lists
that describe the regex signatures of each declaration kind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mnemosyne.chunkers.code_chunker import ChunkCandidate
from mnemosyne.models import estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Declaration pattern descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeclPattern:
    """A regex pattern that matches a declaration head.

    Attributes:
        pattern:    Compiled regex with at least one named group for the symbol.
        chunk_type: Structural category (``'function'``, ``'class'``, ``'block'``,
                    or ``'imports'``).
        name_group: Name of the regex group that captures the symbol name.
        has_block:  True if the declaration is followed by a ``{...}`` block.
                    False for statement-level patterns (``using``, ``import``,
                    ``package``) that end at ``;`` or ``)``.  Defaults to True.
    """

    pattern: re.Pattern[str]
    chunk_type: str
    name_group: str
    has_block: bool = True


# ---------------------------------------------------------------------------
# Brace depth scanner
# ---------------------------------------------------------------------------


class BraceDepthScanner:
    """Find matching closing brace from a given position, skipping strings
    and comments.
    """

    @staticmethod
    def find_block_end(source: str, open_brace_pos: int) -> int:
        """Scan from *open_brace_pos* (must point to ``{``) forward,
        tracking brace depth.

        Skips:
        - Single-quoted strings (``'...'`` with ``\\'`` escape)
        - Double-quoted strings (``"..."`` with ``\\"`` escape)
        - Backtick strings (Go raw strings / JS template literals)
        - Line comments  (``// ...``)
        - Block comments (``/* ... */``)

        Returns:
            Index of the matching ``}``, or ``len(source)`` if unmatched.
        """
        depth = 0
        i = open_brace_pos
        n = len(source)

        while i < n:
            ch = source[i]

            # -- line comment -------------------------------------------------
            if ch == "/" and i + 1 < n and source[i + 1] == "/":
                i += 2
                while i < n and source[i] != "\n":
                    i += 1
                continue

            # -- block comment ------------------------------------------------
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

            # -- string / char literals ---------------------------------------
            if ch in ('"', "'", "`"):
                quote = ch
                i += 1
                while i < n:
                    c = source[i]
                    if c == "\\" and quote != "`" and i + 1 < n:
                        i += 2  # skip escaped char (not in backtick strings)
                        continue
                    if c == quote:
                        i += 1
                        break
                    i += 1
                continue

            # -- brace counting -----------------------------------------------
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i

            i += 1

        return n


# ---------------------------------------------------------------------------
# Base chunker for brace-delimited languages
# ---------------------------------------------------------------------------


class BraceChunker:
    """Base chunker for languages with ``{...}``-delimited blocks.

    Subclasses instantiate this with a list of :class:`DeclPattern` objects
    that describe the regex signatures of declarations in the target language.

    The chunking algorithm:
    1. Find all declaration matches via ``self.patterns``.
    2. For each match locate the opening ``{`` after the match end.
    3. Use :class:`BraceDepthScanner` to find the closing ``}``.
    4. Extract the full declaration as a :class:`ChunkCandidate`.
    5. Code between declarations becomes ``block`` chunks.
    6. Oversized chunks are split via a sliding window.
    """

    def __init__(self, config: "Config", patterns: list[DeclPattern]) -> None:
        self.max_tokens: int = config.chunking.max_chunk_tokens
        self.min_tokens: int = config.chunking.min_chunk_tokens
        self.overlap: int = config.chunking.overlap_lines
        self.patterns = patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, language: str) -> list[ChunkCandidate]:
        """Split *source* into :class:`ChunkCandidate` objects.

        Returns:
            Ordered list of chunk candidates.  Never empty for non-empty input.
        """
        if not source.strip():
            return []

        lines = source.splitlines(keepends=True)
        decls = self._find_declarations(source)

        if not decls:
            return self._sliding_window(lines, symbol_name=None, chunk_type="block")

        candidates = self._build_candidates(source, lines, decls)

        # Post-process: split oversized chunks.
        result: list[ChunkCandidate] = []
        for cand in candidates:
            result.extend(self._maybe_split(cand))

        return result or [
            ChunkCandidate(
                content=source,
                chunk_type="block",
                line_start=1,
                line_end=len(lines),
            )
        ]

    # ------------------------------------------------------------------
    # Declaration discovery
    # ------------------------------------------------------------------

    def _find_declarations(
        self, source: str
    ) -> list[tuple[int, int, str, str]]:
        """Return sorted list of ``(match_start, block_end, chunk_type,
        symbol_name)`` tuples for every declaration found in *source*.

        Overlapping declarations are deduplicated by start position (first
        pattern match wins).
        """
        raw: list[tuple[int, str, str, bool]] = []  # (start, type, name, has_block)

        for dp in self.patterns:
            for m in dp.pattern.finditer(source):
                name = m.group(dp.name_group)
                raw.append((m.start(), dp.chunk_type, name, dp.has_block))

        raw.sort(key=lambda t: t[0])

        # Deduplicate overlapping starts.
        seen: set[int] = set()
        unique: list[tuple[int, str, str, bool]] = []
        for start, ctype, name, has_blk in raw:
            if start not in seen:
                seen.add(start)
                unique.append((start, ctype, name, has_blk))

        # Resolve extents.
        results: list[tuple[int, int, str, str]] = []
        for start, ctype, name, has_blk in unique:
            if not has_blk:
                # Statement-level pattern: extent is to end-of-line or
                # to the closing ')' for grouped patterns like Go's
                # import (...) / const (...).
                end = self._find_statement_extent(source, start)
                results.append((start, end, ctype, name))
                continue

            # Block-level pattern: find opening '{' on the same declaration.
            # Only search up to the next newline-newline gap or another
            # declaration's start to avoid grabbing a brace from a later decl.
            match_end = source.find("\n", start)
            if match_end == -1:
                match_end = len(source)
            # Extend through continuation lines (no blank-line gap) looking
            # for the opening brace -- allows multi-line signatures.
            search_end = match_end
            while search_end < len(source):
                nl = source.find("\n", search_end + 1)
                if nl == -1:
                    search_end = len(source)
                    break
                # Stop if we hit a blank line (two consecutive newlines).
                if nl == search_end + 1:
                    break
                search_end = nl

            brace_pos = source.find("{", start, search_end + 1)
            if brace_pos == -1:
                # No block body found in the declaration region.
                end = match_end
                results.append((start, end, ctype, name))
                continue

            end = BraceDepthScanner.find_block_end(source, brace_pos)
            # Include trailing newline if present.
            if end + 1 < len(source) and source[end + 1] == "\n":
                end += 1
            results.append((start, end, ctype, name))

        return results

    @staticmethod
    def _find_statement_extent(source: str, start: int) -> int:
        """Find the end of a statement-level declaration starting at *start*.

        Handles two cases:
        1. Grouped declarations ending with ``)``: ``import (...)``,
           ``const (...)``, ``var (...)``.
        2. Simple semicolon-terminated statements: ``using X;``,
           ``import X;``, ``package X;``.
        3. Simple newline-terminated statements (no semicolon, no group).

        Returns the inclusive end index.
        """
        n = len(source)
        # Check if there is a '(' before the next newline.
        nl = source.find("\n", start)
        if nl == -1:
            nl = n
        paren_pos = source.find("(", start, nl)
        if paren_pos != -1:
            # Paren-grouped block: find matching ')'.
            depth = 0
            i = paren_pos
            while i < n:
                ch = source[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        # Include trailing newline.
                        if i + 1 < n and source[i + 1] == "\n":
                            return i + 1
                        return i
                i += 1
            return n - 1

        # Check for semicolon on first line.
        semi = source.find(";", start, nl)
        if semi != -1:
            # Include trailing newline.
            if semi + 1 < n and source[semi + 1] == "\n":
                return semi + 1
            return semi

        # Fallback: end at newline.
        return nl if nl < n else n - 1

    # ------------------------------------------------------------------
    # Candidate assembly
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        source: str,
        lines: list[str],
        decls: list[tuple[int, int, str, str]],
    ) -> list[ChunkCandidate]:
        """Build ChunkCandidate objects for each declaration plus
        interstitial gap blocks.

        When a declaration contains child declarations (e.g. a namespace
        containing classes, or a class containing methods), the children
        are preferred and the parent's gaps become block chunks.
        """
        # Flatten: prefer innermost declarations when they nest.
        flat = self._flatten_decls(decls)

        candidates: list[ChunkCandidate] = []
        consumed = 0

        for start, end, ctype, name in flat:
            # Emit gap block before this declaration.
            if start > consumed:
                gap = source[consumed:start]
                if gap.strip():
                    gap_line_start = source[:consumed].count("\n") + 1
                    gap_line_end = source[:start].count("\n")
                    gap_line_end = max(gap_line_start, gap_line_end)
                    candidates.append(
                        ChunkCandidate(
                            content=gap,
                            chunk_type="block",
                            line_start=gap_line_start,
                            line_end=gap_line_end,
                        )
                    )

            chunk_text = source[start : end + 1]
            line_start = source[:start].count("\n") + 1
            line_end = source[: end + 1].count("\n")
            line_end = max(line_start, line_end)

            candidates.append(
                ChunkCandidate(
                    content=chunk_text,
                    chunk_type=ctype,
                    line_start=line_start,
                    line_end=line_end,
                    symbol_name=name,
                )
            )
            consumed = end + 1

        # Trailing gap.
        if consumed < len(source):
            tail = source[consumed:]
            if tail.strip():
                tail_start = source[:consumed].count("\n") + 1
                tail_end = tail_start + tail.count("\n")
                tail_end = max(tail_start, tail_end)
                candidates.append(
                    ChunkCandidate(
                        content=tail,
                        chunk_type="block",
                        line_start=tail_start,
                        line_end=tail_end,
                    )
                )

        return candidates

    @staticmethod
    def _flatten_decls(
        decls: list[tuple[int, int, str, str]],
    ) -> list[tuple[int, int, str, str]]:
        """Flatten nested declarations by replacing parent containers with
        their children when nesting is detected.

        A parent declaration that wholly contains one or more child
        declarations is replaced by: a header chunk (from parent start
        to first child start) preserving the parent's symbol name and
        type, followed by the children.  Gap regions between children
        and after the last child become block chunks.

        Non-nested declarations are passed through unchanged.
        """
        if not decls:
            return []

        # Sort by start, then by descending end (outermost first).
        sorted_d = sorted(decls, key=lambda d: (d[0], -d[1]))

        result: list[tuple[int, int, str, str]] = []
        i = 0

        while i < len(sorted_d):
            parent = sorted_d[i]
            p_start, p_end, p_type, p_name = parent

            # Collect all children fully contained within this parent.
            children: list[tuple[int, int, str, str]] = []
            j = i + 1
            while j < len(sorted_d):
                c_start, c_end, _, _ = sorted_d[j]
                if c_start >= p_start and c_end <= p_end:
                    children.append(sorted_d[j])
                    j += 1
                else:
                    break

            if not children:
                # No nesting -- emit the declaration as-is.
                result.append(parent)
                i += 1
            else:
                # Parent contains children -- emit a header chunk for the
                # parent (from parent start to just before the first child)
                # so the parent's symbol_name and chunk_type are preserved.
                flattened_children = BraceChunker._flatten_decls(children)
                first_child_start = flattened_children[0][0]

                if first_child_start > p_start:
                    # Header: parent declaration line(s) up to first child.
                    header_end = first_child_start - 1
                    result.append((p_start, header_end, p_type, p_name))

                result.extend(flattened_children)
                i = j  # skip past all children

        # Sort the final result by start offset.
        result.sort(key=lambda d: d[0])
        return result

    # ------------------------------------------------------------------
    # Sliding window (for oversized chunks and fallback)
    # ------------------------------------------------------------------

    def _sliding_window(
        self,
        lines: list[str],
        symbol_name: str | None,
        chunk_type: str,
        line_offset: int = 0,
    ) -> list[ChunkCandidate]:
        """Split *lines* using a sliding window with ``overlap`` line overlap."""
        candidates: list[ChunkCandidate] = []
        n = len(lines)
        pos = 0
        is_first = True

        while pos < n:
            window: list[str] = []
            end = pos

            while end < n:
                window.append(lines[end])
                end += 1
                if estimate_tokens("".join(window)) >= self.max_tokens:
                    break

            content = "".join(window)
            abs_start = line_offset + pos + 1
            abs_end = line_offset + end

            candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type=chunk_type,
                    line_start=abs_start,
                    line_end=abs_end,
                    symbol_name=symbol_name if is_first else None,
                )
            )
            is_first = False

            if end >= n:
                break

            advance = max(1, len(window) - self.overlap)
            pos += advance

        # Merge tiny trailing fragment into predecessor.
        if len(candidates) >= 2:
            last = candidates[-1]
            if estimate_tokens(last.content) < self.min_tokens:
                prev = candidates[-2]
                combined = prev.content + last.content
                if estimate_tokens(combined) <= self.max_tokens:
                    candidates[-2] = ChunkCandidate(
                        content=combined,
                        chunk_type=prev.chunk_type,
                        line_start=prev.line_start,
                        line_end=last.line_end,
                        symbol_name=prev.symbol_name,
                    )
                    candidates.pop()

        return candidates

    # ------------------------------------------------------------------
    # Oversized chunk splitting
    # ------------------------------------------------------------------

    def _maybe_split(self, cand: ChunkCandidate) -> list[ChunkCandidate]:
        """If *cand* exceeds ``max_tokens``, split it with the sliding window."""
        if estimate_tokens(cand.content) <= self.max_tokens:
            return [cand]

        sub_lines = cand.content.splitlines(keepends=True)
        line_offset = cand.line_start - 1

        sub_chunks = self._sliding_window(
            lines=sub_lines,
            symbol_name=cand.symbol_name,
            chunk_type=cand.chunk_type,
            line_offset=line_offset,
        )
        return sub_chunks or [cand]
