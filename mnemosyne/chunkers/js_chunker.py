# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
JavaScript-aware structural chunker for Mnemosyne.

Extracts function declarations, class declarations, arrow function constants,
and method definitions from JavaScript (and TypeScript) source.  Each
extracted symbol receives a ``symbol_name`` and an appropriate ``chunk_type``.
Code between declarations is grouped into ``block`` chunks.  Oversized chunks
are split with a sliding window, preserving the ``symbol_name`` on the first
sub-chunk.

Brace-extent detection correctly skips braces inside:
  - single-quoted string literals  ('...')
  - double-quoted string literals  ("...")
  - template literals              (`...`)
  - line comments                  (// ...)
  - block comments                 (/* ... */)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.chunkers.code_chunker import ChunkCandidate
from mnemosyne.models import estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Regex patterns for JavaScript declaration boundaries
# ---------------------------------------------------------------------------

# Each pattern captures the symbol name in a named group ``name``.
# All patterns are anchored to the start of a line (re.M).  Leading
# whitespace is intentionally NOT consumed so that method definitions inside
# a class body (which are indented) are also matched.

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # export default async function foo(  /  export default function foo(
    # export async function foo(  /  export function foo(
    # async function foo(  /  function foo(
    (
        re.compile(
            r"^[ \t]*(export\s+)?(default\s+)?(async\s+)?function\s+(?P<name>\w+)\s*[\(<]",
            re.M,
        ),
        "function",
    ),
    # export class Foo  /  class Foo
    (
        re.compile(
            r"^[ \t]*(export\s+)?(abstract\s+)?class\s+(?P<name>\w+)(?:\s|{|<)",
            re.M,
        ),
        "class",
    ),
    # const/let/var foo = async (...) =>
    # const/let/var foo = (...) =>
    (
        re.compile(
            r"^[ \t]*(export\s+)?(const|let|var)\s+(?P<name>\w+)\s*=\s*(async\s*)?\([^)]*\)\s*=>",
            re.M,
        ),
        "function",
    ),
    # const/let/var foo = async singleParam => ...
    (
        re.compile(
            r"^[ \t]*(export\s+)?(const|let|var)\s+(?P<name>\w+)\s*=\s*(async\s+)\w+\s*=>",
            re.M,
        ),
        "function",
    ),
    # const/let/var foo = function(  /  const/let/var foo = async function(
    (
        re.compile(
            r"^[ \t]*(export\s+)?(const|let|var)\s+(?P<name>\w+)\s*=\s*(async\s+)?function\s*[\w(]",
            re.M,
        ),
        "function",
    ),
    # const/let/var FOO = { ... }  or  const/let/var FOO = [ ... ]
    # Data definitions (object literals, arrays) — named so they get the
    # structured-code boost in retrieval ranking.
    (
        re.compile(
            r"^[ \t]*(export\s+)?(const|let|var)\s+(?P<name>\w+)\s*=\s*[\[{]",
            re.M,
        ),
        "block",
    ),
]

# Method definition inside a class body.
# Matches lines like:  methodName(  /  async methodName(  /  static methodName(
# Also getter/setter: get foo(  /  set foo(
# Does NOT match 'function' keyword lines (those are caught above).
_METHOD_PATTERN = re.compile(
    r"^(?P<indent>[ \t]+)"
    r"(?:(?:static|async|get|set|public|private|protected|abstract|override)\s+)*"
    r"(?!function\b)(?!class\b)(?!const\b)(?!let\b)(?!var\b)"
    r"(?P<name>\w+)\s*[\(<]",
    re.M,
)

# Reserved words that look like method calls but are not method definitions.
_JS_KEYWORDS: frozenset[str] = frozenset({
    "if", "for", "while", "switch", "catch", "return",
    "const", "let", "var", "new", "typeof", "instanceof",
    "import", "export", "default", "class", "extends",
    "super", "this", "null", "undefined", "true", "false",
    "async", "await", "static", "get", "set", "delete",
    "throw", "try", "finally", "yield", "of", "in",
})


# ---------------------------------------------------------------------------
# Brace-extent helper
# ---------------------------------------------------------------------------


def _find_block_end(source: str, open_pos: int) -> int:
    """
    Return the index (inclusive) of the closing ``}`` that matches the ``{``
    at *open_pos* in *source*.

    The scan correctly skips over:
    - string literals (single, double, and backtick-quoted)
    - line comments  (``//``)
    - block comments (``/* ... */``)

    If no matching closing brace is found (malformed source), returns the last
    character index of *source*.

    Args:
        source:   Full source text.
        open_pos: Index of the opening ``{``.

    Returns:
        Index of the matching ``}`` or ``len(source) - 1`` on failure.
    """
    depth = 0
    i = open_pos
    n = len(source)

    while i < n:
        ch = source[i]

        # ---- line comment ------------------------------------------------
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue

        # ---- block comment -----------------------------------------------
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            else:
                i = n  # unterminated block comment; consume to end
            continue

        # ---- string literals ---------------------------------------------
        if ch in ('"', "'", "`"):
            quote = ch
            i += 1
            while i < n:
                c = source[i]
                if c == "\\" and i + 1 < n:
                    i += 2  # skip escaped character
                    continue
                if c == quote:
                    i += 1
                    break
                # Template literal expressions ${ ... } need depth tracking
                if quote == "`" and c == "$" and i + 1 < n and source[i + 1] == "{":
                    i += 1  # now at '{'
                    i = _find_block_end(source, i) + 1
                    continue
                i += 1
            continue

        # ---- brace counting ---------------------------------------------
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    # Malformed source — return end of string
    return n - 1


def _find_statement_end(source: str, start: int) -> int:
    """
    Find the end of a JavaScript statement beginning at *start*.

    Scans forward skipping string literals and comments, stopping at the
    first semicolon or at the end of the line (for concise arrow bodies
    without braces).  This is used for arrow functions whose body is a
    single expression (no ``{``).

    Returns the inclusive index of the last character of the statement
    (the ``;`` if present, otherwise the last non-whitespace character on
    the line, or the newline).

    Args:
        source: Full source text.
        start:  Position to start scanning from (should be just after ``=>``).

    Returns:
        Inclusive end index.
    """
    i = start
    n = len(source)

    while i < n:
        ch = source[i]

        # Line comment ends at newline
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            return i - 1 if i > start else i

        # Block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue

        # String literals
        if ch in ('"', "'", "`"):
            quote = ch
            i += 1
            while i < n:
                c = source[i]
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == quote:
                    break
                i += 1
            i += 1
            continue

        # Semicolon — end of statement
        if ch == ";":
            return i

        # Newline — end of concise body (common JS style)
        if ch == "\n":
            return i - 1 if i > start else i

        i += 1

    return n - 1


# ---------------------------------------------------------------------------
# Symbol boundary dataclass (internal)
# ---------------------------------------------------------------------------


class _Boundary:
    """Internal record for a detected declaration boundary."""

    __slots__ = ("match_start", "line_start", "chunk_type", "symbol_name")

    def __init__(
        self,
        match_start: int,
        line_start: int,
        chunk_type: str,
        symbol_name: str,
    ) -> None:
        self.match_start = match_start
        self.line_start = line_start  # 1-based
        self.chunk_type = chunk_type
        self.symbol_name = symbol_name


# ---------------------------------------------------------------------------
# JSChunker
# ---------------------------------------------------------------------------


class JSChunker:
    """
    JavaScript-aware structural code chunker.

    Extracts named symbols (functions, classes, arrow-function constants,
    and class methods) from JavaScript source.  Each symbol gets a
    ``symbol_name`` on its :class:`~mnemosyne.chunkers.code_chunker.ChunkCandidate`.
    Interstitial code is emitted as ``block`` chunks.

    Oversized chunks are split with a sliding window; the first sub-chunk
    retains the original ``symbol_name``.

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

    def chunk(self, source: str, language: str = "javascript") -> list[ChunkCandidate]:
        """
        Split *source* into :class:`ChunkCandidate` objects.

        Args:
            source:   Full source text of the file.
            language: Language tag (informational; ``'javascript'`` or
                      ``'typescript'``).

        Returns:
            Ordered list of chunk candidates.  Never empty for non-empty source.
        """
        if not source.strip():
            return []

        lines = source.splitlines(keepends=True)
        candidates = self._extract_symbols(source, lines)

        if not candidates:
            # No recognisable declarations — fall back to sliding window.
            return self._sliding_window(lines, symbol_name=None, chunk_type="block")

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
    # Symbol extraction
    # ------------------------------------------------------------------

    def _extract_symbols(
        self, source: str, lines: list[str]
    ) -> list[ChunkCandidate]:
        """
        Find all top-level declarations, determine their extents via brace
        counting, and build ChunkCandidate objects.

        Interstitial text between declarations becomes ``block`` chunks.
        For class declarations the full class body is emitted as one chunk
        *and* each method inside is emitted as an additional function chunk
        (mirroring the Python AST chunker's behaviour).  Method symbol names
        are already qualified as ``ClassName.methodName`` by
        :meth:`_find_boundaries`.
        """
        boundaries = self._find_boundaries(source)
        if not boundaries:
            return []

        # Separate top-level boundaries from method boundaries so we can
        # handle them independently.  A boundary is a "method boundary" if
        # its symbol_name contains a dot (i.e. "ClassName.method").
        top_level: list[_Boundary] = []
        method_bdries: list[_Boundary] = []
        for b in boundaries:
            if "." in b.symbol_name:
                method_bdries.append(b)
            else:
                top_level.append(b)

        candidates: list[ChunkCandidate] = []
        # Character offset of the start of unconsumed source.
        consumed_char = 0

        for bdry in top_level:
            # Emit any interstitial code before this boundary as a block.
            if bdry.match_start > consumed_char:
                interstitial = source[consumed_char : bdry.match_start]
                if interstitial.strip():
                    seg_line_start = source[:consumed_char].count("\n") + 1
                    seg_line_end = source[: bdry.match_start].count("\n")
                    seg_line_end = max(seg_line_start, seg_line_end)
                    candidates.append(
                        ChunkCandidate(
                            content=interstitial,
                            chunk_type="block",
                            line_start=seg_line_start,
                            line_end=seg_line_end,
                            symbol_name=None,
                        )
                    )

            # Determine the end of this symbol's chunk.
            end_char = self._find_symbol_end(source, bdry)

            chunk_text = source[bdry.match_start : end_char + 1]
            chunk_line_end = source[: end_char + 1].count("\n")
            chunk_line_end = max(bdry.line_start, chunk_line_end)

            candidates.append(
                ChunkCandidate(
                    content=chunk_text,
                    chunk_type=bdry.chunk_type,
                    line_start=bdry.line_start,
                    line_end=chunk_line_end,
                    symbol_name=bdry.symbol_name,
                )
            )

            # For class declarations, also emit each method as a separate
            # function chunk so granular retrieval is possible.
            if bdry.chunk_type == "class":
                for mbdry in method_bdries:
                    if bdry.match_start < mbdry.match_start <= end_char:
                        m_end = self._find_symbol_end(source, mbdry)
                        m_text = source[mbdry.match_start : m_end + 1]
                        m_line_end = source[: m_end + 1].count("\n")
                        m_line_end = max(mbdry.line_start, m_line_end)
                        candidates.append(
                            ChunkCandidate(
                                content=m_text,
                                chunk_type="function",
                                line_start=mbdry.line_start,
                                line_end=m_line_end,
                                symbol_name=mbdry.symbol_name,
                            )
                        )

            consumed_char = end_char + 1

        # Emit any trailing code after the last symbol.
        if consumed_char < len(source):
            tail = source[consumed_char:]
            if tail.strip():
                tail_line_start = source[:consumed_char].count("\n") + 1
                tail_line_end = tail_line_start + tail.count("\n") - 1
                tail_line_end = max(tail_line_start, tail_line_end)
                candidates.append(
                    ChunkCandidate(
                        content=tail,
                        chunk_type="block",
                        line_start=tail_line_start,
                        line_end=tail_line_end,
                        symbol_name=None,
                    )
                )

        return candidates

    def _find_symbol_end(self, source: str, bdry: _Boundary) -> int:
        """
        Return the inclusive character index of the last character belonging
        to the symbol starting at *bdry.match_start*.

        Strategy:
        1. Search for the first ``{`` that belongs to this symbol's block.
           A ``{`` "belongs" to this symbol if it is closer than any
           unambiguous statement terminator (``\\n`` at the top level for
           concise arrow bodies, or ``;``).
        2. If a block ``{`` is found, use brace counting to find its matching
           ``}``.
        3. Otherwise (concise arrow body), find the end of the expression
           statement.

        Args:
            source: Full source text.
            bdry:   The boundary record for the symbol.

        Returns:
            Inclusive end index.
        """
        n = len(source)
        start = bdry.match_start

        # Find the first '{' scanning forward from the declaration start,
        # being careful to skip strings and comments.
        brace_pos = _next_brace_or_statement_end(source, start)

        if brace_pos is None or source[brace_pos] != "{":
            # No block body — concise arrow or abstract declaration.
            # Use end-of-statement as the chunk boundary.
            stmt_end = _find_statement_end(source, start)
            # Include trailing newline if present.
            if stmt_end + 1 < n and source[stmt_end + 1] == "\n":
                stmt_end += 1
            return stmt_end

        # Block body: brace-count to find the matching '}'.
        close_pos = _find_block_end(source, brace_pos)
        end_char = close_pos

        # Include optional trailing semicolon.
        j = end_char + 1
        while j < n and source[j] in (" ", "\t"):
            j += 1
        if j < n and source[j] == ";":
            end_char = j

        # Include trailing newline.
        if end_char + 1 < n and source[end_char + 1] == "\n":
            end_char += 1

        return end_char

    def _find_boundaries(self, source: str) -> list[_Boundary]:
        """
        Collect all declaration boundaries from *source* using the compiled
        regex patterns, sorted by character offset.

        Top-level declarations come from :data:`_PATTERNS`.  Method
        definitions are discovered by finding class bodies and scanning them
        with :data:`_METHOD_PATTERN`; each method gets a fully-qualified
        ``ClassName.methodName`` symbol name.

        Duplicates at the same character offset are collapsed (first match
        wins).
        """
        raw: list[_Boundary] = []

        # Top-level declarations.
        for pattern, chunk_type in _PATTERNS:
            for m in pattern.finditer(source):
                raw.append(
                    _Boundary(
                        match_start=m.start(),
                        line_start=source[: m.start()].count("\n") + 1,
                        chunk_type=chunk_type,
                        symbol_name=m.group("name"),
                    )
                )

        # Method definitions inside class bodies.
        class_pattern = re.compile(
            r"^[ \t]*(export\s+)?(abstract\s+)?class\s+(?P<name>\w+)(?:\s|{|<)",
            re.M,
        )
        for cls_m in class_pattern.finditer(source):
            cls_name = cls_m.group("name")
            brace_pos = source.find("{", cls_m.start())
            if brace_pos == -1:
                continue
            cls_end = _find_block_end(source, brace_pos)
            class_body = source[brace_pos + 1 : cls_end]
            body_offset = brace_pos + 1

            for meth_m in _METHOD_PATTERN.finditer(class_body):
                meth_name = meth_m.group("name")
                if meth_name in _JS_KEYWORDS:
                    continue
                abs_start = body_offset + meth_m.start()
                raw.append(
                    _Boundary(
                        match_start=abs_start,
                        line_start=source[:abs_start].count("\n") + 1,
                        chunk_type="function",
                        symbol_name=f"{cls_name}.{meth_name}",
                    )
                )

        # Sort by character position; deduplicate by match_start (first wins).
        raw.sort(key=lambda b: b.match_start)
        seen: set[int] = set()
        unique: list[_Boundary] = []
        for b in raw:
            if b.match_start not in seen:
                seen.add(b.match_start)
                unique.append(b)

        return unique

    # ------------------------------------------------------------------
    # Sliding-window fallback
    # ------------------------------------------------------------------

    def _sliding_window(
        self,
        lines: list[str],
        symbol_name: str | None,
        chunk_type: str,
        line_offset: int = 0,
    ) -> list[ChunkCandidate]:
        """
        Split *lines* using a sliding window with ``overlap`` line overlap.

        The first sub-chunk receives *symbol_name*; subsequent sub-chunks
        get ``symbol_name=None`` so that retrieval correctly targets the
        symbol entry point.

        Args:
            lines:        Lines to split (with newlines kept).
            symbol_name:  Symbol name for the first sub-chunk.
            chunk_type:   Chunk type for all sub-chunks.
            line_offset:  0-based line index of the first line in *lines*
                          within the original source (used to compute
                          absolute line numbers).

        Returns:
            List of :class:`ChunkCandidate` objects.
        """
        candidates: list[ChunkCandidate] = []
        n = len(lines)
        pos = 0  # 0-based index into lines
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
            abs_start = line_offset + pos + 1   # 1-based
            abs_end = line_offset + end          # 1-based inclusive

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

        # Merge a tiny trailing fragment into its predecessor.
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
    # Post-processing: split oversized chunks
    # ------------------------------------------------------------------

    def _maybe_split(self, cand: ChunkCandidate) -> list[ChunkCandidate]:
        """
        If *cand* exceeds ``max_tokens``, split it with the sliding window.

        The first sub-chunk retains the original ``symbol_name``; all
        subsequent sub-chunks get ``symbol_name=None``.

        Args:
            cand: Candidate to check and potentially split.

        Returns:
            A list with one or more :class:`ChunkCandidate` objects.
        """
        if estimate_tokens(cand.content) <= self.max_tokens:
            return [cand]

        sub_lines = cand.content.splitlines(keepends=True)
        line_offset = cand.line_start - 1  # convert to 0-based for arithmetic

        sub_chunks = self._sliding_window(
            lines=sub_lines,
            symbol_name=cand.symbol_name,
            chunk_type=cand.chunk_type,
            line_offset=line_offset,
        )

        return sub_chunks or [cand]


# ---------------------------------------------------------------------------
# Module-level helper (used by JSChunker._find_symbol_end)
# ---------------------------------------------------------------------------


def _next_brace_or_statement_end(source: str, start: int) -> int | None:
    """
    Scan forward from *start*, skipping strings and comments, and return
    the index of the first ``{`` found before any unambiguous statement
    boundary.

    A statement boundary is a ``;`` or, for concise arrow bodies, a newline
    that follows the ``=>`` token without an opening brace on the same line.

    Returns:
        Index of the ``{`` character, or ``None`` if a statement end is
        reached first (indicating a concise/no-body declaration).
    """
    i = start
    n = len(source)
    # Track whether we've passed the arrow (=>) so we know when a bare
    # newline terminates a concise body.
    after_arrow = False
    paren_depth = 0  # depth of ( ) — ignore newlines inside parameter lists

    while i < n:
        ch = source[i]

        # Line comment
        if ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
            continue

        # Block comment
        if ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i < n - 1:
                if source[i] == "*" and source[i + 1] == "/":
                    i += 2
                    break
                i += 1
            continue

        # String literal
        if ch in ('"', "'", "`"):
            quote = ch
            i += 1
            while i < n:
                c = source[i]
                if c == "\\" and i + 1 < n:
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                i += 1
            continue

        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "{":
            return i
        elif ch == ";":
            return i  # caller will see ';', not '{'
        elif ch == "\n" and after_arrow and paren_depth == 0:
            # Concise arrow body terminated by newline
            return i  # caller will see '\n', not '{'
        elif ch == "=" and i + 1 < n and source[i + 1] == ">":
            after_arrow = True
            i += 2
            continue

        i += 1

    return None
