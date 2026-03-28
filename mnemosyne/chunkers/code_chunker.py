# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
AST-based code chunker for Python source, with regex fallback for other languages.

For Python, the :class:`CodeChunker` walks the module-level AST to extract
logical units (imports, functions, classes, loose blocks).  Methods inside a
class are also extracted as child chunks so that granular retrieval is possible.

For non-Python languages a set of language-specific regex patterns is used to
find function/class/statement boundaries.  This is intentionally conservative
— it is better to produce a slightly larger chunk than to split mid-construct.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mnemosyne.models import estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ChunkCandidate:
    """
    A raw chunk candidate produced by a chunker before database insertion.

    Attributes:
        content:       The source text of the chunk.
        chunk_type:    Structural category: ``'function'``, ``'class'``,
                       ``'imports'``, or ``'block'``.
        line_start:    1-based inclusive start line.
        line_end:      1-based inclusive end line.
        symbol_name:   Qualified name of the symbol (function/class name), or
                       None for imports and loose blocks.
        parent_symbol: Name of the enclosing class when this chunk is a method,
                       otherwise None.
    """

    content: str
    chunk_type: str  # 'function' | 'class' | 'imports' | 'block'
    line_start: int
    line_end: int
    symbol_name: str | None = None
    parent_symbol: str | None = None


# ---------------------------------------------------------------------------
# Regex patterns for non-Python languages
# ---------------------------------------------------------------------------

# Each entry maps a language name to a list of compiled patterns whose match
# signals the start of a new logical chunk.  We look for these at the
# beginning of a line (after optional leading whitespace only for indented
# constructs).

_BOUNDARY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "javascript": [
        re.compile(r"^(export\s+)?(default\s+)?(async\s+)?function\s+\w+", re.M),
        re.compile(r"^(export\s+)?(abstract\s+)?class\s+\w+", re.M),
        re.compile(r"^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", re.M),
        re.compile(r"^(export\s+)?(const|let|var)\s+\w+\s*=\s*function", re.M),
    ],
    "typescript": [
        re.compile(r"^(export\s+)?(default\s+)?(async\s+)?function\s+\w+", re.M),
        re.compile(r"^(export\s+)?(abstract\s+)?class\s+\w+", re.M),
        re.compile(r"^(export\s+)?(interface|type|enum)\s+\w+", re.M),
        re.compile(r"^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", re.M),
        re.compile(r"^(export\s+)?(const|let|var)\s+\w+\s*=\s*function", re.M),
    ],
    "sql": [
        re.compile(
            r"^(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH)\b",
            re.M | re.I,
        ),
    ],
    "shell": [
        re.compile(r"^\w[\w_-]*\s*\(\s*\)\s*\{", re.M),
        re.compile(r"^function\s+\w+", re.M),
    ],
    "html": [
        re.compile(r"^<(div|section|article|header|footer|main|nav|form|script|style)\b", re.M | re.I),
    ],
    "css": [
        re.compile(r"^[\w.#:\[\*@].*\{", re.M),
    ],
}


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class CodeChunker:
    """
    Structural code chunker.

    Uses Python's ``ast`` module for Python source and language-specific regex
    boundary detection for all other languages.
    """

    def __init__(self, config: "Config") -> None:
        self.max_tokens: int = config.chunking.max_chunk_tokens
        self.min_tokens: int = config.chunking.min_chunk_tokens
        self.granularity: str = config.chunking.code_granularity  # 'function' | 'file'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, source: str, language: str = "python") -> list[ChunkCandidate]:
        """
        Split *source* into :class:`ChunkCandidate` objects.

        Args:
            source:   Full source text of the file.
            language: Language tag (e.g. ``"python"``).

        Returns:
            Ordered list of chunk candidates.  Never empty for non-empty source.
        """
        if not source.strip():
            return []
        if language == "python":
            return self._chunk_python(source)
        return self._chunk_by_regex(source, language)

    # ------------------------------------------------------------------
    # Python AST chunker
    # ------------------------------------------------------------------

    def _chunk_python(self, source: str) -> list[ChunkCandidate]:
        """Use the ``ast`` module to extract logical units from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._chunk_by_regex(source, "python")

        source_lines = source.splitlines(keepends=True)
        candidates: list[ChunkCandidate] = []

        # Track consecutive import statements so we can merge them.
        import_run: list[ast.stmt] = []

        def _flush_imports() -> None:
            if not import_run:
                return
            first = import_run[0]
            last = import_run[-1]
            content = "".join(source_lines[first.lineno - 1 : last.end_lineno])
            candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type="imports",
                    line_start=first.lineno,
                    line_end=last.end_lineno,
                )
            )
            import_run.clear()

        # Loose non-import, non-function, non-class top-level statements
        # accumulate into "block" chunks.
        block_run: list[ast.stmt] = []

        def _flush_block() -> None:
            if not block_run:
                return
            first = block_run[0]
            last = block_run[-1]
            content = "".join(source_lines[first.lineno - 1 : last.end_lineno])
            candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type="block",
                    line_start=first.lineno,
                    line_end=last.end_lineno,
                )
            )
            block_run.clear()

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _flush_block()
                import_run.append(node)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _flush_imports()
                _flush_block()
                fn_chunks = self._extract_function(node, source_lines)
                candidates.extend(fn_chunks)

            elif isinstance(node, ast.ClassDef):
                _flush_imports()
                _flush_block()
                cls_chunks = self._extract_class(node, source_lines)
                candidates.extend(cls_chunks)

            else:
                _flush_imports()
                block_run.append(node)

        _flush_imports()
        _flush_block()

        # Post-process: split any chunk that is over max_tokens
        result: list[ChunkCandidate] = []
        for cand in candidates:
            result.extend(self._maybe_split(cand, source_lines))

        return result or self._whole_file_chunk(source)

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source_lines: list[str],
        parent_symbol: str | None = None,
    ) -> list[ChunkCandidate]:
        """Create a ChunkCandidate for a function/method node."""
        content = "".join(source_lines[node.lineno - 1 : node.end_lineno])
        return [
            ChunkCandidate(
                content=content,
                chunk_type="function",
                line_start=node.lineno,
                line_end=node.end_lineno,
                symbol_name=node.name,
                parent_symbol=parent_symbol,
            )
        ]

    def _extract_class(
        self,
        node: ast.ClassDef,
        source_lines: list[str],
    ) -> list[ChunkCandidate]:
        """
        Create a class-level chunk plus individual method chunks.

        The class chunk contains the full class text so that the class
        signature and docstring are always retrievable.  Method chunks allow
        granular retrieval when only a specific method is needed.
        """
        class_content = "".join(source_lines[node.lineno - 1 : node.end_lineno])
        chunks: list[ChunkCandidate] = [
            ChunkCandidate(
                content=class_content,
                chunk_type="class",
                line_start=node.lineno,
                line_end=node.end_lineno,
                symbol_name=node.name,
            )
        ]

        # Only add method-level chunks when granularity is 'function'.
        if self.granularity == "function":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.extend(
                        self._extract_function(item, source_lines, parent_symbol=node.name)
                    )

        return chunks

    # ------------------------------------------------------------------
    # Post-processing: split oversized chunks
    # ------------------------------------------------------------------

    def _maybe_split(
        self,
        cand: ChunkCandidate,
        source_lines: list[str],
    ) -> list[ChunkCandidate]:
        """
        If *cand* exceeds ``max_tokens``, attempt to split it.

        For class chunks: split at inner method boundaries.
        For function/block/imports chunks: split at the midpoint.
        """
        if estimate_tokens(cand.content) <= self.max_tokens:
            return [cand]

        lines = cand.content.splitlines(keepends=True)

        if cand.chunk_type == "class":
            return self._split_class_chunk(cand, lines)
        else:
            return self._split_at_midpoint(cand, lines)

    def _split_class_chunk(
        self,
        cand: ChunkCandidate,
        lines: list[str],
    ) -> list[ChunkCandidate]:
        """
        Split an oversized class chunk at method boundaries (``def`` lines).

        Keeps the class header (lines before the first method) in the first
        sub-chunk, then groups methods into sub-chunks that stay within budget.
        """
        method_starts: list[int] = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                method_starts.append(i)

        if not method_starts:
            # No inner methods; fall back to midpoint split.
            return self._split_at_midpoint(cand, lines)

        result: list[ChunkCandidate] = []
        # Boundaries: class header → each method → remainder after last method
        boundaries = [0] + method_starts + [len(lines)]
        current_lines: list[str] = []
        current_start_offset = 0  # offset into cand.content lines

        for seg_start, seg_end in zip(boundaries, boundaries[1:]):
            seg = lines[seg_start:seg_end]
            candidate_lines = current_lines + seg
            if (
                estimate_tokens("".join(candidate_lines)) > self.max_tokens
                and current_lines
            ):
                # Flush current accumulation as a sub-chunk
                abs_start = cand.line_start + current_start_offset
                abs_end = abs_start + len(current_lines) - 1
                result.append(
                    ChunkCandidate(
                        content="".join(current_lines),
                        chunk_type="block",
                        line_start=abs_start,
                        line_end=abs_end,
                        symbol_name=cand.symbol_name,
                        parent_symbol=cand.parent_symbol,
                    )
                )
                current_start_offset += len(current_lines)
                current_lines = seg
            else:
                current_lines = candidate_lines

        if current_lines:
            abs_start = cand.line_start + current_start_offset
            abs_end = abs_start + len(current_lines) - 1
            result.append(
                ChunkCandidate(
                    content="".join(current_lines),
                    chunk_type="block",
                    line_start=abs_start,
                    line_end=abs_end,
                    symbol_name=cand.symbol_name,
                    parent_symbol=cand.parent_symbol,
                )
            )

        return result if result else [cand]

    def _split_at_midpoint(
        self,
        cand: ChunkCandidate,
        lines: list[str],
    ) -> list[ChunkCandidate]:
        """Split *cand* into two roughly equal halves."""
        mid = max(1, len(lines) // 2)
        first_lines = lines[:mid]
        second_lines = lines[mid:]

        result: list[ChunkCandidate] = []
        if first_lines:
            result.append(
                ChunkCandidate(
                    content="".join(first_lines),
                    chunk_type=cand.chunk_type,
                    line_start=cand.line_start,
                    line_end=cand.line_start + mid - 1,
                    symbol_name=cand.symbol_name,
                    parent_symbol=cand.parent_symbol,
                )
            )
        if second_lines:
            result.append(
                ChunkCandidate(
                    content="".join(second_lines),
                    chunk_type=cand.chunk_type,
                    line_start=cand.line_start + mid,
                    line_end=cand.line_end,
                    symbol_name=cand.symbol_name,
                    parent_symbol=cand.parent_symbol,
                )
            )
        return result if result else [cand]

    def _whole_file_chunk(self, source: str) -> list[ChunkCandidate]:
        """Fallback: treat the entire source as a single block chunk."""
        lines = source.splitlines()
        return [
            ChunkCandidate(
                content=source,
                chunk_type="block",
                line_start=1,
                line_end=max(1, len(lines)),
            )
        ]

    # ------------------------------------------------------------------
    # Regex-based fallback for non-Python languages
    # ------------------------------------------------------------------

    def _chunk_by_regex(self, source: str, language: str) -> list[ChunkCandidate]:
        """
        Split *source* into chunks using language-specific regex boundary patterns.

        For each pattern match, we treat that line as the start of a new chunk.
        Lines between two boundary markers form a single chunk.  If no patterns
        are defined for *language*, the source is split purely by the token
        budget (sliding window with no overlap — use GenericChunker for overlap).
        """
        patterns = _BOUNDARY_PATTERNS.get(language, [])
        lines = source.splitlines(keepends=True)

        if not patterns:
            # No language-specific patterns: chunk by token budget only.
            return self._budget_split(lines)

        # Collect all boundary line indices (0-based).
        boundary_set: set[int] = {0}
        for pat in patterns:
            for m in pat.finditer(source):
                # Convert char offset to line index
                line_idx = source[: m.start()].count("\n")
                boundary_set.add(line_idx)

        boundaries = sorted(boundary_set)
        # Append sentinel
        boundaries.append(len(lines))

        candidates: list[ChunkCandidate] = []
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_lines = lines[seg_start:seg_end]
            if not seg_lines:
                continue
            content = "".join(seg_lines)
            if not content.strip():
                continue
            candidates.append(
                ChunkCandidate(
                    content=content,
                    chunk_type="block",
                    line_start=seg_start + 1,  # 1-based
                    line_end=seg_end,
                )
            )

        # Split any over-budget chunks
        result: list[ChunkCandidate] = []
        for cand in candidates:
            cand_lines = cand.content.splitlines(keepends=True)
            result.extend(self._maybe_split(cand, cand_lines))

        return result or self._whole_file_chunk(source)

    def _budget_split(self, lines: list[str]) -> list[ChunkCandidate]:
        """
        Greedily accumulate lines into chunks that stay within ``max_tokens``.

        Used when no boundary patterns exist for the target language.
        """
        candidates: list[ChunkCandidate] = []
        current: list[str] = []
        current_start = 1  # 1-based

        for i, line in enumerate(lines, start=1):
            current.append(line)
            if estimate_tokens("".join(current)) >= self.max_tokens:
                candidates.append(
                    ChunkCandidate(
                        content="".join(current),
                        chunk_type="block",
                        line_start=current_start,
                        line_end=i,
                    )
                )
                current = []
                current_start = i + 1

        if current:
            candidates.append(
                ChunkCandidate(
                    content="".join(current),
                    chunk_type="block",
                    line_start=current_start,
                    line_end=current_start + len(current) - 1,
                )
            )

        return candidates or self._whole_file_chunk("".join(lines))
