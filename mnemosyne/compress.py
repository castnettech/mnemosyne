# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Context compression engine for Mnemosyne.

The :class:`Compressor` applies a four-stage pipeline to reduce a
:class:`~mnemosyne.models.Chunk`'s token count while preserving the
information most useful for LLM context injection.

Stages
------
1. **Mark preserved** -- identify lines that must never be removed (signatures,
   docstrings, return/raise/assert statements, annotated comments).
2. **Collapse boilerplate** -- replace repetitive or low-value patterns with
   compact summaries (``# [N imports: ...]``, ``# [N log statements]``).
3. **TF-IDF importance filter** -- score non-preserved lines by the IDF weight
   of their vocabulary terms and remove the lowest-scoring lines until the
   ``target_ratio`` is reached.
4. **Density filter** -- remove near-duplicate lines and normalise whitespace.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.config import Config
    from mnemosyne.models import Chunk
    from mnemosyne.embeddings.tfidf_backend import TFIDFBackend


# ---------------------------------------------------------------------------
# Regex helpers used across multiple stages
# ---------------------------------------------------------------------------

# Import lines (Python)
_IMPORT_RE = re.compile(r"^\s*(import|from)\s+")

# Logging / print statements
_LOG_RE = re.compile(
    r"^\s*(logger\.|logging\.|log\.|print\(|self\._log|self\.log)",
    re.IGNORECASE,
)

# Consecutive self.x = y or x = y assignment patterns (boilerplate setters)
_ASSIGNMENT_RE = re.compile(r"^\s*self\.\w+\s*=\s*\S+")

# Signature lines: def / async def / class declarations
_SIGNATURE_RE = re.compile(r"^\s*(async\s+)?def\s+\w+|^\s*class\s+\w+")

# Return and raise statements
_RETURN_RAISE_RE = re.compile(r"^\s*(return|raise|yield)\b")

# Assert statements
_ASSERT_RE = re.compile(r"^\s*assert\b")

# Annotation comments: TODO, FIXME, HACK, NOTE, XXX
_ANNOTATION_COMMENT_RE = re.compile(r"#\s*(TODO|FIXME|HACK|NOTE|XXX)\b", re.IGNORECASE)

# Control flow lines -- structurally load-bearing; removing them destroys semantics
_CONTROL_FLOW_RE = re.compile(
    r"^\s*(if\b|elif\b|else\s*:|for\b|while\b|try\s*:|except\b|finally\s*:|with\b|switch\b|case\b)"
)

# Triple-quoted string opener/closer
_TRIPLE_QUOTE_RE = re.compile(r'"""|\'\'\'' )

# Consecutive blank lines
_BLANK_LINE_RE = re.compile(r"^\s*$")


class Compressor:
    """
    Four-stage context compressor.

    Args:
        config:        Mnemosyne :class:`~mnemosyne.config.Config` instance.
        tfidf_backend: Optional :class:`~mnemosyne.embeddings.tfidf_backend.TFIDFBackend`
                       used in stage 3.  When ``None``, stage 3 is skipped.
    """

    def __init__(self, config: "Config", tfidf_backend: "TFIDFBackend | None" = None) -> None:
        self.config = config.compression
        self.tfidf = tfidf_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        chunk: "Chunk",
        context_idf: "dict[str, float] | None" = None,
        strict: bool = False,
    ) -> str:
        """
        Apply the full four-stage compression pipeline to *chunk*.

        Args:
            chunk:       The :class:`~mnemosyne.models.Chunk` to compress.
            context_idf: Optional per-query IDF override for stage 3.
                         When provided, terms that are rare *in the current
                         query context* are weighted more heavily, improving
                         relevance-aware pruning.  Falls back to the global
                         :attr:`tfidf.idf` when ``None``.
            strict:      When True, skip Stage 3 (TF-IDF importance filter)
                         entirely.  Used for symbol chunks (named functions/
                         classes) where code is semantically dense and thinning
                         loses more than it saves.

        Returns:
            Compressed source text.  Always returns a non-empty string
            (at minimum the preserved lines).
        """
        text = chunk.content
        if not text.strip():
            return text

        lines = text.splitlines(keepends=True)

        # Stage 1: Identify lines that must be kept verbatim.
        preserved: set[int] = self._mark_preserved(lines, chunk.chunk_type)

        # Stage 2: Replace boilerplate runs with compact inline summaries.
        lines, preserved = self._collapse_boilerplate(lines, preserved)

        # Stage 3: Remove low-importance lines based on TF-IDF weights.
        # Skipped in strict mode (symbol chunks where density matters).
        if (
            not strict
            and self.tfidf is not None
            and self.config.target_ratio < 1.0
        ):
            lines, preserved = self._importance_filter(lines, preserved, context_idf)

        # Stage 4: Remove near-duplicate lines and normalise whitespace runs.
        lines = self._density_filter(lines, preserved)

        return "".join(lines)

    # ------------------------------------------------------------------
    # Stage 1: Mark preserved lines
    # ------------------------------------------------------------------

    def _mark_preserved(self, lines: list[str], chunk_type: str) -> set[int]:
        """
        Return the set of 0-based line indices that must not be removed.

        Always preserved:
        - Function/class signature lines (``def``, ``async def``, ``class``).
        - Return, raise, and yield statements.
        - Assert statements.
        - Lines inside triple-quoted docstrings.
        - Lines containing TODO/FIXME/HACK/NOTE/XXX comments.

        When ``config.preserve_signatures`` is False, signature lines are
        still preserved (they are structurally load-bearing).
        """
        preserved: set[int] = set()

        in_docstring = False
        docstring_quote: str | None = None

        for i, line in enumerate(lines):
            # Track docstring state
            quotes_found = _TRIPLE_QUOTE_RE.findall(line)
            for q in quotes_found:
                if not in_docstring:
                    in_docstring = True
                    docstring_quote = q
                elif q == docstring_quote:
                    in_docstring = False
                    docstring_quote = None
                    preserved.add(i)  # closing line
                    continue

            if in_docstring and self.config.preserve_docstrings:
                preserved.add(i)
                continue

            # Signature lines
            if _SIGNATURE_RE.match(line):
                preserved.add(i)
                continue

            # Return / raise / yield
            if _RETURN_RAISE_RE.match(line):
                preserved.add(i)
                continue

            # Assert
            if _ASSERT_RE.match(line):
                preserved.add(i)
                continue

            # Annotated comments
            if _ANNOTATION_COMMENT_RE.search(line):
                preserved.add(i)
                continue

            # Control flow lines (if/elif/else, for/while, try/except/finally,
            # switch/case, with) -- structurally load-bearing
            if _CONTROL_FLOW_RE.match(line):
                preserved.add(i)
                continue

        return preserved

    # ------------------------------------------------------------------
    # Stage 2: Collapse boilerplate
    # ------------------------------------------------------------------

    def _collapse_boilerplate(
        self,
        lines: list[str],
        preserved: set[int],
    ) -> tuple[list[str], set[int]]:
        """
        Replace boilerplate patterns with compact single-line summaries.

        Patterns handled:
        - **Import blocks** > 3 consecutive import lines -> ``# [N imports: a, b, ...]``
        - **Consecutive self.x = x assignments** -> ``# [N assignments: x, y, ...]``
        - **Logging/print calls** -> accumulated count, then ``# [N log statements]``
        - **Consecutive blank lines** > 1 -> single blank line

        Returns updated ``(lines, preserved)`` tuple.
        """
        if not self.config.collapse_boilerplate and not self.config.collapse_imports:
            return lines, preserved

        new_lines: list[str] = []
        new_preserved: set[int] = set()

        i = 0
        while i < len(lines):
            line = lines[i]

            # --- Import run ---
            if self.config.collapse_imports and _IMPORT_RE.match(line) and i not in preserved:
                run = [i]
                j = i + 1
                while j < len(lines) and _IMPORT_RE.match(lines[j]) and j not in preserved:
                    run.append(j)
                    j += 1

                if len(run) > 3:
                    # Extract module names for the summary
                    names: list[str] = []
                    for idx in run:
                        m = re.match(r"^\s*(?:from\s+(\S+)|import\s+(\S+))", lines[idx])
                        if m:
                            names.append(m.group(1) or m.group(2))
                    summary = f"# [{len(run)} imports: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}]\n"
                    new_lines.append(summary)
                    i = j
                    continue
                else:
                    # Keep short import blocks verbatim
                    for idx in run:
                        out_i = len(new_lines)
                        if idx in preserved:
                            new_preserved.add(out_i)
                        new_lines.append(lines[idx])
                    i = j
                    continue

            # --- self.x = y assignment run ---
            if (
                self.config.collapse_boilerplate
                and _ASSIGNMENT_RE.match(line)
                and i not in preserved
            ):
                run = [i]
                j = i + 1
                while (
                    j < len(lines)
                    and _ASSIGNMENT_RE.match(lines[j])
                    and j not in preserved
                ):
                    run.append(j)
                    j += 1

                if len(run) > 2:
                    attr_names: list[str] = []
                    for idx in run:
                        m = re.match(r"^\s*self\.(\w+)\s*=", lines[idx])
                        if m:
                            attr_names.append(m.group(1))
                    summary = f"# [{len(run)} assignments: {', '.join(attr_names[:8])}{'...' if len(attr_names) > 8 else ''}]\n"
                    new_lines.append(summary)
                    i = j
                    continue
                else:
                    for idx in run:
                        out_i = len(new_lines)
                        if idx in preserved:
                            new_preserved.add(out_i)
                        new_lines.append(lines[idx])
                    i = j
                    continue

            # --- Logging/print run ---
            if (
                self.config.collapse_boilerplate
                and _LOG_RE.match(line)
                and i not in preserved
            ):
                count = 1
                j = i + 1
                while j < len(lines) and _LOG_RE.match(lines[j]) and j not in preserved:
                    count += 1
                    j += 1

                if count > 1:
                    new_lines.append(f"# [{count} log statements]\n")
                    i = j
                    continue
                # Single log line: fall through to normal handling

            # --- Consecutive blank lines: collapse to one ---
            if _BLANK_LINE_RE.match(line) and i not in preserved:
                out_i = len(new_lines)
                new_lines.append(line)
                j = i + 1
                while j < len(lines) and _BLANK_LINE_RE.match(lines[j]) and j not in preserved:
                    j += 1
                i = j
                continue

            # --- Default: keep line as-is ---
            out_i = len(new_lines)
            if i in preserved:
                new_preserved.add(out_i)
            new_lines.append(line)
            i += 1

        return new_lines, new_preserved

    # ------------------------------------------------------------------
    # Stage 3: TF-IDF importance filter
    # ------------------------------------------------------------------

    def _importance_filter(
        self,
        lines: list[str],
        preserved: set[int],
        context_idf: "dict[str, float] | None",
    ) -> tuple[list[str], set[int]]:
        """
        Remove the least-important non-preserved lines until the token target
        is reached.

        Importance score for a line = sum of IDF weights of its vocabulary
        terms.  Lines with only stopwords (low-IDF terms) are removed first.

        When *context_idf* is provided it overrides the global backend IDF,
        allowing query-aware pruning where terms rare in the query context are
        weighted more highly.

        Args:
            lines:       Current line list (after stage 2).
            preserved:   Set of line indices that must not be removed.
            context_idf: Per-context IDF override, or None.

        Returns:
            Updated ``(lines, preserved)``.
        """
        idf_map = context_idf if context_idf else (self.tfidf.idf if self.tfidf else {})
        if not idf_map:
            return lines, preserved

        total_tokens = sum(len(line.split()) for line in lines)
        target_tokens = int(total_tokens * self.config.target_ratio)

        if total_tokens <= target_tokens:
            return lines, preserved

        # Score each non-preserved line
        scored: list[tuple[int, float]] = []
        for i, line in enumerate(lines):
            if i in preserved:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            # Tokenise and sum IDF weights
            raw_terms = re.findall(r"[a-zA-Z_]\w{1,}", stripped.lower())
            score = sum(idf_map.get(t, 0.0) for t in raw_terms)
            scored.append((i, score))

        # Sort ascending by importance (lowest importance first = remove first)
        scored.sort(key=lambda x: x[1])

        # Cap: never remove more than max_prune_ratio of removable lines,
        # even if target_ratio requests more aggressive pruning.
        max_prune_ratio = getattr(self.config, "max_prune_ratio", 0.7)
        max_removable = int(len(scored) * max_prune_ratio)

        # Mark indices to remove until target is met
        to_remove: set[int] = set()
        removed_tokens = 0
        for idx, _ in scored:
            if len(to_remove) >= max_removable:
                break
            line_tokens = len(lines[idx].split())
            to_remove.add(idx)
            removed_tokens += line_tokens
            if (total_tokens - removed_tokens) <= target_tokens:
                break

        # Rebuild lines and preserved set
        new_lines: list[str] = []
        new_preserved: set[int] = set()
        for i, line in enumerate(lines):
            if i in to_remove:
                continue
            out_i = len(new_lines)
            if i in preserved:
                new_preserved.add(out_i)
            new_lines.append(line)

        return new_lines, new_preserved

    # ------------------------------------------------------------------
    # Stage 4: Density filter
    # ------------------------------------------------------------------

    def _density_filter(self, lines: list[str], preserved: set[int]) -> list[str]:
        """
        Remove near-duplicate lines and normalise multi-blank-line runs.

        A line is considered a near-duplicate of a previous line if their
        :class:`~difflib.SequenceMatcher` ratio exceeds 0.85, the earlier
        line is not in *preserved*, and both are non-blank.

        Args:
            lines:     Current line list (after stage 3).
            preserved: Set of 0-based indices that must survive.

        Returns:
            Filtered line list.
        """
        result: list[str] = []
        seen: list[str] = []          # lines added so far (stripped, for comparison)
        blank_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Blank line handling: allow at most one consecutive blank
            if not stripped:
                blank_count += 1
                if blank_count <= 1:
                    result.append(line)
                continue
            else:
                blank_count = 0

            # Preserved lines are always kept
            if i in preserved:
                result.append(line)
                seen.append(stripped)
                continue

            # Check for near-duplicates against recent lines (look back at most 20)
            is_duplicate = False
            for prev in seen[-20:]:
                if not prev:
                    continue
                ratio = SequenceMatcher(None, stripped, prev, autojunk=False).ratio()
                if ratio > 0.85:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append(line)
                seen.append(stripped)

        return result
