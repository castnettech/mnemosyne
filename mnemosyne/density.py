# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Information density analysis utilities for Mnemosyne.

These functions analyse source-text fragments for:
- **Line entropy**: approximate information density of a single line.
- **Boilerplate detection**: identify runs of lines matching common low-value
  patterns (imports, logging, assignments, blank lines, getters/setters).
- **Near-duplicate detection**: find pairs of lines with high string similarity.
- **Repetition scoring**: quantify how repetitive a block of lines is overall.

All functions are pure (no side effects) and operate on plain strings.
No external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Low-entropy keywords: structural but carry little unique information on their own
_LOW_ENTROPY_KEYWORDS: frozenset[str] = frozenset(
    {
        "if", "else", "elif", "for", "while", "try", "except", "finally",
        "with", "pass", "break", "continue", "return", "yield", "raise",
        "and", "or", "not", "in", "is", "as", "from", "import", "class",
        "def", "lambda", "global", "nonlocal", "del", "assert", "True",
        "False", "None",
    }
)


# ---------------------------------------------------------------------------
# Line entropy
# ---------------------------------------------------------------------------


def compute_line_entropy(line: str) -> float:
    """
    Approximate the information density (entropy) of a single source line.

    The score is a float in ``[0.0, 1.0]`` where:
    - ``0.0`` = empty or whitespace-only
    - Low (~0.1 - 0.3) = lines composed mostly of structural keywords
    - High (~0.7 - 1.0) = lines with unique identifiers, literals, operators

    Algorithm:
    1. Extract tokens (alphanumeric + underscores).
    2. Score = fraction of tokens that are NOT in the low-entropy keyword set,
       weighted by Shannon entropy of the character distribution.

    Args:
        line: A single source line (may include trailing newline).

    Returns:
        Float density score in ``[0.0, 1.0]``.
    """
    stripped = line.strip()
    if not stripped:
        return 0.0

    # Extract word tokens
    tokens = re.findall(r"[a-zA-Z_]\w+", stripped)
    if not tokens:
        # Pure punctuation / operators -- moderately informative
        return 0.3

    # Fraction of tokens that are not low-entropy structural keywords
    non_keyword_count = sum(1 for t in tokens if t.lower() not in _LOW_ENTROPY_KEYWORDS)
    keyword_ratio = non_keyword_count / len(tokens)

    # Shannon entropy of character distribution (normalised by max possible)
    char_counts = Counter(stripped)
    total_chars = len(stripped)
    char_entropy = 0.0
    for count in char_counts.values():
        p = count / total_chars
        char_entropy -= p * math.log2(p)

    # Normalise character entropy by log2(len(unique chars)) -- max possible
    unique_chars = len(char_counts)
    max_char_entropy = math.log2(unique_chars) if unique_chars > 1 else 1.0
    normalised_char_entropy = char_entropy / max_char_entropy

    # Combine: keyword ratio (70%) + normalised char entropy (30%)
    score = 0.7 * keyword_ratio + 0.3 * normalised_char_entropy
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Boilerplate detection
# ---------------------------------------------------------------------------

# Pattern definitions: (compiled regex, human-readable pattern name)
_BOILERPLATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(import|from)\s+\S+"), "import"),
    (re.compile(r"^\s*(logger\.|logging\.|log\.|print\()", re.IGNORECASE), "logging"),
    (re.compile(r"^\s*self\.\w+\s*=\s*\S+"), "self_assignment"),
    (re.compile(r"^\s*$"), "blank"),
    (re.compile(r"^\s*#(?!\s*(TODO|FIXME|HACK|NOTE|XXX))", re.IGNORECASE), "comment"),
    (re.compile(r"^\s*@\w+"), "decorator"),
    (re.compile(r"^\s*(pass|\.\.\.)\s*$"), "placeholder"),
    (re.compile(r"^\s*raise\s+NotImplementedError"), "not_implemented"),
]


def detect_boilerplate_patterns(
    lines: list[str],
) -> list[tuple[int, int, str]]:
    """
    Find runs of consecutive lines matching the same boilerplate pattern.

    Only runs of length >= 2 are reported (single boilerplate lines are not
    flagged as patterns worth collapsing).

    Args:
        lines: Source lines (with or without trailing newlines).

    Returns:
        List of ``(start_idx, end_idx, pattern_name)`` tuples where
        ``start_idx`` and ``end_idx`` are 0-based inclusive indices into
        *lines*.  Tuples are sorted by ``start_idx``.
    """
    results: list[tuple[int, int, str]] = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        matched_pattern: str | None = None

        for pattern, name in _BOILERPLATE_PATTERNS:
            if pattern.match(line):
                matched_pattern = name
                break

        if matched_pattern is None:
            i += 1
            continue

        # Extend the run as far as the same pattern matches consecutive lines
        run_start = i
        j = i + 1
        while j < n:
            for pattern, name in _BOILERPLATE_PATTERNS:
                if name == matched_pattern and pattern.match(lines[j]):
                    j += 1
                    break
            else:
                break

        run_end = j - 1  # inclusive
        if run_end > run_start:  # run length >= 2
            results.append((run_start, run_end, matched_pattern))

        i = j

    return results


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def find_near_duplicates(
    lines: list[str],
    threshold: float = 0.85,
) -> list[tuple[int, int]]:
    """
    Find pairs of lines that are near-duplicates of each other.

    Uses :class:`~difflib.SequenceMatcher` to compare every pair.  The
    comparison is O(n²) so this function should only be called on bounded
    windows (e.g. chunks of up to ~200 lines).

    Blank lines are excluded from comparison.

    Args:
        lines:     List of source lines.
        threshold: Similarity ratio above which two lines are considered
                   near-duplicates.  Default ``0.85``.

    Returns:
        List of ``(i, j)`` index pairs (0-based, ``i < j``) where
        ``lines[i]`` and ``lines[j]`` are near-duplicates.
    """
    # Normalise: strip whitespace for comparison but keep original indices
    stripped = [line.strip() for line in lines]
    results: list[tuple[int, int]] = []
    n = len(stripped)

    for i in range(n):
        if not stripped[i]:
            continue
        for j in range(i + 1, n):
            if not stripped[j]:
                continue
            ratio = SequenceMatcher(
                None, stripped[i], stripped[j], autojunk=False
            ).ratio()
            if ratio >= threshold:
                results.append((i, j))

    return results


# ---------------------------------------------------------------------------
# Repetition score
# ---------------------------------------------------------------------------


def repetition_score(lines: list[str]) -> float:
    """
    Score how repetitive a block of lines is.

    Returns a float in ``[0.0, 1.0]`` where:
    - ``0.0`` = every line is unique
    - ``1.0`` = every line is identical

    Algorithm:
    1. Strip whitespace from each non-blank line.
    2. Count duplicates using a :class:`~collections.Counter`.
    3. Score = 1 - (unique_lines / total_non_blank_lines).

    The score is high when many lines are identical or near-identical to
    others in the block, signalling boilerplate or machine-generated code.

    Args:
        lines: Source lines of the block.

    Returns:
        Float repetition score in ``[0.0, 1.0]``.
    """
    non_blank = [line.strip() for line in lines if line.strip()]
    if not non_blank:
        return 0.0

    total = len(non_blank)
    unique = len(set(non_blank))

    # unique / total ranges from 1/total (all same) to 1.0 (all unique)
    # repetition = 1 - (unique / total)
    return 1.0 - (unique / total)
