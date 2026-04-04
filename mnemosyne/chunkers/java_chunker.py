# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Java-aware structural chunker for Mnemosyne.

Extracts class, interface, enum, and method declarations from Java source.
Annotations (``@Override``, ``@Deprecated``, etc.) preceding a declaration are
included in the declaration's chunk.  ``import`` and ``package`` statements
are captured as ``imports`` and ``block`` chunks respectively.

Also handles Kotlin source (same brace-delimited structure; Kotlin-specific
``fun`` and ``object`` keywords are included).

Built on :class:`~mnemosyne.chunkers.brace_chunker.BraceChunker`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.chunkers.brace_chunker import BraceChunker, DeclPattern
from mnemosyne.chunkers.code_chunker import ChunkCandidate

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Java / Kotlin declaration patterns
# ---------------------------------------------------------------------------

_ACCESS = r"(?:public|private|protected)"
_MODIFIERS = r"(?:static|abstract|final|synchronized|native|transient|volatile|default|strictfp)"

# class Name  /  public abstract class Name<T>
_CLASS = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:{_MODIFIERS})\s+)*"
        r"class\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# interface Name  /  public interface Name<T>
_INTERFACE = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:{_MODIFIERS})\s+)*"
        r"interface\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# enum Name {
_ENUM = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:{_MODIFIERS})\s+)*"
        r"enum\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# Method declarations:
# (access)? (modifiers)* ReturnType name(
# ReturnType can be void, String, int, boolean, List<String>, etc.
_METHOD = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:{_MODIFIERS})\s+)*"
        r"(?:[\w.<>\[\],\s?]+?)\s+(?P<name>\w+)\s*\(",
        re.M,
    ),
    chunk_type="function",
    name_group="name",
)

# Kotlin: fun name(  /  private fun name(  /  suspend fun name(
_KOTLIN_FUN = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:suspend|inline|infix|operator|override|open|internal)\s+)*"
        r"fun\s+(?P<name>\w+)\s*[\(<]",
        re.M,
    ),
    chunk_type="function",
    name_group="name",
)

# Kotlin: object Name {
_KOTLIN_OBJECT = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:companion|data)\s+)?object\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# import statements: import java.util.List; -- statement-level, no block
_IMPORT = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>import)\s+(?:static\s+)?[\w.*]+\s*;",
        re.M,
    ),
    chunk_type="imports",
    name_group="name",
    has_block=False,
)

# package declaration: package com.example.foo; -- statement-level, no block
_PACKAGE = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>package)\s+[\w.]+\s*;",
        re.M,
    ),
    chunk_type="block",
    name_group="name",
    has_block=False,
)

# Annotation lines: @Override, @Deprecated, @SuppressWarnings("unchecked")
# These are consumed as part of the following declaration (see post-processing).
_ANNOTATION = re.compile(
    r"^[ \t]*@\w+(?:\s*\([^)]*\))?\s*$",
    re.M,
)


_JAVA_PATTERNS: list[DeclPattern] = [
    _CLASS,
    _INTERFACE,
    _ENUM,
    _KOTLIN_FUN,
    _KOTLIN_OBJECT,
    _METHOD,
    _IMPORT,
    _PACKAGE,
]

# Keywords that look like method declarations but are not.
_JAVA_KEYWORDS: frozenset[str] = frozenset({
    "if", "for", "while", "switch", "catch", "return",
    "import", "package", "class", "interface", "enum",
    "new", "throw", "try", "finally", "assert",
    "super", "this", "synchronized", "instanceof",
    "default", "extends", "implements", "throws",
    "fun", "object", "companion", "when",
})


# ---------------------------------------------------------------------------
# JavaChunker
# ---------------------------------------------------------------------------


class JavaChunker(BraceChunker):
    """Java/Kotlin-aware structural chunker.

    Produces :class:`ChunkCandidate` objects for classes, interfaces, enums,
    methods, import blocks, and package declarations.  Annotations preceding
    a declaration are folded into the declaration chunk.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__(config, _JAVA_PATTERNS)

    def chunk(self, source: str, language: str = "java") -> list[ChunkCandidate]:
        candidates = super().chunk(source, language)
        candidates = self._filter_keywords(candidates)
        candidates = self._fold_annotations(source, candidates)
        return candidates

    @staticmethod
    def _filter_keywords(
        candidates: list[ChunkCandidate],
    ) -> list[ChunkCandidate]:
        """Remove false-positive method matches on keywords."""
        return [
            c for c in candidates
            if not (
                c.chunk_type == "function"
                and c.symbol_name in _JAVA_KEYWORDS
            )
        ]

    @staticmethod
    def _fold_annotations(
        source: str, candidates: list[ChunkCandidate]
    ) -> list[ChunkCandidate]:
        """Extend declaration chunks upward to include preceding annotations.

        If the lines immediately before a function/class chunk are annotations,
        fold them into the chunk's content and adjust ``line_start``.
        """
        source_lines = source.splitlines(keepends=True)
        result: list[ChunkCandidate] = []

        for cand in candidates:
            if cand.chunk_type not in ("function", "class"):
                result.append(cand)
                continue

            # Walk backward from line_start to collect annotation lines.
            first_line = cand.line_start  # 1-based
            idx = first_line - 2  # 0-based index of the line before

            while idx >= 0:
                line = source_lines[idx]
                if _ANNOTATION.match(line):
                    idx -= 1
                else:
                    break

            new_start = idx + 2  # convert back to 1-based

            if new_start < first_line:
                prefix = "".join(source_lines[new_start - 1 : first_line - 1])
                cand = ChunkCandidate(
                    content=prefix + cand.content,
                    chunk_type=cand.chunk_type,
                    line_start=new_start,
                    line_end=cand.line_end,
                    symbol_name=cand.symbol_name,
                    parent_symbol=cand.parent_symbol,
                )

            result.append(cand)

        return result
