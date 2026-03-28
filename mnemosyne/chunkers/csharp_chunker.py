# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
C#-aware structural chunker for Mnemosyne.

Extracts class, interface, struct, enum, namespace, and method declarations
from C# source.  ``using`` directive runs are captured as ``imports`` chunks.
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
# C# declaration patterns
# ---------------------------------------------------------------------------

_ACCESS = r"(?:public|private|protected|internal)"
_MODIFIERS = r"(?:static|async|virtual|override|abstract|sealed|partial|readonly|extern|new)"

# namespace Foo.Bar {
_NAMESPACE = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*namespace\s+(?P<name>[\w.]+)",
        re.M,
    ),
    chunk_type="block",
    name_group="name",
)

# class / struct / record declarations
# (public|...) (static|sealed|abstract|partial)* class Name
_CLASS = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:{_MODIFIERS}\s+)*"
        r"(?:class|struct|record)\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# interface Name
_INTERFACE = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?interface\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# enum Name {
_ENUM = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:{_MODIFIERS}\s+)*enum\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# Method-like declarations:
# (access)? (modifiers)* ReturnType Name(
# ReturnType can be void, string, int, bool, Task, Task<T>, List<T>, etc.
# We accept any identifier-like token (including dotted and generic) as the
# return type, followed by the method name and an opening paren.
_METHOD = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*(?:{_ACCESS}\s+)?(?:(?:{_MODIFIERS})\s+)*"
        r"(?:[\w.<>\[\],\s?]+?)\s+(?P<name>\w+)\s*\(",
        re.M,
    ),
    chunk_type="function",
    name_group="name",
)

# using directives run: using System; using System.Collections.Generic;
_USING = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>using)\s+[\w.]+\s*;",
        re.M,
    ),
    chunk_type="imports",
    name_group="name",
    has_block=False,
)


# Order matters: more specific patterns first so they win dedup at same offset.
_CSHARP_PATTERNS: list[DeclPattern] = [
    _NAMESPACE,
    _INTERFACE,
    _ENUM,
    _CLASS,
    _METHOD,
    _USING,
]

# Words that should never be captured as method names (they look like methods
# syntactically but are control-flow keywords).
_CS_KEYWORDS: frozenset[str] = frozenset({
    "if", "for", "foreach", "while", "switch", "catch", "return",
    "using", "namespace", "class", "struct", "interface", "enum",
    "new", "typeof", "sizeof", "default", "throw", "try", "finally",
    "lock", "fixed", "checked", "unchecked", "delegate", "event",
    "record", "get", "set", "add", "remove", "value",
})


# ---------------------------------------------------------------------------
# CSharpChunker
# ---------------------------------------------------------------------------


class CSharpChunker(BraceChunker):
    """C#-aware structural chunker.

    Produces :class:`ChunkCandidate` objects for classes, interfaces, enums,
    namespaces, methods, and using-directive blocks.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__(config, _CSHARP_PATTERNS)

    def chunk(self, source: str, language: str = "csharp") -> list[ChunkCandidate]:
        candidates = super().chunk(source, language)
        # Filter out false-positive method matches on keywords.
        return [
            c for c in candidates
            if not (
                c.chunk_type == "function"
                and c.symbol_name in _CS_KEYWORDS
            )
        ]
