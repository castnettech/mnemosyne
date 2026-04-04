# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Go-aware structural chunker for Mnemosyne.

Extracts function declarations (including methods with receivers), struct and
interface type declarations, and ``const``/``var`` grouping blocks from Go
source.  Built on top of :class:`~mnemosyne.chunkers.brace_chunker.BraceChunker`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.chunkers.brace_chunker import BraceChunker, DeclPattern
from mnemosyne.chunkers.code_chunker import ChunkCandidate

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Go declaration patterns
# ---------------------------------------------------------------------------

# Method with receiver: func (r *Type) Name(
# The symbol name is "Type.Name".
_METHOD_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*func\s+\(\s*\w+\s+\*?(?P<recv>\w+)\s*\)\s+(?P<name>\w+)\s*\(",
        re.M,
    ),
    chunk_type="function",
    name_group="name",  # post-processed to "Recv.Name" below
)

# Plain function: func Name(
_FUNC_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*func\s+(?P<name>\w+)\s*\(",
        re.M,
    ),
    chunk_type="function",
    name_group="name",
)

# type Name struct {
_STRUCT_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*type\s+(?P<name>\w+)\s+struct\b",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# type Name interface {
_INTERFACE_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*type\s+(?P<name>\w+)\s+interface\b",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# const ( ... ) or var ( ... ) grouped blocks -- paren-delimited, not brace
_CONST_VAR_BLOCK = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>(?:const|var))\s*\(",
        re.M,
    ),
    chunk_type="imports",
    name_group="name",
    has_block=False,
)

# import ( ... ) or import "..." -- paren-delimited or single-line
_IMPORT_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>import)\s*(?:\(|\")",
        re.M,
    ),
    chunk_type="imports",
    name_group="name",
    has_block=False,
)

# package declaration -- single line, no block
_PACKAGE_PATTERN = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>package)\s+\w+",
        re.M,
    ),
    chunk_type="block",
    name_group="name",
    has_block=False,
)


# Ordered list -- method pattern must precede plain function so that its
# regex match wins at the same offset when a receiver is present.
_GO_PATTERNS: list[DeclPattern] = [
    _METHOD_PATTERN,
    _FUNC_PATTERN,
    _STRUCT_PATTERN,
    _INTERFACE_PATTERN,
    _CONST_VAR_BLOCK,
    _IMPORT_PATTERN,
    _PACKAGE_PATTERN,
]


# ---------------------------------------------------------------------------
# GoChunker
# ---------------------------------------------------------------------------


class GoChunker(BraceChunker):
    """Go-aware structural chunker.

    Produces :class:`ChunkCandidate` objects with ``symbol_name`` set to the
    declaration name.  For methods with a receiver the symbol is formatted as
    ``ReceiverType.MethodName``.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__(config, _GO_PATTERNS)
        # Pre-compile a pattern used to qualify method names.
        self._method_re = _METHOD_PATTERN.pattern

    # Override chunk to post-process receiver-qualified method names.
    def chunk(self, source: str, language: str = "go") -> list[ChunkCandidate]:
        candidates = super().chunk(source, language)
        return self._qualify_methods(source, candidates)

    def _qualify_methods(
        self, source: str, candidates: list[ChunkCandidate]
    ) -> list[ChunkCandidate]:
        """Rewrite ``symbol_name`` for method declarations to include the
        receiver type: ``ReceiverType.MethodName``."""
        # Build a lookup: match_start -> recv_name
        recv_map: dict[int, str] = {}
        for m in self._method_re.finditer(source):
            recv_map[m.start()] = m.group("recv")

        if not recv_map:
            return candidates

        result: list[ChunkCandidate] = []
        for cand in candidates:
            if (
                cand.chunk_type == "function"
                and cand.symbol_name
                and "." not in cand.symbol_name
            ):
                # Find if this candidate's content starts at a method offset.
                # Use the line_start to compute approximate char offset.
                lines_before = source.split("\n")[: cand.line_start - 1]
                char_offset = sum(len(ln) + 1 for ln in lines_before)
                # Look for a receiver match near this offset.
                for start, recv in recv_map.items():
                    if abs(start - char_offset) <= 2:
                        cand = ChunkCandidate(
                            content=cand.content,
                            chunk_type=cand.chunk_type,
                            line_start=cand.line_start,
                            line_end=cand.line_end,
                            symbol_name=f"{recv}.{cand.symbol_name}",
                            parent_symbol=cand.parent_symbol,
                        )
                        break
            result.append(cand)

        return result
