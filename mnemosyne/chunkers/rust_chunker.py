# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Rust-aware structural chunker for Mnemosyne.

Extracts ``fn``, ``impl``, ``struct``, ``enum``, ``trait``, and ``mod``
declarations from Rust source.  ``use`` statement runs are captured as
``imports`` chunks.  Built on
:class:`~mnemosyne.chunkers.brace_chunker.BraceChunker`.

Handles Rust-specific syntax:
- Lifetime parameters (``'a``) are not confused with char literals.
- Raw strings (``r#"..."#``) do not break brace scanning (the base
  :class:`BraceDepthScanner` treats backtick and single-quote strings as
  terminated by the same delimiter; Rust raw strings use ``"`` so they are
  handled natively).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mnemosyne.chunkers.brace_chunker import BraceChunker, DeclPattern
from mnemosyne.chunkers.code_chunker import ChunkCandidate

if TYPE_CHECKING:
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Rust declaration patterns
# ---------------------------------------------------------------------------

_VIS = r"(?:pub(?:\s*\([^)]*\))?\s+)?"

# fn name(  /  pub fn name(  /  pub async fn name(  /  pub(crate) fn name(
_FN = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}(?:(?:async|unsafe|const|extern\s+\"C\")\s+)*fn\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="function",
    name_group="name",
)

# impl Name {  /  impl Trait for Name {  /  impl<T> Name<T> {
_IMPL = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}(?:unsafe\s+)?impl(?:<[^>]*>)?\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# struct Name {  /  pub struct Name<T> {
_STRUCT = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}struct\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# enum Name {
_ENUM = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}enum\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# trait Name {
_TRAIT = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}(?:unsafe\s+)?trait\s+(?P<name>\w+)",
        re.M,
    ),
    chunk_type="class",
    name_group="name",
)

# mod name {  (inline modules)
_MOD = DeclPattern(
    pattern=re.compile(
        rf"^[ \t]*{_VIS}mod\s+(?P<name>\w+)\s*\{{",
        re.M,
    ),
    chunk_type="block",
    name_group="name",
)

# use ... ;  (import runs) — statement-level, no block
_USE = DeclPattern(
    pattern=re.compile(
        r"^[ \t]*(?P<name>use)\s+[\w:]+",
        re.M,
    ),
    chunk_type="imports",
    name_group="name",
    has_block=False,
)


_RUST_PATTERNS: list[DeclPattern] = [
    _FN,
    _IMPL,
    _STRUCT,
    _ENUM,
    _TRAIT,
    _MOD,
    _USE,
]


# ---------------------------------------------------------------------------
# RustChunker
# ---------------------------------------------------------------------------


class RustChunker(BraceChunker):
    """Rust-aware structural chunker.

    Produces :class:`ChunkCandidate` objects for functions, impl blocks,
    structs, enums, traits, modules, and use-statement blocks.
    """

    def __init__(self, config: "Config") -> None:
        super().__init__(config, _RUST_PATTERNS)
