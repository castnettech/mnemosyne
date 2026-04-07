# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Chunker registry and dispatch for Mnemosyne.

Maps file extensions to language tags and selects the appropriate chunker
implementation. Python source gets AST-based chunking; JS/TS gets regex+brace
chunking; Go/C#/Rust/Java/Kotlin get their own brace-based chunkers;
Markdown/prose gets heading-aware text chunking; everything else gets the
sliding-window fallback.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mnemosyne.chunkers.code_chunker import CodeChunker
from mnemosyne.chunkers.text_chunker import TextChunker
from mnemosyne.chunkers.generic_chunker import GenericChunker
from mnemosyne.chunkers.js_chunker import JSChunker
from mnemosyne.chunkers.go_chunker import GoChunker
from mnemosyne.chunkers.csharp_chunker import CSharpChunker
from mnemosyne.chunkers.rust_chunker import RustChunker
from mnemosyne.chunkers.java_chunker import JavaChunker
from mnemosyne.chunkers.schema_chunker import SchemaChunker
from mnemosyne.chunkers.document_chunker import DocumentChunker

if TYPE_CHECKING:
    from mnemosyne.config import Config

# ---------------------------------------------------------------------------
# Extension -> language mapping
# ---------------------------------------------------------------------------

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".cs": "csharp",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
    ".sh": "shell",
    ".html": "html",
    ".css": "css",
    # Document formats -- routed through extractors, chunked by DocumentChunker
    ".pdf": "document",
    ".docx": "document",
    ".csv": "document",
    ".tsv": "document",
    # Additional plaintext formats
    ".log": "text",
    ".cfg": "text",
    ".ini": "text",
    ".conf": "text",
    ".rst": "text",
    ".xml": "markup",
    ".svg": "markup",
}

# Languages that the TextChunker handles well
_TEXT_LANGUAGES: frozenset[str] = frozenset({"markdown", "text"})

# Languages that get AST/structural CodeChunker treatment
_CODE_LANGUAGES: frozenset[str] = frozenset({"python"})

# Languages that get JS-aware structural chunking
_JS_LANGUAGES: frozenset[str] = frozenset({"javascript", "typescript"})

# Languages that get DDL-aware structural chunking
_SCHEMA_LANGUAGES: frozenset[str] = frozenset({"sql", "sql_schema"})

# Document formats -- routed through extractor + DocumentChunker
_DOCUMENT_LANGUAGES: frozenset[str] = frozenset({"document"})

# Languages that get brace-based structural chunking
_GO_LANGUAGES: frozenset[str] = frozenset({"go"})
_CSHARP_LANGUAGES: frozenset[str] = frozenset({"csharp"})
_RUST_LANGUAGES: frozenset[str] = frozenset({"rust"})
_JAVA_LANGUAGES: frozenset[str] = frozenset({"java", "kotlin"})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_language(file_path: str) -> str:
    """
    Detect language from the file extension of *file_path*.

    Returns the language tag (e.g. ``"python"``) or ``"unknown"`` when the
    extension is not in :data:`LANGUAGE_MAP`.

    Args:
        file_path: Any string path or filename -- only the extension is used.

    Returns:
        A lowercase language tag string.
    """
    _, ext = os.path.splitext(file_path)
    return LANGUAGE_MAP.get(ext.lower(), "unknown")


def get_chunker(
    language: str,
    config: "Config",
) -> CodeChunker | TextChunker | GenericChunker | JSChunker | GoChunker | CSharpChunker | RustChunker | JavaChunker | SchemaChunker:
    """
    Return an appropriately configured chunker for *language*.

    Dispatch rules (checked in order):
    - ``"python"``                -> :class:`CodeChunker` (AST-based)
    - ``"javascript"``/``"typescript"`` -> :class:`JSChunker` (regex+brace)
    - ``"go"``                    -> :class:`GoChunker` (regex+brace)
    - ``"csharp"``                -> :class:`CSharpChunker` (regex+brace)
    - ``"rust"``                  -> :class:`RustChunker` (regex+brace)
    - ``"java"``/``"kotlin"``     -> :class:`JavaChunker` (regex+brace)
    - ``"sql"``/``"sql_schema"``  -> :class:`SchemaChunker` (DDL-aware)
    - ``"markdown"``/``"text"``   -> :class:`TextChunker` (heading/paragraph-aware)
    - everything else             -> :class:`GenericChunker` (sliding-window fallback)

    Args:
        language: Language tag as returned by :func:`detect_language`.
        config:   Mnemosyne :class:`~mnemosyne.config.Config` instance.

    Returns:
        A chunker instance ready to call ``.chunk(source, language)``.
    """
    if language in _DOCUMENT_LANGUAGES:
        return DocumentChunker(config)
    if language in _SCHEMA_LANGUAGES:
        return SchemaChunker(config)
    if language in _CODE_LANGUAGES:
        return CodeChunker(config)
    if language in _JS_LANGUAGES:
        return JSChunker(config)
    if language in _GO_LANGUAGES:
        return GoChunker(config)
    if language in _CSHARP_LANGUAGES:
        return CSharpChunker(config)
    if language in _RUST_LANGUAGES:
        return RustChunker(config)
    if language in _JAVA_LANGUAGES:
        return JavaChunker(config)
    if language in _TEXT_LANGUAGES:
        return TextChunker(config)
    return GenericChunker(config)


__all__ = [
    "LANGUAGE_MAP",
    "CodeChunker",
    "JSChunker",
    "GoChunker",
    "CSharpChunker",
    "RustChunker",
    "JavaChunker",
    "SchemaChunker",
    "DocumentChunker",
    "TextChunker",
    "GenericChunker",
    "detect_language",
    "get_chunker",
]
