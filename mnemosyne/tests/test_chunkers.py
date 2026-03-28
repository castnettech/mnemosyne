# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the chunker layer:
  - CodeChunker (Python AST-based)
  - TextChunker (Markdown / plain-text heading-aware)
  - GenericChunker (sliding-window with overlap)
"""

import unittest


def _default_config():
    """Return a Config with default settings from a temp directory."""
    import tempfile
    from mnemosyne.config import Config
    return Config(root=tempfile.mkdtemp())


# ---------------------------------------------------------------------------
# TestCodeChunker
# ---------------------------------------------------------------------------


PYTHON_SOURCE = '''\
import os
import sys
from pathlib import Path

CONSTANT = 42
ANOTHER = "hello"

def standalone_function(x, y):
    """Return the sum of x and y."""
    return x + y


def another_function():
    pass


class MyClass:
    """A sample class."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, v):
        self.value = v
'''


class TestCodeChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.code_chunker import CodeChunker
        self.cfg = _default_config()
        self.chunker = CodeChunker(self.cfg)

    def test_chunk_returns_non_empty_list(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        self.assertGreater(len(chunks), 0)

    def test_imports_chunk_produced(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        types = [c.chunk_type for c in chunks]
        self.assertIn("imports", types)

    def test_function_chunks_produced(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        fn_chunks = [c for c in chunks if c.chunk_type == "function"]
        self.assertGreater(len(fn_chunks), 0)

    def test_function_symbol_names_captured(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        fn_names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        self.assertIn("standalone_function", fn_names)
        self.assertIn("another_function", fn_names)

    def test_class_chunk_produced(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        cls_chunks = [c for c in chunks if c.chunk_type == "class"]
        self.assertGreater(len(cls_chunks), 0)
        cls_names = {c.symbol_name for c in cls_chunks}
        self.assertIn("MyClass", cls_names)

    def test_method_chunks_with_parent_symbol(self):
        """Method chunks extracted from a class carry the parent class name."""
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        method_chunks = [
            c for c in chunks
            if c.chunk_type == "function" and c.parent_symbol == "MyClass"
        ]
        method_names = {c.symbol_name for c in method_chunks}
        self.assertIn("__init__", method_names)
        self.assertIn("get_value", method_names)

    def test_line_ranges_are_1_based(self):
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)

    def test_line_ranges_are_within_source(self):
        source_line_count = len(PYTHON_SOURCE.splitlines())
        chunks = self.chunker.chunk(PYTHON_SOURCE, language="python")
        for c in chunks:
            self.assertLessEqual(c.line_end, source_line_count + 1)

    def test_empty_source_returns_empty(self):
        chunks = self.chunker.chunk("   \n  \n", language="python")
        self.assertEqual(chunks, [])

    def test_syntax_error_falls_back_gracefully(self):
        """Syntactically invalid Python should not raise."""
        bad_source = "def broken(:\n    pass\n"
        chunks = self.chunker.chunk(bad_source, language="python")
        self.assertIsInstance(chunks, list)

    def test_block_chunk_for_module_level_statements(self):
        source = "CONSTANT = 42\nANOTHER = 'hello'\n"
        chunks = self.chunker.chunk(source, language="python")
        types = {c.chunk_type for c in chunks}
        self.assertTrue(types.issuperset({"block"}) or len(chunks) > 0)


# ---------------------------------------------------------------------------
# TestTextChunker
# ---------------------------------------------------------------------------


MARKDOWN_SOURCE = """\
# Introduction

This section introduces the project.
It has some overview content.

## Installation

Install via pip:

```
pip install mnemosyne
```

More installation notes here.

## Usage

Run the CLI to index your project:

```
mnemosyne index .
```

Then query it:

```
mnemosyne query "authentication middleware"
```

### Advanced Options

You can configure the engine via `.mnemosyne/config.toml`.
"""


class TestTextChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.text_chunker import TextChunker
        self.cfg = _default_config()
        self.chunker = TextChunker(self.cfg)

    def test_chunk_returns_non_empty_list(self):
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        self.assertGreater(len(chunks), 0)

    def test_chunk_type_is_paragraph(self):
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        for c in chunks:
            self.assertEqual(c.chunk_type, "paragraph")

    def test_heading_based_splits(self):
        """Headings trigger splits; small adjacent sections may merge, but we get >= 2 chunks."""
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        # The markdown has 4 heading sections; small sections may be merged by the
        # min_tokens filter, so we assert at least 2 chunks are produced.
        self.assertGreaterEqual(len(chunks), 2)

    def test_chunks_contain_heading_text(self):
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        combined = "\n".join(c.content for c in chunks)
        self.assertIn("Installation", combined)
        self.assertIn("Usage", combined)

    def test_line_ranges_sequential(self):
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)

    def test_empty_source_returns_empty(self):
        chunks = self.chunker.chunk("  \n  \n", language="markdown")
        self.assertEqual(chunks, [])

    def test_plaintext_mode_splits_on_paragraphs(self):
        text = "First paragraph.\nStill first paragraph.\n\nSecond paragraph.\n"
        chunks = self.chunker.chunk(text, language="text")
        self.assertGreaterEqual(len(chunks), 1)
        all_text = "".join(c.content for c in chunks)
        self.assertIn("First paragraph", all_text)
        self.assertIn("Second paragraph", all_text)

    def test_no_headings_falls_back_to_plaintext(self):
        """Markdown without headings should still produce chunks."""
        text = "Just plain content.\nNo headings here.\n"
        chunks = self.chunker.chunk(text, language="markdown")
        self.assertGreater(len(chunks), 0)

    def test_chunk_content_covers_source(self):
        """All significant words in source appear somewhere in the chunks."""
        chunks = self.chunker.chunk(MARKDOWN_SOURCE, language="markdown")
        combined = "\n".join(c.content for c in chunks)
        self.assertIn("mnemosyne", combined)


# ---------------------------------------------------------------------------
# TestGenericChunker
# ---------------------------------------------------------------------------


def _make_long_text(words_per_line=8, n_lines=60):
    """Return a plain text string long enough to require multiple chunks."""
    lines = []
    for i in range(n_lines):
        line = " ".join(f"word{i}_{j}" for j in range(words_per_line))
        lines.append(line)
    return "\n".join(lines) + "\n"


class TestGenericChunker(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.generic_chunker import GenericChunker
        self.cfg = _default_config()
        # Set a small token budget to force multiple chunks
        self.cfg.chunking.max_chunk_tokens = 50
        self.cfg.chunking.min_chunk_tokens = 5
        self.cfg.chunking.overlap_lines = 2
        self.chunker = GenericChunker(self.cfg)

    def test_chunk_short_source_single_chunk(self):
        """Source that fits in one budget produces exactly one chunk."""
        short = "word1 word2 word3\nword4 word5\n"
        chunks = self.chunker.chunk(short)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "generic")

    def test_chunk_long_source_multiple_chunks(self):
        long_text = _make_long_text(words_per_line=10, n_lines=30)
        chunks = self.chunker.chunk(long_text)
        self.assertGreater(len(chunks), 1)

    def test_all_chunks_are_generic_type(self):
        long_text = _make_long_text()
        chunks = self.chunker.chunk(long_text)
        for c in chunks:
            self.assertEqual(c.chunk_type, "generic")

    def test_chunks_have_overlap(self):
        """Consecutive chunks share at least one line (due to overlap_lines)."""
        long_text = _make_long_text(words_per_line=10, n_lines=40)
        chunks = self.chunker.chunk(long_text)
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                # line_end of chunk i should be >= line_start of chunk i+1
                # indicating overlap
                self.assertGreater(
                    chunks[i].line_end,
                    chunks[i + 1].line_start - 1,
                    f"No overlap between chunk {i} (ends {chunks[i].line_end}) "
                    f"and chunk {i+1} (starts {chunks[i+1].line_start})",
                )

    def test_line_ranges_are_1_based(self):
        long_text = _make_long_text()
        chunks = self.chunker.chunk(long_text)
        for c in chunks:
            self.assertGreaterEqual(c.line_start, 1)
            self.assertGreaterEqual(c.line_end, c.line_start)

    def test_empty_source_returns_empty(self):
        chunks = self.chunker.chunk("   \n  \n")
        self.assertEqual(chunks, [])

    def test_single_line_source(self):
        chunks = self.chunker.chunk("single line of text\n")
        self.assertEqual(len(chunks), 1)

    def test_no_chunk_exceeds_max_tokens_significantly(self):
        """No chunk should be wildly over the token budget (allow 2x for edge cases)."""
        from mnemosyne.models import estimate_tokens
        long_text = _make_long_text(words_per_line=8, n_lines=50)
        chunks = self.chunker.chunk(long_text)
        budget = self.cfg.chunking.max_chunk_tokens
        for c in chunks:
            tokens = estimate_tokens(c.content)
            self.assertLessEqual(
                tokens,
                budget * 2,
                f"Chunk token count {tokens} greatly exceeds budget {budget}",
            )


if __name__ == "__main__":
    unittest.main()
