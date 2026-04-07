# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the DocumentChunker."""

from __future__ import annotations

import pytest

from mnemosyne.config import Config
from mnemosyne.chunkers.document_chunker import DocumentChunker
from mnemosyne.extractors.base import ExtractedContent, ExtractedPage


@pytest.fixture
def config(tmp_path):
    return Config(root=tmp_path)


@pytest.fixture
def chunker(config):
    return DocumentChunker(config)


class TestDocumentChunkerString:
    """Test chunk(source, language) -- plain string interface."""

    def test_chunk_plain_text(self, chunker):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        candidates = chunker.chunk(text)
        assert len(candidates) >= 1
        for cand in candidates:
            assert cand.page_number == 0
            assert cand.chunk_type == "paragraph"

    def test_chunk_empty_string(self, chunker):
        candidates = chunker.chunk("")
        assert candidates == []


class TestDocumentChunkerExtracted:
    """Test chunk_extracted(content) -- ExtractedContent interface."""

    def test_single_page(self, chunker):
        content = ExtractedContent(
            pages=[ExtractedPage(page_number=1, text="Hello world. This is a test document.")],
            page_count=1,
        )
        candidates = chunker.chunk_extracted(content)
        assert len(candidates) >= 1
        assert candidates[0].page_number == 1

    def test_multi_page(self, chunker):
        content = ExtractedContent(
            pages=[
                ExtractedPage(page_number=1, text="Page one content here."),
                ExtractedPage(page_number=2, text="Page two content here."),
                ExtractedPage(page_number=3, text="Page three content here."),
            ],
            page_count=3,
        )
        candidates = chunker.chunk_extracted(content)
        assert len(candidates) >= 3
        page_numbers = [c.page_number for c in candidates]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers

    def test_empty_page_skipped(self, chunker):
        content = ExtractedContent(
            pages=[
                ExtractedPage(page_number=1, text="Actual content here."),
                ExtractedPage(page_number=2, text="   "),  # whitespace only
                ExtractedPage(page_number=3, text="More content here."),
            ],
            page_count=3,
        )
        candidates = chunker.chunk_extracted(content)
        page_numbers = [c.page_number for c in candidates]
        assert 2 not in page_numbers

    def test_line_offsets_increase_across_pages(self, chunker):
        content = ExtractedContent(
            pages=[
                ExtractedPage(page_number=1, text="Line 1\nLine 2\nLine 3"),
                ExtractedPage(page_number=2, text="Line 4\nLine 5\nLine 6"),
            ],
            page_count=2,
        )
        candidates = chunker.chunk_extracted(content)
        if len(candidates) >= 2:
            # Page 2 chunks should have higher line numbers than page 1
            page1_max = max(c.line_end for c in candidates if c.page_number == 1)
            page2_min = min(c.line_start for c in candidates if c.page_number == 2)
            assert page2_min > page1_max

    def test_no_pages(self, chunker):
        content = ExtractedContent(pages=[], page_count=0)
        candidates = chunker.chunk_extracted(content)
        assert candidates == []

    def test_large_page_split(self, chunker):
        """A page with lots of text should be split into multiple chunks."""
        # Generate text that exceeds max_chunk_tokens (default 300)
        paragraphs = [f"Paragraph {i}. " + "word " * 50 for i in range(20)]
        big_text = "\n\n".join(paragraphs)

        content = ExtractedContent(
            pages=[ExtractedPage(page_number=1, text=big_text)],
            page_count=1,
        )
        candidates = chunker.chunk_extracted(content)
        assert len(candidates) > 1
        for cand in candidates:
            assert cand.page_number == 1
