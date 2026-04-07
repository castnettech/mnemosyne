# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the document extractor modules."""

from __future__ import annotations

import os
import tempfile
import textwrap
import zipfile

import pytest


# ---------------------------------------------------------------------------
# DOCX extractor
# ---------------------------------------------------------------------------


class TestDocxExtractor:
    """Tests for extractors.docx_extractor."""

    def _make_docx(self, paragraphs: list[str], tmp_dir: str, meta: dict | None = None) -> str:
        """Create a minimal valid DOCX file with the given paragraphs."""
        path = os.path.join(tmp_dir, "test.docx")
        with zipfile.ZipFile(path, "w") as zf:
            # Build word/document.xml
            ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            parts = [f'<?xml version="1.0" encoding="UTF-8"?>']
            parts.append(f'<w:document xmlns:w="{ns}"><w:body>')
            for para in paragraphs:
                parts.append(f"<w:p><w:r><w:t>{para}</w:t></w:r></w:p>")
            parts.append("</w:body></w:document>")
            zf.writestr("word/document.xml", "\n".join(parts))

            # Build [Content_Types].xml (required for valid DOCX)
            ct = '<?xml version="1.0" encoding="UTF-8"?>'
            ct += '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            ct += '<Default Extension="xml" ContentType="application/xml"/>'
            ct += "</Types>"
            zf.writestr("[Content_Types].xml", ct)

            # Optional metadata
            if meta:
                dc_ns = "http://purl.org/dc/elements/1.1/"
                cp_ns = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                core = f'<?xml version="1.0" encoding="UTF-8"?>'
                core += f'<cp:coreProperties xmlns:cp="{cp_ns}" xmlns:dc="{dc_ns}">'
                for key, val in meta.items():
                    core += f"<dc:{key}>{val}</dc:{key}>"
                core += "</cp:coreProperties>"
                zf.writestr("docProps/core.xml", core)

        return path

    def test_extract_basic(self):
        from mnemosyne.extractors import docx_extractor

        with tempfile.TemporaryDirectory() as tmp:
            # Generate enough text to exceed the 200-char "good" threshold
            paragraphs = [
                "Quarterly financial report for Q1 2026 fiscal year.",
                "Revenue increased by 15% compared to the previous quarter.",
                "Operating expenses were reduced through efficiency gains.",
                "Customer retention improved significantly across all segments.",
                "The marketing budget allocation was optimized for ROI.",
            ]
            path = self._make_docx(paragraphs, tmp)
            result = docx_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "good"
            assert result.extraction_method == "direct"
            assert len(result.pages) == 1
            assert "Quarterly" in result.pages[0].text
            assert "Revenue" in result.pages[0].text

    def test_extract_with_metadata(self):
        from mnemosyne.extractors import docx_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_docx(
                ["Content here"],
                tmp,
                meta={"title": "Test Doc", "creator": "Unit Test"},
            )
            result = docx_extractor.extract(path)
            assert result is not None
            assert result.metadata.get("title") == "Test Doc"
            assert result.metadata.get("author") == "Unit Test"

    def test_extract_empty_docx(self):
        from mnemosyne.extractors import docx_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = self._make_docx([], tmp)
            result = docx_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "failed"

    def test_extract_not_a_zip(self):
        from mnemosyne.extractors import docx_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.docx")
            with open(path, "w") as f:
                f.write("This is not a ZIP file")
            result = docx_extractor.extract(path)
            # Should return failed or None, not crash
            assert result is None or result.extraction_quality == "failed"

    def test_extract_zip_without_document_xml(self):
        from mnemosyne.extractors import docx_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notdocx.docx")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("random.txt", "just a zip file")
            result = docx_extractor.extract(path)
            assert result is None

    def test_supported_extensions(self):
        from mnemosyne.extractors import docx_extractor
        assert ".docx" in docx_extractor.supported_extensions()


# ---------------------------------------------------------------------------
# CSV extractor
# ---------------------------------------------------------------------------


class TestCsvExtractor:
    """Tests for extractors.csv_extractor."""

    def test_extract_csv_with_header(self):
        from mnemosyne.extractors import csv_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            with open(path, "w") as f:
                f.write("name,age,city,department,notes\n")
                for i in range(10):
                    f.write(f"Person{i},{20+i},City{i},Dept{i},Some detailed notes here\n")

            result = csv_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "good"
            assert "name: Person0" in result.pages[0].text
            assert "age: 20" in result.pages[0].text
            assert result.metadata["row_count"] == "10"
            assert result.metadata["column_count"] == "5"

    def test_extract_tsv(self):
        from mnemosyne.extractors import csv_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.tsv")
            with open(path, "w") as f:
                f.write("col1\tcol2\n")
                f.write("val1\tval2\n")

            result = csv_extractor.extract(path)
            assert result is not None
            # TSV sniffer may or may not detect header; just verify content
            assert "val1" in result.pages[0].text
            assert "val2" in result.pages[0].text

    def test_extract_empty_csv(self):
        from mnemosyne.extractors import csv_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.csv")
            with open(path, "w") as f:
                f.write("col1,col2\n")

            result = csv_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "failed"

    def test_supported_extensions(self):
        from mnemosyne.extractors import csv_extractor
        exts = csv_extractor.supported_extensions()
        assert ".csv" in exts
        assert ".tsv" in exts


# ---------------------------------------------------------------------------
# Plaintext extractor
# ---------------------------------------------------------------------------


class TestPlaintextExtractor:
    """Tests for extractors.plaintext_extractor."""

    def test_extract_log_file(self):
        from mnemosyne.extractors import plaintext_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "app.log")
            with open(path, "w") as f:
                f.write("2026-04-06 INFO Starting server\n" * 50)

            result = plaintext_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "good"
            assert "Starting server" in result.pages[0].text

    def test_extract_ini_file(self):
        from mnemosyne.extractors import plaintext_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.ini")
            with open(path, "w") as f:
                f.write("[database]\nhost = localhost\nport = 5432\n" * 20)

            result = plaintext_extractor.extract(path)
            assert result is not None
            assert "host = localhost" in result.pages[0].text

    def test_extract_empty_file(self):
        from mnemosyne.extractors import plaintext_extractor

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.log")
            with open(path, "w") as f:
                f.write("")

            result = plaintext_extractor.extract(path)
            assert result is not None
            assert result.extraction_quality == "failed"

    def test_supported_extensions(self):
        from mnemosyne.extractors import plaintext_extractor
        exts = plaintext_extractor.supported_extensions()
        assert ".log" in exts
        assert ".cfg" in exts
        assert ".ini" in exts


# ---------------------------------------------------------------------------
# Base module helpers
# ---------------------------------------------------------------------------


class TestBaseHelpers:
    """Tests for extractors.base utility functions."""

    def test_classify_quality_good(self):
        from mnemosyne.extractors.base import classify_quality
        assert classify_quality(90.0, 500) == "good"

    def test_classify_quality_poor_low_confidence(self):
        from mnemosyne.extractors.base import classify_quality
        assert classify_quality(55.0, 500) == "poor"

    def test_classify_quality_poor_short_text(self):
        from mnemosyne.extractors.base import classify_quality
        assert classify_quality(90.0, 50) == "poor"

    def test_classify_quality_failed(self):
        from mnemosyne.extractors.base import classify_quality
        assert classify_quality(30.0, 500) == "failed"

    def test_extracted_content_full_text(self):
        from mnemosyne.extractors.base import ExtractedContent, ExtractedPage
        content = ExtractedContent(pages=[
            ExtractedPage(page_number=1, text="Page one"),
            ExtractedPage(page_number=2, text="Page two"),
            ExtractedPage(page_number=3, text=""),  # empty page
        ])
        assert content.full_text == "Page one\n\nPage two"

    def test_extracted_content_mean_confidence(self):
        from mnemosyne.extractors.base import ExtractedContent, ExtractedPage
        content = ExtractedContent(pages=[
            ExtractedPage(page_number=1, text="A", confidence=80.0),
            ExtractedPage(page_number=2, text="B", confidence=60.0),
        ])
        assert content.mean_confidence == 70.0


# ---------------------------------------------------------------------------
# Extractor dispatcher
# ---------------------------------------------------------------------------


class TestExtractorDispatch:
    """Tests for extractors.__init__ dispatcher."""

    def test_get_extractor_docx(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("test.docx")
        assert ext is not None

    def test_get_extractor_csv(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("data.csv")
        assert ext is not None

    def test_get_extractor_tsv(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("data.tsv")
        assert ext is not None

    def test_get_extractor_log(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("server.log")
        assert ext is not None

    def test_get_extractor_unknown(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("binary.exe")
        assert ext is None

    def test_get_extractor_python_not_document(self):
        from mnemosyne.extractors import get_extractor
        ext = get_extractor("main.py")
        assert ext is None

    def test_supported_extensions_includes_pdf(self):
        from mnemosyne.extractors import supported_extensions
        exts = supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".csv" in exts
