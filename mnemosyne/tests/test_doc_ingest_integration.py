# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integration tests for document ingestion pipeline.

Tests the full flow: file scan -> extractor -> chunker -> store -> retrieval.
"""

from __future__ import annotations

import os
import tempfile
import zipfile

import pytest

from mnemosyne.config import Config
from mnemosyne.schema import open_store
from mnemosyne.store import Store
from mnemosyne.bloom import BloomFilter
from mnemosyne.audit import AuditLog
from mnemosyne.embeddings import get_backend
from mnemosyne.ingest import Ingester


def _make_docx(path: str, paragraphs: list[str]) -> None:
    """Create a minimal valid DOCX file."""
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    parts = [f'<?xml version="1.0" encoding="UTF-8"?>']
    parts.append(f'<w:document xmlns:w="{ns}"><w:body>')
    for para in paragraphs:
        parts.append(f"<w:p><w:r><w:t>{para}</w:t></w:r></w:p>")
    parts.append("</w:body></w:document>")

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", "\n".join(parts))
        ct = '<?xml version="1.0" encoding="UTF-8"?>'
        ct += '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        ct += '<Default Extension="xml" ContentType="application/xml"/>'
        ct += "</Types>"
        zf.writestr("[Content_Types].xml", ct)


@pytest.fixture
def project(tmp_path):
    """Set up a project directory with mixed code + document files."""
    # Python source file
    py_file = tmp_path / "main.py"
    py_file.write_text("def hello():\n    return 'world'\n")

    # CSV file (enough rows for "good" quality threshold)
    csv_file = tmp_path / "data.csv"
    csv_lines = ["name,value,category,description"]
    for i in range(15):
        csv_lines.append(f"item_{i},{i*100},cat_{i%3},A detailed description for item number {i}")
    csv_file.write_text("\n".join(csv_lines) + "\n")

    # DOCX file
    docx_path = tmp_path / "report.docx"
    _make_docx(
        str(docx_path),
        [
            "Quarterly Financial Report for Q1 2026 Fiscal Year",
            "Revenue increased by 15% compared to Q4 2025 across all business segments.",
            "Key findings include significantly improved customer retention rates.",
            "The marketing budget allocation was optimized for maximum return on investment.",
            "Operating expenses were reduced through automation and efficiency improvements.",
        ],
    )

    # Log file
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "2026-04-06 10:00:01 INFO Server started on port 8080\n"
        "2026-04-06 10:00:02 INFO Connected to database\n"
        "2026-04-06 10:00:05 WARN Slow query detected: 2.3s\n"
        "2026-04-06 10:01:00 ERROR Connection timeout to redis\n"
        * 10
    )

    # INI config
    ini_file = tmp_path / "app.ini"
    ini_file.write_text(
        "[server]\nhost = 0.0.0.0\nport = 8080\n\n"
        "[database]\nurl = postgresql://localhost/mydb\npool_size = 10\n"
        * 10
    )

    return tmp_path


@pytest.fixture
def engine(project):
    """Set up the full ingestion engine with doc partition."""
    from mnemosyne.doc_store import DocStore

    config = Config(root=project)
    db_dir = project / ".mnemosyne"
    conn = open_store(db_dir)
    store = Store(conn)
    doc_store = DocStore(conn)
    bloom = BloomFilter()
    tfidf = get_backend(config, store)
    doc_tfidf = get_backend(config, store=None)
    audit = AuditLog(db_dir)
    ingester = Ingester(
        project, config, store, bloom, tfidf, audit,
        doc_store=doc_store, doc_tfidf=doc_tfidf,
    )
    return ingester, store, doc_store, config


class TestDocumentIngestion:
    """Test full document ingestion pipeline with partition isolation."""

    def test_ingest_mixed_project(self, engine):
        ingester, store, doc_store, config = engine
        stats = ingester.ingest()

        assert stats["files_scanned"] > 0
        assert stats["files_indexed"] > 0
        assert stats["files_failed"] == 0

    def test_ingest_stats_include_files_by_extension(self, engine):
        """Per-extension breakdown must be present + match files_scanned.

        Added so a dry-run / ingest-preview surface can show "what
        would be ingested by extension", not just total counts.
        Plumbed here so a single engine ingest call carries enough
        info; no caller has to reproduce the scan.
        """
        ingester, _store, _doc_store, _config = engine
        stats = ingester.ingest()

        assert "files_by_extension" in stats, "missing per-extension stat"
        by_ext = stats["files_by_extension"]
        assert isinstance(by_ext, dict)
        # Sum of per-extension counts must equal files_scanned (every
        # file in the resolved scan list contributes exactly one
        # extension count).
        assert sum(by_ext.values()) == stats["files_scanned"]
        # Test fixture writes .docx, .log, .ini files; at least those
        # should appear in the breakdown.
        for ext in (".docx", ".log", ".ini"):
            assert ext in by_ext, f"{ext} missing from files_by_extension {by_ext!r}"

    def test_ingest_dry_run_populates_files_by_extension(self, engine):
        """Dry-run path must produce the same extension breakdown as
        a real ingest -- because the per-extension stat is built off
        the resolved scan list, not the indexing loop's writes.
        """
        ingester, _store, _doc_store, _config = engine
        stats = ingester.ingest(dry_run=True)
        by_ext = stats["files_by_extension"]
        assert isinstance(by_ext, dict)
        assert sum(by_ext.values()) == stats["files_scanned"]
        # Dry-run still tallies every scanned file's extension.
        assert len(by_ext) > 0

    def test_csv_chunks_in_doc_partition(self, engine):
        ingester, store, doc_store, config = engine
        ingester.ingest()

        file_rec = store.get_file_record_by_path("data.csv")
        assert file_rec is not None
        assert file_rec.source_type == "document"
        assert file_rec.extraction_method == "direct"
        assert file_rec.extraction_quality == "good"

        # Chunks should be in doc partition, not code partition
        doc_chunks = doc_store.get_chunks_for_file(file_rec.file_id)
        assert len(doc_chunks) > 0
        full_text = " ".join(c.content for c in doc_chunks)
        assert "item_0" in full_text

        # Code partition should NOT have these chunks
        code_chunks = store.get_chunks_for_file(file_rec.file_id)
        assert len(code_chunks) == 0

    def test_docx_chunks_in_doc_partition(self, engine):
        ingester, store, doc_store, config = engine
        ingester.ingest()

        file_rec = store.get_file_record_by_path("report.docx")
        assert file_rec is not None
        assert file_rec.source_type == "document"

        doc_chunks = doc_store.get_chunks_for_file(file_rec.file_id)
        assert len(doc_chunks) > 0
        full_text = " ".join(c.content for c in doc_chunks)
        assert "Revenue" in full_text

    def test_log_file_in_doc_partition(self, engine):
        ingester, store, doc_store, config = engine
        ingester.ingest()

        file_rec = store.get_file_record_by_path("server.log")
        assert file_rec is not None
        assert file_rec.source_type == "document"

        doc_chunks = doc_store.get_chunks_for_file(file_rec.file_id)
        assert len(doc_chunks) > 0

    def test_python_still_works(self, engine):
        """Existing code indexing is unaffected."""
        ingester, store, doc_store, config = engine
        ingester.ingest()

        file_rec = store.get_file_record_by_path("main.py")
        assert file_rec is not None
        assert file_rec.language == "python"
        assert file_rec.source_type == "file"

        chunks = store.get_chunks_for_file(file_rec.file_id)
        assert len(chunks) > 0
        assert any("hello" in c.content for c in chunks)

    def test_incremental_reindex(self, engine, project):
        """Second ingest skips unchanged files."""
        ingester, store, doc_store, config = engine

        stats1 = ingester.ingest()
        stats2 = ingester.ingest()

        assert stats2["files_indexed"] == 0
        assert stats2["files_skipped"] == stats1["files_indexed"]

    def test_ini_file_in_doc_partition(self, engine):
        ingester, store, doc_store, config = engine
        ingester.ingest()

        file_rec = store.get_file_record_by_path("app.ini")
        assert file_rec is not None

        doc_chunks = doc_store.get_chunks_for_file(file_rec.file_id)
        assert len(doc_chunks) > 0
        full_text = " ".join(c.content for c in doc_chunks)
        assert "pool_size" in full_text


class TestSchemaIntegration:
    """Verify schema migration 2->3 works correctly."""

    def test_migration_adds_columns(self, project):
        """New columns exist after init_db."""
        config = Config(root=project)
        db_dir = project / ".mnemosyne"
        conn = open_store(db_dir)

        # Check files table has new columns
        cursor = conn.execute("PRAGMA table_info(files)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "extraction_method" in columns
        assert "extraction_quality" in columns
        assert "page_count" in columns
        assert "source_type" in columns

        # Check chunks table has page_number
        cursor = conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "page_number" in columns

        conn.close()
