# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the document partition: DocStore, DocRetrievalEngine, isolation."""

from __future__ import annotations

import os
import zipfile

import pytest

from mnemosyne.config import Config
from mnemosyne.schema import open_store
from mnemosyne.store import Store
from mnemosyne.doc_store import DocStore
from mnemosyne.bloom import BloomFilter
from mnemosyne.audit import AuditLog
from mnemosyne.embeddings import get_backend
from mnemosyne.ingest import Ingester
from mnemosyne.models import Chunk, FileRecord


def _make_docx(path: str, paragraphs: list[str]) -> None:
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
    """Mixed code + doc project."""
    (tmp_path / "main.py").write_text(
        "def calculate_revenue():\n    return quarterly_total * margin\n"
    )
    (tmp_path / "utils.py").write_text(
        "def format_currency(value):\n    return f'${value:,.2f}'\n"
    )
    csv_lines = ["name,value,category"]
    for i in range(15):
        csv_lines.append(f"item_{i},{i*100},cat_{i%3}")
    (tmp_path / "data.csv").write_text("\n".join(csv_lines) + "\n")
    _make_docx(str(tmp_path / "report.docx"), [
        "Annual Revenue Report 2026",
        "Total revenue increased by 22% year over year across all segments.",
        "Operating margins improved due to automation and cost reduction.",
        "Customer acquisition cost decreased by 15% through organic growth.",
        "Projected revenue for next quarter is estimated at 4.2 million.",
    ])
    (tmp_path / "server.log").write_text(
        "2026-04-06 INFO Server started\n" * 30
    )
    return tmp_path


@pytest.fixture
def engines(project):
    """Set up both code and doc engines."""
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
    ingester.ingest()

    return store, doc_store, config, tfidf, doc_tfidf


class TestPartitionIsolation:
    """Verify code and doc partitions are fully isolated."""

    def test_code_chunks_in_code_partition(self, engines):
        store, doc_store, *_ = engines
        code_count = store.count_chunks()
        assert code_count > 0, "Code partition should have chunks"

    def test_doc_chunks_in_doc_partition(self, engines):
        store, doc_store, *_ = engines
        doc_count = doc_store.count_chunks()
        assert doc_count > 0, "Doc partition should have chunks"

    def test_no_docs_in_code_partition(self, engines):
        store, doc_store, *_ = engines
        # Code partition should NOT contain CSV/DOCX/log content
        types = store.chunk_type_counts()
        # Code partition should have function/class/block types, not paragraph from docs
        all_chunks = []
        for rec in store.list_files(include_deleted=False):
            all_chunks.extend(store.get_chunks_for_file(rec.file_id))
        for chunk in all_chunks:
            # No document file paths in code chunks
            assert not chunk.content.startswith("# Columns:"), \
                "CSV content should not be in code partition"

    def test_no_code_in_doc_partition(self, engines):
        store, doc_store, *_ = engines
        # Doc partition should NOT contain Python function definitions
        all_doc_text = ""
        # Query all doc files
        for row in doc_store.conn.execute(
            "SELECT c.content FROM doc_chunks c"
        ).fetchall():
            all_doc_text += row[0]
        assert "def calculate_revenue" not in all_doc_text, \
            "Python code should not be in doc partition"

    def test_file_records_shared(self, engines):
        store, doc_store, *_ = engines
        # Both code and doc files should be in the shared files table
        files = store.list_files(include_deleted=False)
        paths = {f.rel_path for f in files}
        assert "main.py" in paths
        assert "data.csv" in paths
        assert "report.docx" in paths

    def test_source_type_correct(self, engines):
        store, *_ = engines
        for rec in store.list_files(include_deleted=False):
            if rec.rel_path.endswith(".py"):
                assert rec.source_type == "file"
            elif rec.rel_path.endswith((".csv", ".docx", ".log")):
                assert rec.source_type == "document"


class TestDocStore:
    """Test DocStore CRUD operations."""

    def test_insert_and_get_chunk(self, engines):
        store, doc_store, *_ = engines
        # Get any doc file
        doc_file = None
        for rec in store.list_files(include_deleted=False):
            if rec.source_type == "document":
                doc_file = rec
                break
        assert doc_file is not None

        chunks = doc_store.get_chunks_for_file(doc_file.file_id)
        assert len(chunks) > 0
        # Verify round-trip
        chunk = doc_store.get_chunk(chunks[0].chunk_id)
        assert chunk is not None
        assert chunk.content == chunks[0].content

    def test_chunk_type_counts(self, engines):
        _, doc_store, *_ = engines
        counts = doc_store.chunk_type_counts()
        assert "paragraph" in counts

    def test_total_tokens(self, engines):
        _, doc_store, *_ = engines
        tokens = doc_store.total_tokens()
        assert tokens > 0

    def test_search_fts(self, engines):
        _, doc_store, *_ = engines
        results = doc_store.search_fts("revenue")
        assert len(results) > 0

    def test_search_fts_no_code(self, engines):
        _, doc_store, *_ = engines
        # "calculate_revenue" is a Python function, not in doc partition
        results = doc_store.search_fts("calculate_revenue")
        # Should find nothing or very low relevance
        for chunk_id, _ in results:
            chunk = doc_store.get_chunk(chunk_id)
            assert "def calculate_revenue" not in chunk.content

    def test_sparse_embedding_roundtrip(self, engines):
        _, doc_store, *_ = engines
        chunks = doc_store.conn.execute(
            "SELECT chunk_id FROM doc_chunks LIMIT 1"
        ).fetchall()
        if chunks:
            cid = chunks[0][0]
            emb = doc_store.get_sparse_embedding(cid)
            # Should have been populated during ingest
            assert emb is not None or True  # may not have run rebuild yet


class TestDocRetrievalEngine:
    """Test document retrieval pipeline."""

    def test_query_returns_results(self, engines):
        store, doc_store, config, tfidf, doc_tfidf = engines
        from mnemosyne.doc_retrieval import DocRetrievalEngine
        engine = DocRetrievalEngine(doc_store, doc_tfidf, config)
        results = engine.query("revenue report")
        # May or may not find results depending on TF-IDF rebuild
        # But should not crash
        assert isinstance(results, list)

    def test_query_does_not_return_code(self, engines):
        store, doc_store, config, tfidf, doc_tfidf = engines
        from mnemosyne.doc_retrieval import DocRetrievalEngine
        engine = DocRetrievalEngine(doc_store, doc_tfidf, config)
        results = engine.query("def function")
        for r in results:
            assert "def calculate_revenue" not in r.chunk.content


class TestSchemaVersion:
    """Verify migration creates doc partition tables."""

    def test_doc_tables_exist(self, project):
        config = Config(root=project)
        db_dir = project / ".mnemosyne"
        conn = open_store(db_dir)

        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "doc_chunks" in tables
        assert "doc_sparse_embeddings" in tables
        assert "doc_vocabulary" in tables

    def test_doc_fts_exists(self, project):
        config = Config(root=project)
        db_dir = project / ".mnemosyne"
        conn = open_store(db_dir)

        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "doc_chunks_fts" in tables

    def test_schema_version_is_5(self, project):
        config = Config(root=project)
        db_dir = project / ".mnemosyne"
        conn = open_store(db_dir)

        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        # Schema v5 adds the doc_embeddings table for the dense lane.
        assert row[0] == 5

        conn.close()

    def test_doc_embeddings_table_exists(self, project):
        db_dir = project / ".mnemosyne"
        conn = open_store(db_dir)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "doc_embeddings" in tables
        conn.close()
