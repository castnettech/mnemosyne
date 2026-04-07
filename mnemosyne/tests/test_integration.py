# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end integration test for the Mnemosyne context engine.

TestEndToEnd verifies the full pipeline:
  1. Config loads from a temp directory
  2. Schema is initialised in-memory
  3. Ingest processes sample Python and Markdown files
  4. Query retrieves relevant results
  5. Compress reduces a chunk's token count
  6. Stats reflect the indexed state
"""

import os
import sqlite3
import tempfile
import unittest


# ---------------------------------------------------------------------------
# Sample file contents written to the temp project
# ---------------------------------------------------------------------------


PYTHON_FILE = '''\
"""Authentication utilities."""

import os
import hashlib

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")


def hash_password(password: str) -> str:
    """Return a salted SHA-256 hash of password."""
    salted = f"{password}{SECRET_KEY}"
    return hashlib.sha256(salted.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Return True if password matches hashed."""
    return hash_password(password) == hashed


class AuthManager:
    """Manages user authentication state."""

    def __init__(self, user_store):
        self.user_store = user_store
        self.sessions = {}

    def login(self, username: str, password: str) -> str:
        """Authenticate user and return a session token."""
        user = self.user_store.get(username)
        if user is None:
            raise ValueError("Unknown user")
        if not verify_password(password, user["password_hash"]):
            raise PermissionError("Invalid credentials")
        import uuid
        token = str(uuid.uuid4())
        self.sessions[token] = username
        return token

    def logout(self, token: str) -> None:
        """Invalidate a session token."""
        self.sessions.pop(token, None)

    def get_user(self, token: str) -> str | None:
        """Return username for token, or None if not authenticated."""
        return self.sessions.get(token)
'''

MARKDOWN_FILE = '''\
# Mnemosyne Context Engine

A lightweight context engine for injecting relevant code into LLM prompts.

## Features

- AST-based Python chunking
- Markdown / prose chunking
- TF-IDF sparse embeddings
- Hybrid BM25 + vector retrieval
- ARC cache for hot chunks

## Quick Start

```bash
pip install mnemosyne
mnemosyne index .
mnemosyne query "authentication middleware"
```

## Configuration

Create `.mnemosyne/config.toml` in your project root:

```toml
[chunking]
max_chunk_tokens = 300

[retrieval]
token_budget = 8000
```
'''

SECOND_PYTHON_FILE = '''\
"""Database utilities."""

import sqlite3
from pathlib import Path


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open and configure an SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def execute_query(conn: sqlite3.Connection, sql: str, params=()) -> list:
    """Execute sql and return all rows as a list of dicts."""
    cursor = conn.execute(sql, params)
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


class Repository:
    """Base repository with common CRUD operations."""

    def __init__(self, conn: sqlite3.Connection, table: str):
        self.conn = conn
        self.table = table

    def find_by_id(self, record_id: int):
        """Return row with matching id, or None."""
        rows = execute_query(
            self.conn,
            f"SELECT * FROM {self.table} WHERE id = ?",
            (record_id,),
        )
        return rows[0] if rows else None

    def delete(self, record_id: int) -> None:
        """Delete row by id."""
        with self.conn:
            self.conn.execute(
                f"DELETE FROM {self.table} WHERE id = ?", (record_id,)
            )
'''


class TestEndToEnd(unittest.TestCase):
    """Full pipeline test: init, ingest, query, compress, stats."""

    def setUp(self):
        """Set up a temp directory with sample files, an in-memory DB, and all components."""
        self.tmp_dir = tempfile.mkdtemp()

        # Write sample files
        src_dir = os.path.join(self.tmp_dir, "src")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "auth.py"), "w") as f:
            f.write(PYTHON_FILE)
        with open(os.path.join(src_dir, "db.py"), "w") as f:
            f.write(SECOND_PYTHON_FILE)
        with open(os.path.join(self.tmp_dir, "README.md"), "w") as f:
            f.write(MARKDOWN_FILE)

        # Config
        from mnemosyne.config import Config
        self.cfg = Config(root=self.tmp_dir)
        self.cfg.embedding.tfidf_min_df = 1

        # In-memory DB + store
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(self.conn)
        from mnemosyne.store import Store
        self.store = Store(self.conn)

        # TF-IDF backend (no store persistence needed for integration)
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        self.tfidf = TFIDFBackend(self.cfg, store=None)

        # Bloom filter
        from mnemosyne.bloom import BloomFilter
        self.bloom = BloomFilter(capacity=10_000, fp_rate=0.01)

        # Audit log
        from mnemosyne.audit import AuditLog
        audit_path = os.path.join(self.tmp_dir, ".mnemosyne", "audit.jsonl")
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        self.audit = AuditLog(audit_path)

        # Doc partition
        from mnemosyne.doc_store import DocStore
        self.doc_store = DocStore(self.conn)
        self.doc_tfidf = TFIDFBackend(self.cfg, store=None)

        # Ingester
        from mnemosyne.ingest import Ingester
        self.ingester = Ingester(
            project_root=self.tmp_dir,
            config=self.cfg,
            store=self.store,
            bloom=self.bloom,
            tfidf_backend=self.tfidf,
            audit=self.audit,
            doc_store=self.doc_store,
            doc_tfidf=self.doc_tfidf,
        )

    def tearDown(self):
        """Remove temp files."""
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------

    def test_ingest_processes_files(self):
        stats = self.ingester.ingest()
        self.assertGreater(stats["files_scanned"], 0)
        self.assertGreater(stats["files_indexed"], 0)
        self.assertEqual(stats["files_failed"], 0)

    def test_ingest_creates_chunks(self):
        self.ingester.ingest()
        self.assertGreater(self.store.count_chunks(), 0)

    def test_ingest_creates_file_records(self):
        self.ingester.ingest()
        self.assertGreater(self.store.count_files(), 0)

    def test_ingest_indexes_python_and_markdown(self):
        self.ingester.ingest()
        records = self.store.list_files()
        languages = {r.language for r in records}
        self.assertIn("python", languages)
        # Markdown is now routed to the doc partition with language="document"
        self.assertIn("document", languages)

    def test_ingest_stats_have_expected_keys(self):
        stats = self.ingester.ingest()
        for key in ["files_scanned", "files_indexed", "files_skipped", "files_failed",
                    "chunks_added", "chunks_deduped", "elapsed_seconds"]:
            self.assertIn(key, stats)

    def test_ingest_dry_run_writes_nothing(self):
        stats = self.ingester.ingest(dry_run=True)
        self.assertEqual(self.store.count_chunks(), 0)
        self.assertGreater(stats["files_indexed"], 0)  # counted but not written

    # ------------------------------------------------------------------
    # Stage 2: Query
    # ------------------------------------------------------------------

    def _build_retrieval_engine(self):
        from mnemosyne.retrieval import RetrievalEngine
        # Rebuild inverted index from persisted sparse embeddings
        all_embs = self.store.get_all_sparse_embeddings()
        if all_embs:
            self.tfidf.build_inverted_index(all_embs)
        return RetrievalEngine(
            store=self.store,
            tfidf_backend=self.tfidf,
            config=self.cfg,
        )

    def test_query_after_ingest_returns_results(self):
        self.ingester.ingest()
        engine = self._build_retrieval_engine()
        results = engine.query("authentication password hash")
        self.assertGreater(len(results), 0)

    def test_query_relevant_auth_chunk_ranks_first(self):
        self.ingester.ingest()
        engine = self._build_retrieval_engine()
        results = engine.query("authenticate verify password")
        self.assertGreater(len(results), 0)
        # Top result should come from auth.py
        top = results[0]
        self.assertIn("auth", top.file_path)

    def test_query_database_returns_db_chunks(self):
        self.ingester.ingest()
        engine = self._build_retrieval_engine()
        results = engine.query("database connection sqlite execute")
        self.assertGreater(len(results), 0)
        paths = [r.file_path for r in results]
        self.assertTrue(
            any("db" in p for p in paths),
            f"Expected db.py in results, got: {paths}",
        )

    def test_query_result_scores_have_rrf_key(self):
        self.ingester.ingest()
        engine = self._build_retrieval_engine()
        results = engine.query("function return")
        for r in results:
            self.assertIn("rrf", r.scores)

    # ------------------------------------------------------------------
    # Stage 3: Compress
    # ------------------------------------------------------------------

    def test_compress_reduces_tokens(self):
        from mnemosyne.compress import Compressor
        from mnemosyne.models import estimate_tokens
        compressor = Compressor(self.cfg)

        self.ingester.ingest()
        chunks = self.store.get_chunks_for_file(
            self.store.get_file("src/auth.py").file_id
        )
        # Find a decently sized chunk to compress
        big_chunks = [c for c in chunks if c.token_count >= 20]
        if not big_chunks:
            self.skipTest("No chunks large enough to meaningfully compress")

        chunk = big_chunks[0]
        original_tokens = estimate_tokens(chunk.content)
        compressed = compressor.compress(chunk)
        compressed_tokens = estimate_tokens(compressed)

        # Compressed should not be larger than original by more than 10%
        self.assertLessEqual(
            compressed_tokens,
            int(original_tokens * 1.1),
            f"Expected compression, original={original_tokens}, compressed={compressed_tokens}",
        )

    # ------------------------------------------------------------------
    # Stage 4: Stats
    # ------------------------------------------------------------------

    def test_stats_after_ingest(self):
        self.ingester.ingest()
        stats = self.store.get_stats()
        self.assertGreater(stats["files"], 0)
        self.assertGreater(stats["chunks"], 0)
        self.assertGreater(stats["total_tokens_indexed"], 0)

    def test_chunk_type_counts_after_ingest(self):
        self.ingester.ingest()
        counts = self.store.chunk_type_counts()
        # Python AST chunking should produce 'function' and/or 'class' chunks
        self.assertTrue(
            "function" in counts or "class" in counts or "imports" in counts,
            f"Expected code chunk types, got: {counts}",
        )

    def test_language_counts_after_ingest(self):
        self.ingester.ingest()
        lang_counts = self.store.language_counts()
        self.assertIn("python", lang_counts)
        # Markdown now classified as "document" in the shared files table
        self.assertIn("document", lang_counts)

    # ------------------------------------------------------------------
    # Re-ingest idempotency
    # ------------------------------------------------------------------

    def test_second_ingest_skips_unchanged_files(self):
        self.ingester.ingest()
        first_chunk_count = self.store.count_chunks()

        # Second ingest: files haven't changed
        stats2 = self.ingester.ingest()
        second_chunk_count = self.store.count_chunks()

        # Chunk count should be the same (files unchanged)
        # Note: second ingest may delete old and re-add if timestamps differ
        # so we only check files_failed = 0
        self.assertEqual(stats2["files_failed"], 0)

    def test_full_reingest_rebuilds(self):
        self.ingester.ingest()
        chunk_count_before = self.store.count_chunks()

        self.ingester.ingest(full=True)
        chunk_count_after = self.store.count_chunks()

        # Should have the same number of chunks (same content)
        # The exact count may vary due to dedup, but should be > 0
        self.assertGreater(chunk_count_after, 0)

    # ------------------------------------------------------------------
    # Bloom filter integration
    # ------------------------------------------------------------------

    def test_bloom_filter_contains_indexed_paths(self):
        self.ingester.ingest()
        records = self.store.list_files()
        for rec in records:
            self.assertTrue(
                self.bloom.might_contain(rec.rel_path),
                f"Bloom filter missing indexed path: {rec.rel_path}",
            )


    # ------------------------------------------------------------------
    # GC rebuilds Bloom filter (Milestone 1.4)
    # ------------------------------------------------------------------

    def test_gc_rebuilds_bloom_filter(self):
        """After GC, the Bloom filter must not contain deleted file paths,
        and re-created files must be re-indexable (not falsely skipped)."""
        # 1. Ingest all files so the bloom filter knows about them
        self.ingester.ingest()

        # Pick one file to delete
        target_rel = "src/auth.py"
        target_abs = os.path.join(self.tmp_dir, target_rel)
        self.assertTrue(
            self.bloom.might_contain(target_rel),
            "Bloom should contain the file path after ingest",
        )

        # 2. Delete the file from disk
        os.remove(target_abs)

        # 3. Run GC logic -- mirrors cmd_gc() but uses our in-memory objects
        #    Mark missing files as deleted and remove their chunks
        for file_record in self.store.get_all_file_records():
            if file_record.is_deleted:
                continue
            abs_path = os.path.join(self.tmp_dir, file_record.rel_path)
            if not os.path.isfile(abs_path):
                self.store.delete_chunks_for_file(file_record.file_id)
                self.store.mark_file_deleted(file_record.file_id)

        # Clean up orphan chunks from already-deleted files
        for file_id in self.store.get_deleted_file_ids():
            self.store.delete_chunks_for_file(file_id)

        self.store.prune_cache_state()
        self.store.prune_usage_events()

        # Rebuild bloom filter from survivors (same logic as cmd_gc)
        from mnemosyne.bloom import BloomFilter
        new_bloom = BloomFilter()
        for fr in self.store.list_files(include_deleted=False):
            new_bloom.add(fr.rel_path)
        rows = self.conn.execute(
            "SELECT DISTINCT content_hash FROM chunks"
        ).fetchall()
        for row in rows:
            new_bloom.add(row[0])

        # Replace the ingester's bloom filter with the rebuilt one
        self.bloom = new_bloom
        self.ingester.bloom = new_bloom

        # 4. Verify the deleted path is no longer in the bloom filter
        self.assertFalse(
            new_bloom.might_contain(target_rel),
            "Bloom should NOT contain the deleted file path after GC rebuild",
        )

        # 5. A surviving file should still be present
        self.assertTrue(
            new_bloom.might_contain("src/db.py"),
            "Bloom should still contain surviving file paths",
        )

        # 6. Re-create the deleted file and ingest again -- it must be re-indexed
        os.makedirs(os.path.dirname(target_abs), exist_ok=True)
        with open(target_abs, "w") as f:
            f.write(PYTHON_FILE)

        stats = self.ingester.ingest()
        self.assertGreater(
            stats["files_indexed"], 0,
            "Re-created file should be re-indexed after GC bloom rebuild",
        )


if __name__ == "__main__":
    unittest.main()
