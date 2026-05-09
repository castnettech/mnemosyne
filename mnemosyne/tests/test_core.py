# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the foundation layer:
  - Config (config.py)
  - Models / estimate_tokens (models.py)
  - Hasher: content_hash, file_hash, is_binary (hasher.py)
  - BloomFilter: add, might_contain, save/load (bloom.py)
  - Schema: init_db creates all expected tables, FTS5 works (schema.py)
"""

import os
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    """Config loads defaults and provides dot-access."""

    def _make_config(self, root=None):
        from mnemosyne.config import Config
        return Config(root=root)

    def test_defaults_load_without_toml(self):
        """Config with no config.toml returns defaults unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            self.assertEqual(cfg.chunking.max_chunk_tokens, 300)
            self.assertEqual(cfg.chunking.min_chunk_tokens, 20)
            self.assertEqual(cfg.chunking.overlap_lines, 3)
            self.assertEqual(cfg.embedding.backend, "tfidf")
            self.assertEqual(cfg.retrieval.max_results, 20)

    def test_dot_access_general_section(self):
        """config.general.max_file_size_kb is accessible."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            self.assertEqual(cfg.general.max_file_size_kb, 1024)
            self.assertIsInstance(cfg.general.supported_extensions, list)
            self.assertIn(".py", cfg.general.supported_extensions)

    def test_default_extensions_cover_brace_family_languages(self):
        """Regression guard: supported_extensions must expose every
        language the chunker dispatcher already handles structurally.
        Historically the defaults were Python/Node-flavoured and
        silently filtered .cs / .rs / .go / .java / .kt / .c / .cpp
        out of every ingest even though dedicated chunkers existed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            exts = cfg.general.supported_extensions
            # Dedicated structural chunkers exist for these.
            for required in (
                ".cs", ".rs", ".go", ".java", ".kt",
                ".mjs", ".cjs",
            ):
                self.assertIn(
                    required,
                    exts,
                    msg=f"{required} missing from default supported_extensions",
                )
            # LANGUAGE_MAP-tagged (chunked by GenericChunker today, but
            # still strictly better than silent exclusion).
            for tagged in (".c", ".h", ".cpp", ".hpp", ".svg"):
                self.assertIn(
                    tagged,
                    exts,
                    msg=f"{tagged} missing from default supported_extensions",
                )

    def test_default_ignore_patterns_cover_build_outputs(self):
        """Regression guard: ignore_patterns must hide common build-output
        directories and runtime artifacts from indexing. Without these,
        a .NET project's `bin/Release/*.deps.json` (NuGet dependency
        manifest) ends up in the doc store with high-IDF tokens that
        dominate BM25 scoring on unrelated queries -- the precise
        failure mode that triggered this hardening.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            patterns = cfg.general.ignore_patterns
            # .NET noise that triggered the rework.
            for required in ("bin", "obj", "*.deps.json",
                             "*.runtimeconfig.json"):
                self.assertIn(
                    required,
                    patterns,
                    msg=f"{required} missing from default ignore_patterns",
                )
            # Generic build dirs across many languages.
            for required in ("node_modules", "__pycache__", "target",
                             "build", "dist", "out", "vendor",
                             "cmake-build-*", "htmlcov", "*.egg-info"):
                self.assertIn(
                    required,
                    patterns,
                    msg=f"{required} missing from default ignore_patterns",
                )
            # Compiled artifacts that should not be indexed even when
            # checked in.
            for required in ("*.class", "*.jar", "*.dll", "*.exe",
                             "*.pdb", "*.so", "*.dylib",
                             "*.tsbuildinfo"):
                self.assertIn(
                    required,
                    patterns,
                    msg=f"{required} missing from default ignore_patterns",
                )

    def test_user_config_supported_extensions_unions_with_defaults(self):
        """A user config.toml that sets supported_extensions must not
        drop .cs et al. from the effective list -- the deep-merge is
        list-union, so hardened language coverage is preserved even
        when a project overrides the list with its own additions.
        """
        with tempfile.TemporaryDirectory() as tmp:
            mnemo_dir = Path(tmp) / ".mnemosyne"
            mnemo_dir.mkdir(parents=True, exist_ok=True)
            (mnemo_dir / "config.toml").write_text(
                '[general]\nsupported_extensions = [".xyz"]\n',
                encoding="utf-8",
            )
            cfg = self._make_config(root=tmp)
            exts = cfg.general.supported_extensions
            self.assertIn(".xyz", exts, "user addition must land")
            self.assertIn(".cs", exts, "default .cs must survive user override")
            self.assertIn(".py", exts, "default .py must survive user override")

    def test_get_method(self):
        """config.get(section, key) returns the correct value."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            self.assertEqual(cfg.get("chunking", "max_chunk_tokens"), 300)
            self.assertIsNone(cfg.get("chunking", "nonexistent_key"))
            self.assertEqual(cfg.get("chunking", "nonexistent_key", "fallback"), "fallback")

    def test_merge_with_toml(self):
        """Values in config.toml override defaults."""
        with tempfile.TemporaryDirectory() as tmp:
            dot_dir = os.path.join(tmp, ".mnemosyne")
            os.makedirs(dot_dir)
            toml_path = os.path.join(dot_dir, "config.toml")
            with open(toml_path, "w") as fh:
                fh.write("[chunking]\nmax_chunk_tokens = 999\n")
            cfg = self._make_config(root=tmp)
            self.assertEqual(cfg.chunking.max_chunk_tokens, 999)
            # Non-overridden defaults remain intact
            self.assertEqual(cfg.chunking.min_chunk_tokens, 20)

    def test_toml_does_not_affect_unrelated_sections(self):
        """A TOML that only touches chunking leaves retrieval defaults alone."""
        with tempfile.TemporaryDirectory() as tmp:
            dot_dir = os.path.join(tmp, ".mnemosyne")
            os.makedirs(dot_dir)
            with open(os.path.join(dot_dir, "config.toml"), "w") as fh:
                fh.write("[chunking]\noverlap_lines = 5\n")
            cfg = self._make_config(root=tmp)
            self.assertEqual(cfg.chunking.overlap_lines, 5)
            self.assertEqual(cfg.retrieval.max_results, 20)

    def test_set_method_creates_new_key(self):
        """config.set() can add new keys to existing sections."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            cfg.set("chunking", "custom_key", 42)
            self.assertEqual(cfg.get("chunking", "custom_key"), 42)

    def test_as_dict_returns_nested_dict(self):
        """config.as_dict() returns all sections as a nested plain dict."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            d = cfg.as_dict()
            self.assertIn("chunking", d)
            self.assertIsInstance(d["chunking"], dict)
            self.assertIn("max_chunk_tokens", d["chunking"])

    def test_save_and_reload(self):
        """Saved config can be reloaded and values are preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(root=tmp)
            cfg.chunking.max_chunk_tokens = 512
            cfg.save()
            cfg2 = self._make_config(root=tmp)
            self.assertEqual(cfg2.chunking.max_chunk_tokens, 512)


# ---------------------------------------------------------------------------
# TestModels
# ---------------------------------------------------------------------------


class TestModels(unittest.TestCase):
    """Domain model dataclass creation and estimate_tokens."""

    def test_file_record_creation(self):
        from mnemosyne.models import FileRecord
        rec = FileRecord(
            file_id=None,
            rel_path="src/main.py",
            content_hash="abc123",
            size_bytes=1024,
            language="python",
            last_modified=1700000000.0,
        )
        self.assertIsNone(rec.file_id)
        self.assertEqual(rec.rel_path, "src/main.py")
        self.assertFalse(rec.is_deleted)
        self.assertIsNone(rec.last_indexed)

    def test_chunk_creation(self):
        from mnemosyne.models import Chunk
        chunk = Chunk(
            chunk_id=None,
            file_id=1,
            content_hash="deadbeef",
            chunk_type="function",
            line_start=10,
            line_end=25,
            token_count=50,
            content="def foo():\n    pass\n",
        )
        self.assertEqual(chunk.chunk_type, "function")
        self.assertIsNone(chunk.compressed)
        self.assertIsNone(chunk.symbol_name)

    def test_usage_event_creation(self):
        from mnemosyne.models import UsageEvent
        event = UsageEvent(
            event_id=None,
            chunk_id=7,
            query_text="find auth middleware",
            session_id="sess-001",
            event_type="retrieved",
        )
        self.assertEqual(event.event_type, "retrieved")
        self.assertIsNone(event.timestamp)

    def test_estimate_tokens_empty_string(self):
        from mnemosyne.models import estimate_tokens
        # Empty string returns 1 (minimum)
        self.assertEqual(estimate_tokens(""), 1)

    def test_estimate_tokens_single_word(self):
        from mnemosyne.models import estimate_tokens
        self.assertEqual(estimate_tokens("hello"), 1)

    def test_estimate_tokens_multi_word(self):
        from mnemosyne.models import estimate_tokens
        text = "def hello_world(x, y):\n    return x + y\n"
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 1)

    def test_estimate_tokens_returns_int(self):
        from mnemosyne.models import estimate_tokens
        result = estimate_tokens("some text here")
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_estimate_tokens_longer_is_more(self):
        from mnemosyne.models import estimate_tokens
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world foo bar baz qux quux corge grault")
        self.assertGreater(long, short)


# ---------------------------------------------------------------------------
# TestHasher
# ---------------------------------------------------------------------------


class TestHasher(unittest.TestCase):
    """content_hash normalisation, file_hash, and is_binary."""

    def test_content_hash_returns_64_hex_chars(self):
        from mnemosyne.hasher import content_hash
        h = content_hash("hello world")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)
        # hex only
        int(h, 16)

    def test_crlf_normalised_to_lf(self):
        """CRLF and LF endings produce the same hash."""
        from mnemosyne.hasher import content_hash
        lf_text = "line one\nline two\n"
        crlf_text = "line one\r\nline two\r\n"
        self.assertEqual(content_hash(lf_text), content_hash(crlf_text))

    def test_trailing_whitespace_stripped(self):
        """Lines with trailing spaces hash the same as stripped lines."""
        from mnemosyne.hasher import content_hash
        clean = "hello\nworld\n"
        trailing = "hello   \nworld   \n"
        self.assertEqual(content_hash(clean), content_hash(trailing))

    def test_different_content_different_hash(self):
        from mnemosyne.hasher import content_hash
        h1 = content_hash("alpha")
        h2 = content_hash("beta")
        self.assertNotEqual(h1, h2)

    def test_same_content_same_hash(self):
        from mnemosyne.hasher import content_hash
        h1 = content_hash("consistent content")
        h2 = content_hash("consistent content")
        self.assertEqual(h1, h2)

    def test_file_hash_matches_content_hash(self):
        from mnemosyne.hasher import content_hash, file_hash
        content = "line one\nline two\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            tmp_path = f.name
        try:
            self.assertEqual(file_hash(tmp_path), content_hash(content))
        finally:
            os.unlink(tmp_path)

    def test_file_hash_crlf_file_matches_lf_content_hash(self):
        """A file written with CRLF hashes the same as its LF equivalent."""
        from mnemosyne.hasher import content_hash, file_hash
        lf_content = "alpha\nbeta\n"
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            f.write(b"alpha\r\nbeta\r\n")
            tmp_path = f.name
        try:
            self.assertEqual(file_hash(tmp_path), content_hash(lf_content))
        finally:
            os.unlink(tmp_path)

    def test_is_binary_text_file_is_false(self):
        from mnemosyne.hasher import is_binary
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    pass\n")
            tmp_path = f.name
        try:
            self.assertFalse(is_binary(tmp_path))
        finally:
            os.unlink(tmp_path)

    def test_is_binary_binary_file_is_true(self):
        from mnemosyne.hasher import is_binary
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03binary data here")
            tmp_path = f.name
        try:
            self.assertTrue(is_binary(tmp_path))
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# TestBloomFilter
# ---------------------------------------------------------------------------


class TestBloomFilter(unittest.TestCase):
    """BloomFilter add, might_contain, false negatives, save/load."""

    def _make_bloom(self, capacity=1000, fp_rate=0.01):
        from mnemosyne.bloom import BloomFilter
        return BloomFilter(capacity=capacity, fp_rate=fp_rate)

    def test_add_and_might_contain_true(self):
        bf = self._make_bloom()
        bf.add("hello.py")
        self.assertTrue(bf.might_contain("hello.py"))

    def test_not_added_might_contain_false(self):
        bf = self._make_bloom()
        bf.add("hello.py")
        # "world.py" was never added -- must not be a false negative for "hello.py"
        self.assertFalse(bf.might_contain("world.py"))

    def test_contains_syntax(self):
        bf = self._make_bloom()
        bf.add("item_one")
        self.assertIn("item_one", bf)

    def test_zero_false_negatives(self):
        """Items that were added must ALWAYS be reported as present."""
        bf = self._make_bloom(capacity=500, fp_rate=0.001)
        items = [f"path/to/file_{i}.py" for i in range(200)]
        for item in items:
            bf.add(item)
        for item in items:
            self.assertTrue(
                bf.might_contain(item),
                f"False negative: {item!r} was added but might_contain returned False",
            )

    def test_save_and_load_roundtrip(self):
        """Bloom filter saved to disk and reloaded preserves membership."""
        bf = self._make_bloom(capacity=500, fp_rate=0.01)
        items = [f"file_{i}.py" for i in range(100)]
        for item in items:
            bf.add(item)

        with tempfile.NamedTemporaryFile(suffix=".bloom", delete=False) as f:
            path = f.name
        try:
            bf.save(path)
            from mnemosyne.bloom import BloomFilter
            bf2 = BloomFilter.load(path)
            for item in items:
                self.assertTrue(bf2.might_contain(item))
        finally:
            os.unlink(path)

    def test_invalid_capacity_raises(self):
        from mnemosyne.bloom import BloomFilter
        with self.assertRaises(ValueError):
            BloomFilter(capacity=0)

    def test_invalid_fp_rate_raises(self):
        from mnemosyne.bloom import BloomFilter
        with self.assertRaises(ValueError):
            BloomFilter(fp_rate=1.5)

    def test_fill_ratio_increases_after_adds(self):
        bf = self._make_bloom(capacity=100)
        ratio_before = bf.fill_ratio
        for i in range(50):
            bf.add(f"item_{i}")
        self.assertGreater(bf.fill_ratio, ratio_before)

    def test_memory_bytes_positive(self):
        bf = self._make_bloom()
        self.assertGreater(bf.memory_bytes, 0)


# ---------------------------------------------------------------------------
# TestSchema
# ---------------------------------------------------------------------------


class TestSchema(unittest.TestCase):
    """Schema initialisation creates all expected tables; FTS5 works."""

    def _open_memory_db(self):
        """Return an in-memory SQLite connection with the schema applied."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(conn)
        return conn

    def _table_names(self, conn):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow') ORDER BY name"
        ).fetchall()
        return {r[0] for r in rows}

    def test_all_core_tables_created(self):
        conn = self._open_memory_db()
        names = self._table_names(conn)
        expected = {
            "files", "chunks", "embeddings", "sparse_embeddings",
            "vocabulary", "summaries", "usage_events", "cache_state",
            "task_patterns", "file_deltas", "schema_version",
        }
        for table in expected:
            self.assertIn(table, names, f"Missing table: {table}")

    def test_schema_version_seeded(self):
        conn = self._open_memory_db()
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["version"], 1)

    def test_init_db_is_idempotent(self):
        """Calling init_db twice on the same connection does not raise."""
        conn = self._open_memory_db()
        from mnemosyne.schema import init_db
        init_db(conn)  # second call
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        self.assertIsNotNone(row)

    def test_fts5_table_exists(self):
        conn = self._open_memory_db()
        # chunks_fts is a virtual table; check via a query
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'chunks_fts'"
        ).fetchall()
        self.assertEqual(len(rows), 1)

    def test_fts5_insert_and_search(self):
        """FTS5 index can be populated via trigger and queried."""
        conn = self._open_memory_db()
        # Insert a file
        conn.execute(
            "INSERT INTO files (rel_path, content_hash, size_bytes, last_modified) "
            "VALUES ('test.py', 'aaa', 100, 0.0)"
        )
        file_id = conn.execute(
            "SELECT file_id FROM files WHERE rel_path = 'test.py'"
        ).fetchone()["file_id"]

        # Insert a chunk -- the trigger auto-populates chunks_fts
        conn.execute(
            "INSERT INTO chunks (file_id, content_hash, chunk_type, line_start, line_end, "
            "token_count, content) VALUES (?, 'bbb', 'function', 1, 5, 10, ?)",
            (file_id, "def authentication_middleware(request):\n    pass\n"),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'authentication'"
        ).fetchall()
        self.assertGreater(len(rows), 0)

    def test_foreign_key_cascade_on_file_delete(self):
        """Deleting a file cascades to its chunks."""
        conn = self._open_memory_db()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO files (rel_path, content_hash, size_bytes, last_modified) "
            "VALUES ('cascade_test.py', 'ccc', 50, 0.0)"
        )
        file_id = conn.execute(
            "SELECT file_id FROM files WHERE rel_path='cascade_test.py'"
        ).fetchone()["file_id"]
        conn.execute(
            "INSERT INTO chunks (file_id, content_hash, chunk_type, line_start, line_end, "
            "token_count, content) VALUES (?, 'ddd', 'block', 1, 3, 5, 'some content')",
            (file_id,),
        )
        conn.commit()

        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(chunk_count, 1)

        conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        conn.commit()

        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.assertEqual(chunk_count, 0)


if __name__ == "__main__":
    unittest.main()
