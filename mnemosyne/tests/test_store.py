# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for Store — the repository / CRUD layer (store.py).

All tests use an in-memory SQLite database (:memory:) to avoid filesystem I/O.
"""

import sqlite3
import unittest


def _make_store():
    """Return a fresh (Store, conn) pair backed by an in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    from mnemosyne.schema import init_db
    init_db(conn)
    from mnemosyne.store import Store
    return Store(conn), conn


def _file_record(rel_path="src/app.py", language="python"):
    from mnemosyne.models import FileRecord
    return FileRecord(
        file_id=None,
        rel_path=rel_path,
        content_hash="abc123",
        size_bytes=512,
        language=language,
        last_modified=1700000000.0,
        last_indexed=None,
        is_deleted=False,
    )


def _chunk(file_id, content="def foo():\n    pass\n", chunk_type="function"):
    from mnemosyne.hasher import content_hash
    from mnemosyne.models import Chunk, estimate_tokens
    return Chunk(
        chunk_id=None,
        file_id=file_id,
        content_hash=content_hash(content),
        chunk_type=chunk_type,
        line_start=1,
        line_end=content.count("\n"),
        token_count=estimate_tokens(content),
        content=content,
    )


class TestStore(unittest.TestCase):

    # ------------------------------------------------------------------
    # File CRUD
    # ------------------------------------------------------------------

    def test_upsert_file_insert_returns_id(self):
        store, _ = _make_store()
        rec = _file_record()
        fid = store.upsert_file(rec)
        self.assertIsInstance(fid, int)
        self.assertGreater(fid, 0)

    def test_get_file_returns_record(self):
        store, _ = _make_store()
        rec = _file_record("src/utils.py")
        store.upsert_file(rec)
        fetched = store.get_file("src/utils.py")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.rel_path, "src/utils.py")
        self.assertEqual(fetched.language, "python")

    def test_get_file_missing_returns_none(self):
        store, _ = _make_store()
        self.assertIsNone(store.get_file("nonexistent.py"))

    def test_get_file_by_id(self):
        store, _ = _make_store()
        rec = _file_record("src/core.py")
        fid = store.upsert_file(rec)
        fetched = store.get_file_by_id(fid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.rel_path, "src/core.py")

    def test_upsert_file_updates_on_conflict(self):
        """Second upsert with same rel_path updates the row."""
        store, _ = _make_store()
        rec = _file_record("src/app.py")
        store.upsert_file(rec)

        from mnemosyne.models import FileRecord
        updated_rec = FileRecord(
            file_id=None,
            rel_path="src/app.py",
            content_hash="newHash999",
            size_bytes=2048,
            language="python",
            last_modified=1700001000.0,
        )
        store.upsert_file(updated_rec)

        fetched = store.get_file("src/app.py")
        self.assertEqual(fetched.content_hash, "newHash999")
        self.assertEqual(fetched.size_bytes, 2048)

    def test_list_files_excludes_deleted(self):
        store, _ = _make_store()
        fid1 = store.upsert_file(_file_record("src/a.py"))
        store.upsert_file(_file_record("src/b.py"))
        store.mark_deleted(fid1)

        live = store.list_files(include_deleted=False)
        paths = [r.rel_path for r in live]
        self.assertNotIn("src/a.py", paths)
        self.assertIn("src/b.py", paths)

    def test_list_files_include_deleted(self):
        store, _ = _make_store()
        fid1 = store.upsert_file(_file_record("src/a.py"))
        store.upsert_file(_file_record("src/b.py"))
        store.mark_deleted(fid1)

        all_files = store.list_files(include_deleted=True)
        self.assertEqual(len(all_files), 2)

    def test_mark_deleted_sets_flag(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record("src/c.py"))
        store.mark_deleted(fid)
        fetched = store.get_file_by_id(fid)
        self.assertTrue(fetched.is_deleted)

    def test_count_files_excludes_deleted(self):
        store, _ = _make_store()
        fid1 = store.upsert_file(_file_record("src/a.py"))
        store.upsert_file(_file_record("src/b.py"))
        store.mark_deleted(fid1)
        self.assertEqual(store.count_files(), 1)

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    def test_insert_chunk_returns_id(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))
        self.assertIsInstance(cid, int)
        self.assertGreater(cid, 0)

    def test_get_chunk_returns_chunk(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid, content="def bar():\n    return 1\n"))
        fetched = store.get_chunk(cid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.chunk_type, "function")
        self.assertIn("def bar", fetched.content)

    def test_get_chunk_missing_returns_none(self):
        store, _ = _make_store()
        self.assertIsNone(store.get_chunk(99999))

    def test_insert_chunks_bulk(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        chunks = [
            _chunk(fid, content=f"def fn_{i}():\n    pass\n")
            for i in range(5)
        ]
        ids = store.insert_chunks(chunks)
        self.assertEqual(len(ids), 5)
        # All IDs are distinct
        self.assertEqual(len(set(ids)), 5)

    def test_get_chunks_for_file_ordered_by_line_start(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        from mnemosyne.models import Chunk
        from mnemosyne.hasher import content_hash
        for start in [10, 1, 5]:
            c = Chunk(
                chunk_id=None,
                file_id=fid,
                content_hash=content_hash(f"content at line {start}"),
                chunk_type="block",
                line_start=start,
                line_end=start + 2,
                token_count=5,
                content=f"content at line {start}",
            )
            store.insert_chunk(c)
        chunks = store.get_chunks_for_file(fid)
        starts = [c.line_start for c in chunks]
        self.assertEqual(starts, sorted(starts))

    def test_delete_chunks_for_file(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        for i in range(3):
            store.insert_chunk(_chunk(fid, content=f"def f{i}():\n    pass\n"))
        self.assertEqual(store.count_chunks(), 3)
        store.delete_chunks_for_file(fid)
        self.assertEqual(store.count_chunks(), 0)

    def test_chunk_exists(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        from mnemosyne.hasher import content_hash
        content = "def check():\n    pass\n"
        ch = content_hash(content)
        cand = _chunk(fid, content=content)
        store.insert_chunk(cand)
        self.assertTrue(store.chunk_exists(ch, fid, 1))
        self.assertFalse(store.chunk_exists("nonexistent_hash", fid, 1))

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def test_search_fts_returns_results(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(
            fid,
            content="def authenticate_user(token):\n    return verify(token)\n",
        ))
        results = store.search_fts("authenticate_user")
        self.assertGreater(len(results), 0)

    def test_search_fts_returns_tuple_pairs(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(fid, content="def run_pipeline():\n    pass\n"))
        results = store.search_fts("run_pipeline")
        for item in results:
            self.assertEqual(len(item), 2)
            chunk_id, score = item
            self.assertIsInstance(chunk_id, int)
            self.assertIsInstance(score, float)

    def test_search_fts_empty_query_returns_empty(self):
        store, _ = _make_store()
        results = store.search_fts("")
        self.assertEqual(results, [])

    def test_search_fts_no_match_returns_empty(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(fid, content="def hello():\n    pass\n"))
        results = store.search_fts("xyzzy_nonexistent_term_99")
        self.assertEqual(results, [])

    # ------------------------------------------------------------------
    # Sparse embeddings
    # ------------------------------------------------------------------

    def test_insert_and_get_sparse_embedding(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))
        terms = {"def": 0.5, "pass": 0.3, "python": 0.8}
        store.insert_sparse_embedding(cid, terms)
        fetched = store.get_sparse_embedding(cid)
        self.assertIsNotNone(fetched)
        self.assertAlmostEqual(fetched["python"], 0.8)

    def test_get_sparse_embedding_missing_returns_none(self):
        store, _ = _make_store()
        self.assertIsNone(store.get_sparse_embedding(99999))

    def test_get_all_sparse_embeddings(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        for i in range(3):
            cid = store.insert_chunk(_chunk(fid, content=f"def fn_{i}():\n    pass\n"))
            store.insert_sparse_embedding(cid, {f"term_{i}": float(i + 1)})
        all_embs = store.get_all_sparse_embeddings()
        self.assertEqual(len(all_embs), 3)

    def test_sparse_embedding_upsert(self):
        """Upserting over an existing chunk_id updates the weights."""
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))
        store.insert_sparse_embedding(cid, {"old_term": 1.0})
        store.insert_sparse_embedding(cid, {"new_term": 2.0})
        fetched = store.get_sparse_embedding(cid)
        self.assertIn("new_term", fetched)
        self.assertNotIn("old_term", fetched)

    # ------------------------------------------------------------------
    # Usage events
    # ------------------------------------------------------------------

    def test_record_usage_and_get_scores(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))

        from mnemosyne.models import UsageEvent
        event = UsageEvent(
            event_id=None,
            chunk_id=cid,
            query_text="test query",
            session_id="s1",
            event_type="retrieved",
        )
        store.record_usage(event)
        scores = store.get_usage_scores()
        self.assertIn(cid, scores)

    def test_count_usage_events(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))
        from mnemosyne.models import UsageEvent
        for _ in range(3):
            store.record_usage(UsageEvent(None, cid, None, "s1", "retrieved"))
        self.assertEqual(store.count_usage_events(), 3)

    def test_get_usage_events_for_chunk(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))
        from mnemosyne.models import UsageEvent
        store.record_usage(UsageEvent(None, cid, "q1", "s1", "selected"))
        store.record_usage(UsageEvent(None, cid, "q2", "s1", "used"))
        events = store.get_usage_events_for_chunk(cid)
        self.assertEqual(len(events), 2)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def test_get_stats_keys(self):
        store, _ = _make_store()
        stats = store.get_stats()
        expected_keys = {
            "files", "files_deleted", "chunks", "summaries",
            "usage_events", "vocabulary_size", "cache_entries",
            "task_patterns", "file_deltas", "total_tokens_indexed",
        }
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing stat key: {key}")

    def test_get_stats_counts_increase_after_insert(self):
        store, _ = _make_store()
        before = store.get_stats()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(fid))
        after = store.get_stats()
        self.assertEqual(after["files"], before["files"] + 1)
        self.assertEqual(after["chunks"], before["chunks"] + 1)

    def test_total_tokens(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(fid, content="def hello():\n    pass\n"))
        total = store.total_tokens()
        self.assertGreater(total, 0)

    def test_chunk_type_counts(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.insert_chunk(_chunk(fid, chunk_type="function"))
        store.insert_chunk(_chunk(fid, content="class Foo:\n    pass\n", chunk_type="class"))
        counts = store.chunk_type_counts()
        self.assertIn("function", counts)
        self.assertIn("class", counts)

    # ------------------------------------------------------------------
    # File deltas
    # ------------------------------------------------------------------

    def test_record_and_get_deltas(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        store.record_delta(fid, "old_hash", "new_hash", "--- a\n+++ b\n+added line\n")
        deltas = store.get_recent_deltas(fid)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["old_hash"], "old_hash")
        self.assertEqual(deltas[0]["new_hash"], "new_hash")

    # ------------------------------------------------------------------
    # Cache state
    # ------------------------------------------------------------------

    def test_save_and_load_cache_state(self):
        store, _ = _make_store()
        fid = store.upsert_file(_file_record())
        cid = store.insert_chunk(_chunk(fid))

        from mnemosyne.models import CacheEntry
        entries = [CacheEntry(chunk_id=cid, tier="T1", access_count=3, last_accessed="2024-01-01T00:00:00Z")]
        store.save_cache_state(entries)
        loaded = store.load_cache_state()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].tier, "T1")
        self.assertEqual(loaded[0].access_count, 3)


    # ------------------------------------------------------------------
    # Index metadata
    # ------------------------------------------------------------------

    def test_set_and_get_index_metadata(self):
        store, _ = _make_store()
        store.set_index_metadata("tokenizer_hash", "abc123")
        val = store.get_index_metadata("tokenizer_hash")
        self.assertEqual(val, "abc123")

    def test_get_index_metadata_missing_returns_none(self):
        store, _ = _make_store()
        self.assertIsNone(store.get_index_metadata("nonexistent_key"))

    def test_set_index_metadata_upserts(self):
        store, _ = _make_store()
        store.set_index_metadata("key1", "old_value")
        store.set_index_metadata("key1", "new_value")
        self.assertEqual(store.get_index_metadata("key1"), "new_value")


class TestConcurrentWrites(unittest.TestCase):
    """Verify that two Store instances can write concurrently without corruption."""

    def test_concurrent_upsert_files(self):
        """Two threads writing files to the same on-disk DB should not corrupt."""
        import tempfile
        import threading
        from pathlib import Path
        from mnemosyne.schema import open_store
        from mnemosyne.store import Store

        tmp = tempfile.mkdtemp()
        n_per_thread = 20
        errors: list[Exception] = []

        def _worker(thread_id: int) -> None:
            try:
                from mnemosyne.schema import get_connection, init_db
                conn = get_connection(Path(tmp) / "mnemosyne.db")
                init_db(conn)
                store = Store(conn)
                for i in range(n_per_thread):
                    store.upsert_file(_file_record(
                        rel_path=f"t{thread_id}/file_{i}.py",
                        language="python",
                    ))
            except Exception as exc:
                errors.append(exc)

        # Pre-create the DB so both threads open the same file.
        conn0 = open_store(tmp)
        conn0.close()

        t1 = threading.Thread(target=_worker, args=(1,))
        t2 = threading.Thread(target=_worker, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        self.assertEqual(errors, [], f"Concurrent writes failed: {errors}")

        # Verify all rows landed.
        from mnemosyne.schema import get_connection, init_db
        conn = get_connection(Path(tmp) / "mnemosyne.db")
        init_db(conn)
        store = Store(conn)
        total = store.count_files()
        self.assertEqual(total, n_per_thread * 2)


if __name__ == "__main__":
    unittest.main()
