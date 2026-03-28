# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the feedback telemetry pipeline (Milestone 2.3).

Covers:
  - Precision-at-k computation with mixed event types
  - Session-scoped precision filtering
  - Graceful handling when no feedback events exist
  - Top-used-chunks ranking
"""

import os
import sqlite3
import tempfile
import unittest

from mnemosyne.config import Config
from mnemosyne.models import Chunk, FileRecord, UsageEvent
from mnemosyne.schema import init_db
from mnemosyne.store import Store
from mnemosyne.analytics import Analytics


class _AnalyticsTestBase(unittest.TestCase):
    """Shared setup: in-memory DB with schema, store, config, analytics."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cfg = Config(root=self.tmp_dir)

        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        init_db(self.conn)

        self.store = Store(self.conn)
        self.analytics = Analytics(self.store, self.cfg)

        # Insert a file record so chunks have a valid FK
        self._file_id = self.store.upsert_file(FileRecord(
            file_id=None,
            rel_path="src/example.py",
            content_hash="abc123",
            size_bytes=100,
            language="python",
            last_modified=1000.0,
            last_indexed=None,
        ))

        # Insert a couple of chunks to reference in events
        self._chunk_ids: list[int] = []
        for i in range(3):
            cid = self.store.insert_chunk(Chunk(
                chunk_id=None,
                file_id=self._file_id,
                content_hash=f"hash_{i}",
                chunk_type="function",
                line_start=i * 10 + 1,
                line_end=(i + 1) * 10,
                token_count=50,
                content=f"def func_{i}(): pass",
                symbol_name=f"func_{i}",
            ))
            self._chunk_ids.append(cid)

    def tearDown(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _record(self, chunk_id: int, event_type: str, session_id: str) -> None:
        """Shorthand to record a usage event via the Analytics layer."""
        self.analytics.start_session(session_id)
        self.analytics.record(chunk_id=chunk_id, event_type=event_type)


class TestPrecisionAtKComputation(_AnalyticsTestBase):
    """Verify precision = used / (used + discarded)."""

    def test_basic_precision(self):
        """3 used + 2 discarded => precision = 3/5 = 0.6."""
        sid = "sess-basic"
        for cid in self._chunk_ids:
            self._record(cid, "used", sid)
        # Record 2 discarded events (reuse first two chunks)
        self._record(self._chunk_ids[0], "discarded", sid)
        self._record(self._chunk_ids[1], "discarded", sid)

        result = self.analytics.compute_precision_at_k()
        self.assertAlmostEqual(result["precision"], 0.6, places=5)
        self.assertEqual(result["total_used"], 3)
        self.assertEqual(result["total_discarded"], 2)

    def test_all_used_gives_perfect_precision(self):
        sid = "sess-perfect"
        for cid in self._chunk_ids:
            self._record(cid, "used", sid)

        result = self.analytics.compute_precision_at_k()
        self.assertAlmostEqual(result["precision"], 1.0, places=5)
        self.assertEqual(result["total_discarded"], 0)

    def test_all_discarded_gives_zero_precision(self):
        sid = "sess-zero"
        for cid in self._chunk_ids:
            self._record(cid, "discarded", sid)

        result = self.analytics.compute_precision_at_k()
        self.assertAlmostEqual(result["precision"], 0.0, places=5)
        self.assertEqual(result["total_used"], 0)

    def test_selected_events_counted_but_not_in_precision(self):
        """'selected' and 'retrieved' do not affect precision numerator/denominator."""
        sid = "sess-sel"
        self._record(self._chunk_ids[0], "used", sid)
        self._record(self._chunk_ids[1], "discarded", sid)
        self._record(self._chunk_ids[2], "selected", sid)
        self._record(self._chunk_ids[0], "retrieved", sid)

        result = self.analytics.compute_precision_at_k()
        # precision = 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(result["precision"], 0.5, places=5)
        self.assertEqual(result["total_selected"], 1)
        self.assertEqual(result["total_retrieved"], 1)

    def test_precision_returns_all_expected_keys(self):
        result = self.analytics.compute_precision_at_k()
        for key in ("precision", "total_retrieved", "total_used",
                     "total_discarded", "total_selected"):
            self.assertIn(key, result)


class TestPrecisionAtKBySession(_AnalyticsTestBase):
    """Verify session_id filtering isolates results correctly."""

    def test_session_filter_isolates_events(self):
        # Session A: 2 used, 0 discarded => precision 1.0
        self._record(self._chunk_ids[0], "used", "sess-A")
        self._record(self._chunk_ids[1], "used", "sess-A")

        # Session B: 0 used, 2 discarded => precision 0.0
        self._record(self._chunk_ids[0], "discarded", "sess-B")
        self._record(self._chunk_ids[1], "discarded", "sess-B")

        result_a = self.analytics.compute_precision_at_k(session_id="sess-A")
        self.assertAlmostEqual(result_a["precision"], 1.0, places=5)
        self.assertEqual(result_a["total_used"], 2)
        self.assertEqual(result_a["total_discarded"], 0)

        result_b = self.analytics.compute_precision_at_k(session_id="sess-B")
        self.assertAlmostEqual(result_b["precision"], 0.0, places=5)
        self.assertEqual(result_b["total_used"], 0)
        self.assertEqual(result_b["total_discarded"], 2)

    def test_nonexistent_session_returns_zeros(self):
        self._record(self._chunk_ids[0], "used", "sess-exists")

        result = self.analytics.compute_precision_at_k(session_id="no-such-session")
        self.assertAlmostEqual(result["precision"], 0.0)
        self.assertEqual(result["total_used"], 0)
        self.assertEqual(result["total_discarded"], 0)

    def test_aggregate_mixes_all_sessions(self):
        # Session A: 1 used
        self._record(self._chunk_ids[0], "used", "sess-A")
        # Session B: 1 discarded
        self._record(self._chunk_ids[1], "discarded", "sess-B")

        result = self.analytics.compute_precision_at_k(session_id=None)
        # 1 used + 1 discarded => precision = 0.5
        self.assertAlmostEqual(result["precision"], 0.5, places=5)


class TestAnalyticsWithNoEvents(_AnalyticsTestBase):
    """Verify graceful handling when the usage_events table is empty."""

    def test_precision_with_no_events(self):
        result = self.analytics.compute_precision_at_k()
        self.assertAlmostEqual(result["precision"], 0.0)
        self.assertEqual(result["total_retrieved"], 0)
        self.assertEqual(result["total_used"], 0)
        self.assertEqual(result["total_discarded"], 0)
        self.assertEqual(result["total_selected"], 0)

    def test_precision_with_session_and_no_events(self):
        result = self.analytics.compute_precision_at_k(session_id="empty")
        self.assertAlmostEqual(result["precision"], 0.0)

    def test_top_used_chunks_empty(self):
        top = self.analytics.get_top_used_chunks(limit=5)
        self.assertEqual(top, [])


class TestTopUsedChunks(_AnalyticsTestBase):
    """Verify top-used-chunks ranking and metadata."""

    def test_ranking_order(self):
        sid = "sess-rank"
        # chunk 0: 3 uses, chunk 1: 1 use, chunk 2: 5 uses
        for _ in range(3):
            self._record(self._chunk_ids[0], "used", sid)
        self._record(self._chunk_ids[1], "used", sid)
        for _ in range(5):
            self._record(self._chunk_ids[2], "used", sid)

        top = self.analytics.get_top_used_chunks(limit=3)
        self.assertEqual(len(top), 3)
        # Highest first
        self.assertEqual(top[0]["chunk_id"], self._chunk_ids[2])
        self.assertEqual(top[0]["use_count"], 5)
        self.assertEqual(top[1]["chunk_id"], self._chunk_ids[0])
        self.assertEqual(top[1]["use_count"], 3)
        self.assertEqual(top[2]["chunk_id"], self._chunk_ids[1])
        self.assertEqual(top[2]["use_count"], 1)

    def test_limit_respected(self):
        sid = "sess-lim"
        for cid in self._chunk_ids:
            self._record(cid, "used", sid)

        top = self.analytics.get_top_used_chunks(limit=1)
        self.assertEqual(len(top), 1)

    def test_entry_metadata(self):
        sid = "sess-meta"
        self._record(self._chunk_ids[0], "used", sid)

        top = self.analytics.get_top_used_chunks(limit=1)
        entry = top[0]
        self.assertEqual(entry["file_path"], "src/example.py")
        self.assertEqual(entry["symbol_name"], "func_0")
        self.assertEqual(entry["line_start"], 1)
        self.assertEqual(entry["line_end"], 10)
        self.assertEqual(entry["use_count"], 1)

    def test_discarded_events_not_counted(self):
        """Only 'used' events should factor into top-used ranking."""
        sid = "sess-disc"
        self._record(self._chunk_ids[0], "discarded", sid)
        self._record(self._chunk_ids[0], "discarded", sid)
        self._record(self._chunk_ids[1], "used", sid)

        top = self.analytics.get_top_used_chunks(limit=5)
        # Only chunk 1 should appear
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["chunk_id"], self._chunk_ids[1])


if __name__ == "__main__":
    unittest.main()
