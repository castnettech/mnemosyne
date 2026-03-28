# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the Mnemosyne JSON-RPC daemon.

Uses threading (not forking) to run the daemon in the background, and a
temp directory for the socket so tests are isolated and do not interfere
with any live daemon.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import tempfile
import threading
import time
import unittest


# ---------------------------------------------------------------------------
# Sample file written into the temp project for queries to find.
# ---------------------------------------------------------------------------

SAMPLE_PYTHON = '''\
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
'''


class _DaemonTestBase(unittest.TestCase):
    """Shared setup: temp project dir, schema, sample files, daemon instance."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mnemosyne_dir = os.path.join(self.tmp_dir, ".mnemosyne")
        os.makedirs(self.mnemosyne_dir)

        # Write sample source file so ingest has something to index.
        src_dir = os.path.join(self.tmp_dir, "src")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "auth.py"), "w") as f:
            f.write(SAMPLE_PYTHON)

        # Initialise the database and write a default config so the daemon
        # can open without errors.
        from mnemosyne.config import Config
        cfg = Config(root=self.tmp_dir)
        cfg.save()

        from mnemosyne.schema import open_store
        conn = open_store(self.mnemosyne_dir)
        conn.close()

        # Build a daemon that points at our temp project.
        from mnemosyne.daemon import MnemosyneDaemon
        self.daemon = MnemosyneDaemon(self.tmp_dir)
        self._daemon_thread: threading.Thread | None = None

    def tearDown(self):
        # Ensure the daemon is stopped even if a test fails.
        try:
            self.daemon.stop()
        except Exception:
            pass
        # Wait for the daemon thread to finish.
        if self._daemon_thread is not None and self._daemon_thread.is_alive():
            self._daemon_thread.join(timeout=3.0)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- helpers ----------------------------------------------------------

    def _start_daemon_in_thread(self) -> None:
        """Start the daemon in a background thread (foreground mode)."""
        self._daemon_thread = threading.Thread(
            target=self.daemon.start, daemon=True,
        )
        self._daemon_thread.start()
        # Give the socket time to bind.
        self._wait_for_socket(timeout=3.0)

    def _wait_for_socket(self, timeout: float = 3.0) -> None:
        """Block until the daemon socket file appears or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(self.daemon._socket_path):
                return
            time.sleep(0.05)
        raise RuntimeError("Daemon socket did not appear in time.")

    def _send_request(self, request: dict) -> dict:
        """Send a JSON-RPC request to the daemon and return the parsed response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect(self.daemon._socket_path)
            payload = json.dumps(request).encode("utf-8") + b"\n"
            sock.sendall(payload)
            # Shutdown write side so the daemon sees EOF after the newline.
            sock.shutdown(socket.SHUT_WR)

            # Read the full response.
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
            return json.loads(data.strip())
        finally:
            sock.close()

    def _send_raw(self, raw_bytes: bytes) -> bytes:
        """Send raw bytes to the daemon and return the raw response bytes."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        try:
            sock.connect(self.daemon._socket_path)
            sock.sendall(raw_bytes)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
            return data
        finally:
            sock.close()


class TestDaemonStartForegroundAndQuery(_DaemonTestBase):
    """Start daemon in a thread, send a query via socket, verify response."""

    def test_query_returns_results(self):
        # Ingest sample files first so queries have content to search.
        from mnemosyne.bloom import BloomFilter
        from mnemosyne.audit import AuditLog
        from mnemosyne.ingest import Ingester

        bloom = BloomFilter()
        audit_path = os.path.join(self.mnemosyne_dir, "audit.log")
        audit = AuditLog(audit_path)
        ingester = Ingester(
            project_root=self.tmp_dir,
            config=self.daemon.config,
            store=self.daemon.store,
            bloom=bloom,
            tfidf_backend=self.daemon.tfidf,
            audit=audit,
        )
        stats = ingester.ingest()
        self.assertGreater(stats["files_indexed"], 0)

        # Rebuild inverted index.
        all_embs = self.daemon.store.get_all_sparse_embeddings()
        if all_embs and hasattr(self.daemon.tfidf, "build_inverted_index"):
            self.daemon.tfidf.build_inverted_index(all_embs)

        self._start_daemon_in_thread()

        resp = self._send_request({
            "method": "query",
            "params": {"query": "hash password", "budget": 4000},
        })

        self.assertIsNone(resp.get("error"))
        result = resp["result"]
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)
        self.assertGreater(len(result["results"]), 0)
        self.assertEqual(result["query"], "hash password")

    def test_stats_returns_counts(self):
        self._start_daemon_in_thread()

        resp = self._send_request({"method": "stats", "params": {}})

        self.assertIsNone(resp.get("error"))
        result = resp["result"]
        self.assertIn("project_root", result)
        self.assertIn("files_indexed", result)
        self.assertIn("chunks", result)
        self.assertIn("total_tokens", result)


class TestDaemonPidFile(_DaemonTestBase):
    """Verify PID file lifecycle."""

    def test_pid_file_created_and_removed(self):
        self._start_daemon_in_thread()

        # PID file should exist.
        self.assertTrue(os.path.isfile(self.daemon._pid_path))

        # PID should be a valid integer.
        with open(self.daemon._pid_path) as f:
            pid_contents = f.read().strip()
        pid = int(pid_contents)
        self.assertGreater(pid, 0)

        # Stop the daemon.
        self.daemon.stop()

        # Wait briefly for the thread to finish cleanup.
        if self._daemon_thread is not None:
            self._daemon_thread.join(timeout=3.0)

        # PID file should be gone.
        self.assertFalse(os.path.isfile(self.daemon._pid_path))


class TestDaemonHandlesInvalidJson(_DaemonTestBase):
    """Send malformed JSON, verify error response (not crash)."""

    def test_invalid_json_returns_parse_error(self):
        self._start_daemon_in_thread()

        raw = self._send_raw(b"this is not json\n")
        resp = json.loads(raw.strip())

        self.assertIsNone(resp.get("result"))
        error = resp.get("error")
        self.assertIsNotNone(error)
        self.assertEqual(error["code"], -32700)
        self.assertIn("Invalid JSON", error["message"])

    def test_empty_request_does_not_crash(self):
        """An empty payload should not cause the daemon to crash."""
        self._start_daemon_in_thread()

        # Send empty bytes — server should handle gracefully.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(self.daemon._socket_path)
            sock.shutdown(socket.SHUT_WR)
        finally:
            sock.close()

        # Daemon should still be alive for another request.
        resp = self._send_request({"method": "stats", "params": {}})
        self.assertIsNone(resp.get("error"))


class TestDaemonHandlesUnknownMethod(_DaemonTestBase):
    """Send unknown method, verify error response."""

    def test_unknown_method_returns_error(self):
        self._start_daemon_in_thread()

        resp = self._send_request({
            "method": "nonexistent_method",
            "params": {},
        })

        self.assertIsNone(resp.get("result"))
        error = resp.get("error")
        self.assertIsNotNone(error)
        self.assertEqual(error["code"], -32601)
        self.assertIn("Unknown method", error["message"])

    def test_missing_method_returns_error(self):
        self._start_daemon_in_thread()

        resp = self._send_request({"params": {}})

        self.assertIsNone(resp.get("result"))
        error = resp.get("error")
        self.assertIsNotNone(error)
        self.assertIn("method", error["message"].lower())


class TestDaemonFeedbackAndIngest(_DaemonTestBase):
    """Test the feedback and ingest RPC methods."""

    def test_ingest_via_daemon(self):
        self._start_daemon_in_thread()

        resp = self._send_request({
            "method": "ingest",
            "params": {},
        })

        self.assertIsNone(resp.get("error"))
        result = resp["result"]
        self.assertIn("files_indexed", result)
        self.assertGreater(result["files_indexed"], 0)

    def test_feedback_records_event(self):
        # Ingest first so we have a chunk_id to reference.
        self._start_daemon_in_thread()

        ingest_resp = self._send_request({
            "method": "ingest",
            "params": {},
        })
        self.assertIsNone(ingest_resp.get("error"))

        # Get stats to confirm we have chunks.
        stats_resp = self._send_request({"method": "stats", "params": {}})
        self.assertGreater(stats_resp["result"]["chunks"], 0)

        # Send feedback for chunk 1.
        fb_resp = self._send_request({
            "method": "feedback",
            "params": {"chunk_id": 1, "event_type": "used"},
        })

        self.assertIsNone(fb_resp.get("error"))
        result = fb_resp["result"]
        self.assertTrue(result["recorded"])
        self.assertEqual(result["chunk_id"], 1)
        self.assertEqual(result["event_type"], "used")

    def test_feedback_rejects_invalid_event_type(self):
        self._start_daemon_in_thread()

        resp = self._send_request({
            "method": "feedback",
            "params": {"chunk_id": 1, "event_type": "invalid_type"},
        })

        self.assertIsNotNone(resp.get("error"))
        self.assertIn("event_type", resp["error"]["message"])


class TestDaemonJsonRpcId(_DaemonTestBase):
    """Verify that JSON-RPC id field is echoed back."""

    def test_id_echoed_in_success(self):
        self._start_daemon_in_thread()

        resp = self._send_request({
            "id": 42,
            "method": "stats",
            "params": {},
        })

        self.assertEqual(resp.get("id"), 42)
        self.assertIsNone(resp.get("error"))

    def test_id_echoed_in_error(self):
        self._start_daemon_in_thread()

        resp = self._send_request({
            "id": "abc",
            "method": "nonexistent",
            "params": {},
        })

        self.assertEqual(resp.get("id"), "abc")
        self.assertIsNotNone(resp.get("error"))


if __name__ == "__main__":
    unittest.main()
