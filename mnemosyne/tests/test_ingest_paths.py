# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Negative security tests and functional tests for the M-FIX path resolution
and directory expansion changes in mnemosyne.ingest.

Covers:
  - Path containment: parent escape, absolute outside root, symlink escape
  - Directory expansion: single dir, mixed files and dirs
  - File filtering: unsupported extensions, non-existent paths
  - Deduplication: same file via relative + absolute paths
  - _scan_dir refactor: _scan_files delegates to _scan_dir
"""

import os
import sqlite3
import unittest
from unittest.mock import patch

from mnemosyne.config import Config
from mnemosyne.ingest import Ingester
from mnemosyne.schema import init_db
from mnemosyne.store import Store
from mnemosyne.bloom import BloomFilter
from mnemosyne.audit import AuditLog
from mnemosyne.embeddings.tfidf_backend import TFIDFBackend


def _build_ingester(project_root: str) -> Ingester:
    """
    Build an Ingester wired to a minimal in-memory stack.

    Path resolution tests never touch store/bloom/tfidf/audit, but the
    Ingester constructor requires them, so we provide real lightweight
    instances rather than mocks.
    """
    cfg = Config(root=project_root)
    cfg.embedding.tfidf_min_df = 1

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    store = Store(conn)

    tfidf = TFIDFBackend(cfg, store=None)
    bloom = BloomFilter(capacity=1_000, fp_rate=0.01)

    audit_path = os.path.join(project_root, ".mnemosyne", "audit.jsonl")
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    audit = AuditLog(audit_path)

    return Ingester(
        project_root=project_root,
        config=cfg,
        store=store,
        bloom=bloom,
        tfidf_backend=tfidf,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# Path containment -- ValueError expected
# ---------------------------------------------------------------------------


class TestResolvePathsContainment(unittest.TestCase):
    """_resolve_paths must reject any path that escapes the project root."""

    def test_resolve_paths_rejects_parent_escape(self, tmp_path=None):
        """Relative parent traversal (../../etc/passwd) must raise ValueError."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)
            with self.assertRaises(ValueError) as ctx:
                ingester._resolve_paths(["../../etc/passwd"])
            self.assertIn("outside project root", str(ctx.exception))

    def test_resolve_paths_rejects_absolute_outside_root(self):
        """Absolute path outside the project root must raise ValueError."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)
            with self.assertRaises(ValueError) as ctx:
                ingester._resolve_paths(["/tmp/evil"])
            self.assertIn("outside project root", str(ctx.exception))

    def test_resolve_paths_rejects_symlink_escape(self):
        """Symlink inside root pointing to /tmp must raise ValueError."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)
            symlink_path = os.path.join(tmp, "escape_link")
            os.symlink("/tmp", symlink_path)
            with self.assertRaises(ValueError) as ctx:
                ingester._resolve_paths([symlink_path])
            self.assertIn("outside project root", str(ctx.exception))


# ---------------------------------------------------------------------------
# Directory expansion -- should work
# ---------------------------------------------------------------------------


class TestResolvePathsDirectoryExpansion(unittest.TestCase):
    """_resolve_paths should expand directories into their indexable files."""

    def test_resolve_paths_expands_directory(self):
        """A directory containing .py files should expand to those files."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)

            sub = os.path.join(tmp, "subpkg")
            os.makedirs(sub)
            for name in ("alpha.py", "beta.py"):
                with open(os.path.join(sub, name), "w") as f:
                    f.write(f"# {name}\n")

            result = ingester._resolve_paths([sub])
            basenames = sorted(os.path.basename(p) for p in result)
            self.assertEqual(basenames, ["alpha.py", "beta.py"])

    def test_resolve_paths_mixed_files_and_dirs(self):
        """Mix of an explicit file and a directory, all within root."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)

            # Explicit file
            single = os.path.join(tmp, "single.py")
            with open(single, "w") as f:
                f.write("# single\n")

            # Directory with one file
            sub = os.path.join(tmp, "pkg")
            os.makedirs(sub)
            with open(os.path.join(sub, "inside.py"), "w") as f:
                f.write("# inside\n")

            result = ingester._resolve_paths([single, sub])
            basenames = sorted(os.path.basename(p) for p in result)
            self.assertEqual(basenames, ["inside.py", "single.py"])


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------


class TestResolvePathsFiltering(unittest.TestCase):
    """_resolve_paths must obey extension and existence filters."""

    def test_resolve_paths_filters_unsupported_extensions(self):
        """Files with unsupported extensions (.xyz) are excluded."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)

            good = os.path.join(tmp, "good.py")
            bad = os.path.join(tmp, "bad.xyz")
            for path in (good, bad):
                with open(path, "w") as f:
                    f.write("content\n")

            result = ingester._resolve_paths([good, bad])
            basenames = [os.path.basename(p) for p in result]
            self.assertIn("good.py", basenames)
            self.assertNotIn("bad.xyz", basenames)

    def test_resolve_paths_skips_nonexistent(self):
        """Non-existent paths are silently skipped -- no error."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)
            result = ingester._resolve_paths(["does_not_exist.py"])
            self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestResolvePathsDedup(unittest.TestCase):
    """_resolve_paths must deduplicate the same file reached via different paths."""

    def test_resolve_paths_deduplicates(self):
        """Same file via relative and absolute paths yields one entry."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)

            target = os.path.join(tmp, "unique.py")
            with open(target, "w") as f:
                f.write("# unique\n")

            # One absolute, one relative (relative to project root)
            rel_path = os.path.relpath(target, tmp)
            result = ingester._resolve_paths([target, rel_path])
            self.assertEqual(len(result), 1, f"Expected 1 entry, got {result}")


# ---------------------------------------------------------------------------
# _scan_dir refactor
# ---------------------------------------------------------------------------


class TestScanDirRefactor(unittest.TestCase):
    """_scan_files must delegate to _scan_dir(self.root) after the refactor."""

    def test_scan_files_uses_scan_dir(self):
        """_scan_files() calls _scan_dir(self.root) exactly once."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            ingester = _build_ingester(tmp)

            # Create a file so the scan has something to find
            with open(os.path.join(tmp, "probe.py"), "w") as f:
                f.write("# probe\n")

            with patch.object(ingester, "_scan_dir", wraps=ingester._scan_dir) as mock_scan:
                files = ingester._scan_files()
                mock_scan.assert_called_once_with(ingester.root)
                # The result should still contain our probe file
                basenames = [os.path.basename(p) for p in files]
                self.assertIn("probe.py", basenames)


if __name__ == "__main__":
    unittest.main()
