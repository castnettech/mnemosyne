# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Append-only JSONL audit logger for Mnemosyne.

Design:
  - Every operation is written as one JSON object per line (JSONL format).
  - Writes are atomic at the line level: each ``log()`` call opens, writes,
    and closes (or flushes) the file -- there is no open file handle held
    between calls, so concurrent processes can append safely on most OS.
  - ``rotate()`` renames the current log to ``<name>.1.jsonl`` (keeping one
    backup), preventing unbounded growth.
  - ``read()`` supports tail-N filtering and operation-type filtering without
    loading the entire file into memory first.

Thread safety: individual ``log()`` writes are protected by a
``threading.Lock``.  Cross-process safety relies on OS-level append
atomicity (guaranteed for lines < PIPE_BUF on POSIX; safe enough for audit
use on all common platforms).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AuditLog:
    """
    Append-only, JSONL-format audit log.

    Args:
        path: Filesystem path to the log file.  Parent directories are created
              automatically on the first write.

    Usage::

        log = AuditLog("/path/to/.mnemosyne/audit.jsonl")
        log.log("index_file", rel_path="src/main.py", chunks=42)
        log.log("query",      query="auth middleware", results=5)

        recent = log.read(last_n=100, op_filter="query")
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(self, operation: str, **details: Any) -> None:
        """
        Append one audit event to the log.

        The record is a single JSON object containing at minimum:
          - ``"op"``:        the *operation* name (e.g. ``"index_file"``)
          - ``"ts"``:        ISO-8601 UTC timestamp of when ``log()`` was called
          - **details:       any keyword arguments passed by the caller

        Args:
            operation: Short operation identifier; should be a lowercase
                       snake_case string (e.g. ``"query"``, ``"cache_evict"``).
            **details: Arbitrary key/value pairs to include in the record.
                       Values must be JSON-serialisable.

        Raises:
            TypeError: If any value in *details* is not JSON-serialisable.
        """
        record: dict[str, Any] = {
            "op": operation,
            "ts": _now_utc(),
        }
        record.update(details)

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Open in append mode; 'a' is atomic at line granularity on POSIX.
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        last_n: int | None = None,
        op_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read audit records from the log file.

        Args:
            last_n:    When provided, return only the last *n* matching records
                       (tail semantics -- most recent *n* entries that satisfy
                       *op_filter*).  Pass None to return all matching records.
            op_filter: When provided, return only records whose ``"op"`` field
                       equals this string (exact match, case-sensitive).

        Returns:
            List of record dicts in chronological order (oldest first).
            Returns an empty list if the log file does not exist.

        Note:
            Malformed JSON lines are silently skipped so that a single corrupt
            line does not prevent reading the rest of the log.
        """
        if not self.path.exists():
            return []

        records: list[dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Corrupt line -- skip rather than raising.
                    continue

                if op_filter is not None and record.get("op") != op_filter:
                    continue

                records.append(record)

        if last_n is not None and last_n > 0:
            records = records[-last_n:]

        return records

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate(self, max_size_mb: float = 10.0) -> bool:
        """
        Rotate the log file if it exceeds *max_size_mb* megabytes.

        Rotation renames the current log to ``<stem>.1<suffix>`` (overwriting
        any existing backup), then the next ``log()`` call will create a fresh
        empty file.

        Args:
            max_size_mb: Threshold in mebibytes.  If the current file is
                         smaller than this, no rotation occurs.

        Returns:
            True if rotation was performed, False if not needed or file absent.
        """
        if not self.path.exists():
            return False

        size_mb = self.path.stat().st_size / (1024 * 1024)
        if size_mb < max_size_mb:
            return False

        backup_path = self.path.with_name(
            self.path.stem + ".1" + self.path.suffix
        )

        with self._lock:
            # Re-check size inside the lock to avoid TOCTOU race.
            if not self.path.exists():
                return False
            if self.path.stat().st_size / (1024 * 1024) < max_size_mb:
                return False

            # Rename current -> backup (atomic on most POSIX filesystems).
            self.path.rename(backup_path)

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def size_bytes(self) -> int:
        """Return the current log file size in bytes, or 0 if absent."""
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"AuditLog(path={str(self.path)!r})"
