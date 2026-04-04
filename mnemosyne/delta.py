# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
File change detection and delta context for Mnemosyne.

Provides:
- ``DeltaTracker`` -- scan for modified/added/deleted files and compute diffs
  between the current state on disk and what was previously indexed.

Diff output uses Python's standard ``difflib.unified_diff`` so that deltas
are familiar to both humans and LLM consumers.
"""

from __future__ import annotations

import difflib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.store import Store


class DeltaTracker:
    """
    Detects file changes against the indexed state and produces unified diffs.

    Also maintains a per-session memory of which chunk content was last sent
    to the LLM, enabling incremental diff delivery (send only what changed
    since the LLM last saw the chunk).

    Args:
        store: The persistent :class:`~mnemosyne.store.Store` instance.
    """

    def __init__(self, store: "Store") -> None:
        self.store = store
        # chunk_id -> content string as of the last retrieval in this session
        self._session_state: dict[int, str] = {}

    # ------------------------------------------------------------------
    # File-level change detection
    # ------------------------------------------------------------------

    def detect_changes(self, project_root: str) -> list[dict]:
        """
        Scan *project_root* and detect what changed since the last index run.

        Compares ``mtime`` and content hash of every file visible on disk
        against the ``FileRecord`` stored in the database.

        Returns:
            List of change dicts, each with keys:

            * ``"file"``   -- relative path from project_root.
            * ``"status"`` -- ``"modified"``, ``"added"``, or ``"deleted"``.
            * ``"diff"``   -- unified diff string for modified files; empty
                             string for added/deleted.
            * ``"file_id"`` -- database file_id (or ``None`` for new files).
        """
        root = os.path.abspath(project_root)
        changes: list[dict] = []

        # Build a lookup of all known files: rel_path -> FileRecord
        known_files: dict[str, object] = {}
        for file_record in self.store.get_all_file_records():
            known_files[file_record.rel_path] = file_record

        seen_rel_paths: set[str] = set()

        # Walk the filesystem
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories and common noise dirs in-place
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and d not in ("__pycache__", "node_modules")
            ]
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root).replace(os.sep, "/")
                seen_rel_paths.add(rel_path)

                file_record = known_files.get(rel_path)

                try:
                    mtime = os.path.getmtime(abs_path)
                except OSError:
                    continue

                if file_record is None:
                    # New file not yet in the index
                    changes.append({
                        "file": rel_path,
                        "status": "added",
                        "diff": "",
                        "file_id": None,
                    })
                    continue

                # Compare mtime first (cheap), then hash (expensive)
                if mtime != file_record.last_modified:
                    diff_text = ""
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                            current_content = fh.read()

                        # Retrieve previous content from store (best-effort)
                        old_content = self.store.get_file_content(file_record.file_id) or ""
                        if old_content != current_content:
                            diff_text = self.compute_diff(old_content, current_content, rel_path)
                            changes.append({
                                "file": rel_path,
                                "status": "modified",
                                "diff": diff_text,
                                "file_id": file_record.file_id,
                            })
                    except OSError:
                        pass

        # Deleted files
        for rel_path, file_record in known_files.items():
            if rel_path not in seen_rel_paths and not file_record.is_deleted:
                changes.append({
                    "file": rel_path,
                    "status": "deleted",
                    "diff": "",
                    "file_id": file_record.file_id,
                })

        return changes

    # ------------------------------------------------------------------
    # Diff computation
    # ------------------------------------------------------------------

    def compute_diff(self, old_content: str, new_content: str, file_path: str) -> str:
        """
        Compute a unified diff between *old_content* and *new_content*.

        Args:
            old_content: Previous file text.
            new_content: Current file text.
            file_path:   Displayed path label in the diff header.

        Returns:
            A unified diff string, or an empty string when the contents
            are identical.
        """
        return "\n".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )

    # ------------------------------------------------------------------
    # Session-level chunk delta tracking
    # ------------------------------------------------------------------

    def mark_retrieved(self, chunk_id: int, content: str) -> None:
        """
        Record that *chunk_id* with *content* was sent to the LLM this session.

        Subsequent calls to :meth:`get_delta_context` will produce a diff
        if the content has changed.

        Args:
            chunk_id: The chunk that was delivered.
            content:  The exact content string that was sent.
        """
        self._session_state[chunk_id] = content

    def get_delta_context(self, chunk_id: int, current_content: str) -> str | None:
        """
        Return a diff if *chunk_id* was previously sent and has since changed.

        Args:
            chunk_id:        The chunk to check.
            current_content: The current content of the chunk.

        Returns:
            A unified diff string if the chunk changed since last delivery,
            or ``None`` if the chunk was not seen in this session or is
            unchanged.
        """
        prev = self._session_state.get(chunk_id)
        if prev is None or prev == current_content:
            return None
        return self.compute_diff(prev, current_content, f"chunk:{chunk_id}")

    # ------------------------------------------------------------------
    # Chunk-level impact analysis
    # ------------------------------------------------------------------

    def get_affected_chunks(self, file_id: int, diff_text: str) -> list[int]:
        """
        Identify which chunk IDs are affected by changes described in *diff_text*.

        Parses the ``@@`` hunk headers of a unified diff to extract the
        changed line ranges, then queries the store for chunks whose
        ``[line_start, line_end]`` intervals overlap with those ranges.

        Args:
            file_id:   The file whose chunks should be checked.
            diff_text: A unified diff string (from :meth:`compute_diff`).

        Returns:
            Sorted list of chunk IDs that overlap with the changed lines.
        """
        # Parse hunk headers: @@ -L,N +L,N @@ ...
        import re
        changed_ranges: list[tuple[int, int]] = []
        for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff_text, re.M):
            start_line = int(m.group(1))
            length = int(m.group(2)) if m.group(2) is not None else 1
            end_line = start_line + max(0, length - 1)
            changed_ranges.append((start_line, end_line))

        if not changed_ranges:
            return []

        # Get all chunks for this file
        chunks = self.store.get_chunks_for_file(file_id)
        affected: set[int] = set()

        for chunk in chunks:
            if chunk.chunk_id is None:
                continue
            for range_start, range_end in changed_ranges:
                # Overlap: not (chunk ends before range starts, or starts after range ends)
                if not (chunk.line_end < range_start or chunk.line_start > range_end):
                    affected.add(chunk.chunk_id)
                    break

        return sorted(affected)
