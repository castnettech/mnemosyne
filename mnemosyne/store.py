# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Repository layer for Mnemosyne -- all SQLite CRUD lives here.

``Store`` is the single access point for persisting and querying all domain
objects.  Every method accepts / returns the dataclasses defined in
``models.py``; no raw SQL leaks out of this module.

Design notes:
- All writes use explicit transactions (``with self.conn`` context manager).
- JSON fields (term_weights, chunk_ids) are encoded / decoded transparently.
- ``search_fts`` returns ``(chunk_id, rank)`` tuples; callers build QueryResult.
- Heavy reads (``get_all_sparse_embeddings``) are intentionally lazy -- call them
  only when the full corpus is needed for IDF / BM25 recomputation.
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from mnemosyne.models import (
    CacheEntry,
    Chunk,
    FileRecord,
    UsageEvent,
    estimate_tokens,
)


def _now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    keys = row.keys()
    return FileRecord(
        file_id=row["file_id"],
        rel_path=row["rel_path"],
        content_hash=row["content_hash"],
        size_bytes=row["size_bytes"],
        language=row["language"],
        last_modified=row["last_modified"],
        last_indexed=row["last_indexed"],
        is_deleted=bool(row["is_deleted"]),
        source_type=row["source_type"] if "source_type" in keys else "file",
        extraction_method=row["extraction_method"] if "extraction_method" in keys else None,
        extraction_quality=row["extraction_quality"] if "extraction_quality" in keys else None,
        page_count=row["page_count"] if "page_count" in keys else None,
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    keys = row.keys()
    return Chunk(
        chunk_id=row["chunk_id"],
        file_id=row["file_id"],
        content_hash=row["content_hash"],
        chunk_type=row["chunk_type"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        token_count=row["token_count"],
        content=row["content"],
        compressed=row["compressed"],
        compression_ratio=row["compression_ratio"],
        symbol_name=row["symbol_name"],
        parent_chunk_id=row["parent_chunk_id"],
        page_number=row["page_number"] if "page_number" in keys else None,
    )


def _db_path_from_conn(conn: sqlite3.Connection) -> Path | None:
    """
    Extract the filesystem path of the main database from a connection.

    Returns ``None`` for ``:memory:`` or unnamed temporary databases.
    """
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    # row columns: (seq, name, file).  file is empty for :memory:.
    db_file = row[2] if len(row) >= 3 else row["file"]
    if not db_file:
        return None
    return Path(db_file)


class Store:
    """
    Repository providing typed access to all Mnemosyne database tables.

    Args:
        conn: An open :class:`sqlite3.Connection`, typically obtained from
              :func:`mnemosyne.schema.get_connection` after :func:`mnemosyne.schema.init_db`.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        # Derive lock-file path from the database file path. For in-memory
        # databases (used in tests) the lock is a no-op.
        self._lock_path: Path | None = None
        db_path = _db_path_from_conn(conn)
        if db_path is not None:
            self._lock_path = db_path.parent / "mnemosyne.lock"

    @contextmanager
    def _with_write_lock(self) -> Iterator[None]:
        """
        Acquire a cross-process exclusive file lock before executing a write.

        The lock file lives alongside the database (``<db_dir>/mnemosyne.lock``).
        For in-memory databases the lock is a no-op so that tests are unaffected.
        """
        if self._lock_path is None:
            yield
            return
        # Ensure the lock file exists.
        self._lock_path.touch(exist_ok=True)
        fd = self._lock_path.open("r")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()

    # ======================================================================
    # File operations
    # ======================================================================

    def upsert_file(self, record: FileRecord) -> int:
        """
        Insert or update a file record.

        Uses ``INSERT OR REPLACE`` so an existing row with the same ``rel_path``
        is fully replaced (SQLite updates the rowid on replace, so dependent FK
        rows that CASCADE on DELETE will be removed -- callers should re-index
        chunks after an upsert when the content hash has changed).

        Returns:
            The ``file_id`` of the inserted / updated row.
        """
        sql = """
            INSERT INTO files
                (rel_path, content_hash, size_bytes, language,
                 last_modified, last_indexed, is_deleted,
                 source_type, extraction_method, extraction_quality, page_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rel_path) DO UPDATE SET
                content_hash       = excluded.content_hash,
                size_bytes         = excluded.size_bytes,
                language           = excluded.language,
                last_modified      = excluded.last_modified,
                last_indexed       = excluded.last_indexed,
                is_deleted         = excluded.is_deleted,
                source_type        = excluded.source_type,
                extraction_method  = excluded.extraction_method,
                extraction_quality = excluded.extraction_quality,
                page_count         = excluded.page_count
        """
        with self._with_write_lock():
            with self.conn:
                cur = self.conn.execute(
                    sql,
                    (
                        record.rel_path,
                        record.content_hash,
                        record.size_bytes,
                        record.language,
                        record.last_modified,
                        record.last_indexed,
                        int(record.is_deleted),
                        record.source_type,
                        record.extraction_method,
                        record.extraction_quality,
                        record.page_count,
                    ),
                )
                # For ON CONFLICT DO UPDATE, lastrowid may be 0; fetch the real id.
                row = self.conn.execute(
                    "SELECT file_id FROM files WHERE rel_path = ?", (record.rel_path,)
                ).fetchone()
        return row["file_id"]

    def get_file(self, rel_path: str) -> FileRecord | None:
        """Return the :class:`FileRecord` for *rel_path*, or None."""
        row = self.conn.execute(
            "SELECT * FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return _row_to_file(row) if row else None

    def get_file_by_id(self, file_id: int) -> FileRecord | None:
        """Return the :class:`FileRecord` for *file_id*, or None."""
        row = self.conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return _row_to_file(row) if row else None

    def list_files(self, include_deleted: bool = False) -> list[FileRecord]:
        """
        Return all tracked files.

        Args:
            include_deleted: When False (default) only live files are returned.
        """
        if include_deleted:
            rows = self.conn.execute("SELECT * FROM files").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM files WHERE is_deleted = 0"
            ).fetchall()
        return [_row_to_file(r) for r in rows]

    def mark_deleted(self, file_id: int) -> None:
        """Soft-delete *file_id* by setting ``is_deleted = 1``."""
        with self.conn:
            self.conn.execute(
                "UPDATE files SET is_deleted = 1 WHERE file_id = ?", (file_id,)
            )

    # ======================================================================
    # Chunk operations
    # ======================================================================

    def insert_chunk(self, chunk: Chunk) -> int:
        """
        Insert a single chunk and return its new ``chunk_id``.

        The FTS5 sync trigger fires automatically on INSERT.
        """
        sql = """
            INSERT INTO chunks
                (file_id, content_hash, chunk_type, line_start, line_end,
                 token_count, content, compressed, compression_ratio,
                 symbol_name, parent_chunk_id, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.conn:
            cur = self.conn.execute(
                sql,
                (
                    chunk.file_id,
                    chunk.content_hash,
                    chunk.chunk_type,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.token_count,
                    chunk.content,
                    chunk.compressed,
                    chunk.compression_ratio,
                    chunk.symbol_name,
                    chunk.parent_chunk_id,
                    chunk.page_number,
                ),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def insert_chunks(self, chunks: list[Chunk]) -> list[int]:
        """
        Bulk-insert a list of chunks in a single transaction.

        Returns:
            List of assigned ``chunk_id`` values, in the same order as *chunks*.
        """
        sql = """
            INSERT INTO chunks
                (file_id, content_hash, chunk_type, line_start, line_end,
                 token_count, content, compressed, compression_ratio,
                 symbol_name, parent_chunk_id, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        ids: list[int] = []
        with self._with_write_lock():
            with self.conn:
                for chunk in chunks:
                    cur = self.conn.execute(
                        sql,
                        (
                            chunk.file_id,
                            chunk.content_hash,
                            chunk.chunk_type,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.token_count,
                            chunk.content,
                            chunk.compressed,
                            chunk.compression_ratio,
                            chunk.symbol_name,
                            chunk.parent_chunk_id,
                            chunk.page_number,
                        ),
                    )
                    ids.append(cur.lastrowid)  # type: ignore[arg-type]
        return ids

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        """Return the :class:`Chunk` with *chunk_id*, or None."""
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_for_file(self, file_id: int) -> list[Chunk]:
        """Return all chunks belonging to *file_id*, ordered by ``line_start``."""
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE file_id = ? ORDER BY line_start", (file_id,)
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def delete_chunks_for_file(self, file_id: int) -> None:
        """
        Delete all chunks for *file_id*.

        Cascade deletes handle ``embeddings``, ``sparse_embeddings``,
        ``usage_events``, and ``cache_state`` rows via FK ON DELETE CASCADE.
        FTS5 sync triggers fire for each deleted chunk row.
        """
        with self._with_write_lock():
            with self.conn:
                self.conn.execute(
                    "DELETE FROM chunks WHERE file_id = ?", (file_id,)
                )

    def chunk_exists(
        self, content_hash: str, file_id: int, line_start: int
    ) -> bool:
        """
        Return True if a chunk with the given (content_hash, file_id, line_start)
        triple already exists.  Used for incremental re-indexing.
        """
        row = self.conn.execute(
            """
            SELECT 1 FROM chunks
            WHERE content_hash = ? AND file_id = ? AND line_start = ?
            LIMIT 1
            """,
            (content_hash, file_id, line_start),
        ).fetchone()
        return row is not None

    # ======================================================================
    # FTS5 search
    # ======================================================================

    def search_fts(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        """
        Full-text search over chunk content using the FTS5 index.

        The FTS5 ``rank`` column is negated so that higher scores sort first
        when we later combine with BM25/vector scores.

        Args:
            query: A plain-text or FTS5 query expression.
            limit: Maximum number of results.

        Returns:
            List of ``(chunk_id, score)`` tuples, ordered by descending score.
            ``score`` is the raw FTS5 rank (negative of BM25, so negative
            values; closer to zero = better match).
        """
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT rowid, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 query (e.g. unmatched quotes) -- degrade gracefully.
            return []
        return [(int(row[0]), float(row[1])) for row in rows]

    # ======================================================================
    # Sparse embeddings
    # ======================================================================

    def insert_sparse_embedding(self, chunk_id: int, terms: dict[str, float]) -> None:
        """Upsert the TF-IDF / BM25 term-weight mapping for *chunk_id*."""
        sql = """
            INSERT INTO sparse_embeddings (chunk_id, term_weights, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(chunk_id) DO UPDATE SET
                term_weights = excluded.term_weights,
                updated_at   = excluded.updated_at
        """
        with self.conn:
            self.conn.execute(sql, (chunk_id, json.dumps(terms, ensure_ascii=False)))

    def get_sparse_embedding(self, chunk_id: int) -> dict[str, float] | None:
        """Return the term-weight dict for *chunk_id*, or None."""
        row = self.conn.execute(
            "SELECT term_weights FROM sparse_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["term_weights"])

    def get_all_sparse_embeddings(self) -> list[tuple[int, dict[str, float]]]:
        """
        Return all stored sparse embeddings as ``(chunk_id, term_weights)`` pairs.

        This is an expensive full-table scan -- call only when recomputing the
        global IDF weights.
        """
        rows = self.conn.execute(
            "SELECT chunk_id, term_weights FROM sparse_embeddings"
        ).fetchall()
        return [(int(r["chunk_id"]), json.loads(r["term_weights"])) for r in rows]

    # ======================================================================
    # Dense embeddings
    # ======================================================================

    def insert_dense_embedding(
        self,
        chunk_id: int,
        vector: bytes,
        dim: int,
        model_tag: str = "minilm-l6-code",
    ) -> None:
        """Upsert a dense vector (packed float32 LE bytes) into the embeddings table."""
        with self._with_write_lock():
            with self.conn:
                self.conn.execute(
                    "INSERT INTO embeddings (chunk_id, vector, dim, model_tag) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET "
                    "vector=excluded.vector, dim=excluded.dim, model_tag=excluded.model_tag",
                    (chunk_id, vector, dim, model_tag),
                )

    def get_dense_embeddings_batch(self, chunk_ids: list[int]) -> dict[int, bytes]:
        """Return {chunk_id: packed_vector_bytes} for the given IDs.

        If *chunk_ids* is empty, returns an empty dict (use a direct query
        on the embeddings table for a full scan).
        """
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {int(r["chunk_id"]): r["vector"] for r in rows}

    # ======================================================================
    # Vocabulary
    # ======================================================================

    def update_vocabulary(
        self, term_freqs: dict[str, int], total_docs: int
    ) -> None:
        """
        Upsert global term-frequency statistics and recompute IDF values.

        ``IDF = log((total_docs + 1) / (doc_freq + 1)) + 1``  (smoothed).

        Args:
            term_freqs: Mapping of ``term -> document_frequency``.
            total_docs: Total number of indexed documents (used for IDF).
        """
        import math

        sql = """
            INSERT INTO vocabulary (term, doc_freq, idf, total_docs)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(term) DO UPDATE SET
                doc_freq   = excluded.doc_freq,
                idf        = excluded.idf,
                total_docs = excluded.total_docs
        """
        with self.conn:
            for term, freq in term_freqs.items():
                idf = math.log((total_docs + 1) / (freq + 1)) + 1.0
                self.conn.execute(sql, (term, freq, idf, total_docs))

    def get_vocabulary(self) -> dict[str, tuple[int, float]]:
        """
        Return the full vocabulary as ``{term: (doc_freq, idf)}``.

        Returns an empty dict if no vocabulary has been built yet.
        """
        rows = self.conn.execute(
            "SELECT term, doc_freq, idf FROM vocabulary"
        ).fetchall()
        return {r["term"]: (int(r["doc_freq"]), float(r["idf"])) for r in rows}

    # ======================================================================
    # Index metadata (key-value store for configuration hashes)
    # ======================================================================

    def set_index_metadata(self, key: str, value: str) -> None:
        """
        Store an index-metadata value under *key*.

        Upserts: existing keys are overwritten.
        """
        sql = """
            INSERT INTO index_metadata (key, value, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                updated_at = excluded.updated_at
        """
        with self.conn:
            self.conn.execute(sql, (key, value))

    def get_index_metadata(self, key: str) -> str | None:
        """
        Retrieve the metadata value for *key*, or ``None`` if absent.
        """
        row = self.conn.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    # ======================================================================
    # Usage events
    # ======================================================================

    def record_usage(self, event: UsageEvent) -> None:
        """Append a :class:`UsageEvent` to the usage_events table."""
        timestamp = event.timestamp or _now_utc()
        sql = """
            INSERT INTO usage_events
                (chunk_id, query_text, session_id, event_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """
        with self.conn:
            self.conn.execute(
                sql,
                (
                    event.chunk_id,
                    event.query_text,
                    event.session_id,
                    event.event_type,
                    timestamp,
                ),
            )

    def get_usage_scores(self) -> dict[int, tuple[int, str]]:
        """
        Return per-chunk usage counts and latest access timestamps.

        Returns:
            ``{chunk_id: (event_count, latest_timestamp)}`` for all chunks
            that have at least one usage event.
        """
        rows = self.conn.execute(
            """
            SELECT chunk_id,
                   COUNT(*)       AS event_count,
                   MAX(timestamp) AS latest_ts
            FROM usage_events
            GROUP BY chunk_id
            """
        ).fetchall()
        return {
            int(r["chunk_id"]): (int(r["event_count"]), r["latest_ts"])
            for r in rows
        }

    # ======================================================================
    # Cache state
    # ======================================================================

    def save_cache_state(self, entries: list[CacheEntry]) -> None:
        """
        Persist ARC cache state for the next session.

        Replaces the entire ``cache_state`` table contents with *entries*.
        """
        sql_insert = """
            INSERT INTO cache_state (chunk_id, tier, access_count, last_accessed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                tier          = excluded.tier,
                access_count  = excluded.access_count,
                last_accessed = excluded.last_accessed
        """
        with self.conn:
            # Remove stale entries not present in the new state.
            if entries:
                placeholders = ",".join("?" * len(entries))
                ids = [e.chunk_id for e in entries]
                self.conn.execute(
                    f"DELETE FROM cache_state WHERE chunk_id NOT IN ({placeholders})",
                    ids,
                )
            else:
                self.conn.execute("DELETE FROM cache_state")
            for entry in entries:
                self.conn.execute(
                    sql_insert,
                    (entry.chunk_id, entry.tier, entry.access_count, entry.last_accessed),
                )

    def load_cache_state(self) -> list[CacheEntry]:
        """Return all persisted :class:`CacheEntry` objects, or an empty list."""
        rows = self.conn.execute(
            "SELECT chunk_id, tier, access_count, last_accessed FROM cache_state"
        ).fetchall()
        return [
            CacheEntry(
                chunk_id=int(r["chunk_id"]),
                tier=r["tier"],
                access_count=int(r["access_count"]),
                last_accessed=r["last_accessed"],
            )
            for r in rows
        ]

    # ======================================================================
    # Task patterns
    # ======================================================================

    def get_pattern(self, signature: str) -> tuple[list[int], int] | None:
        """
        Look up a task pattern by its query signature.

        Returns:
            ``(chunk_ids, hit_count)`` if found, else None.
        """
        row = self.conn.execute(
            "SELECT chunk_ids, hit_count FROM task_patterns WHERE signature = ?",
            (signature,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["chunk_ids"]), int(row["hit_count"])

    def upsert_pattern(
        self, signature: str, chunk_ids: list[int], hit_count: int
    ) -> None:
        """Insert or update a task pattern with its associated chunk list."""
        sql = """
            INSERT INTO task_patterns (signature, chunk_ids, hit_count, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(signature) DO UPDATE SET
                chunk_ids  = excluded.chunk_ids,
                hit_count  = excluded.hit_count,
                updated_at = excluded.updated_at
        """
        with self.conn:
            self.conn.execute(sql, (signature, json.dumps(chunk_ids), hit_count))

    # ======================================================================
    # File deltas
    # ======================================================================

    def record_delta(
        self,
        file_id: int,
        old_hash: str,
        new_hash: str,
        diff_text: str,
        chunk_ids: list[int] | None = None,
    ) -> None:
        """
        Append a change record for *file_id*.

        Args:
            file_id:    The file that changed.
            old_hash:   Content hash before the change.
            new_hash:   Content hash after the change.
            diff_text:  Unified diff or change summary text.
            chunk_ids:  Chunk IDs affected by this change (may be empty / None).
        """
        sql = """
            INSERT INTO file_deltas
                (file_id, old_hash, new_hash, diff_text, chunk_ids, recorded_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """
        with self.conn:
            self.conn.execute(
                sql,
                (
                    file_id,
                    old_hash,
                    new_hash,
                    diff_text,
                    json.dumps(chunk_ids or []),
                ),
            )

    def get_recent_deltas(
        self, file_id: int, since: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Return delta records for *file_id*, optionally filtered by timestamp.

        Args:
            file_id: Which file to query.
            since:   ISO-8601 timestamp; only deltas recorded at or after this
                     time are returned.  Pass None for all deltas.

        Returns:
            List of dicts with keys:
            ``delta_id``, ``old_hash``, ``new_hash``, ``diff_text``,
            ``chunk_ids`` (as a list), ``recorded_at``.
        """
        if since:
            rows = self.conn.execute(
                """
                SELECT delta_id, old_hash, new_hash, diff_text, chunk_ids, recorded_at
                FROM file_deltas
                WHERE file_id = ? AND recorded_at >= ?
                ORDER BY recorded_at DESC
                """,
                (file_id, since),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT delta_id, old_hash, new_hash, diff_text, chunk_ids, recorded_at
                FROM file_deltas
                WHERE file_id = ?
                ORDER BY recorded_at DESC
                """,
                (file_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "delta_id": int(r["delta_id"]),
                    "old_hash": r["old_hash"],
                    "new_hash": r["new_hash"],
                    "diff_text": r["diff_text"],
                    "chunk_ids": json.loads(r["chunk_ids"]),
                    "recorded_at": r["recorded_at"],
                }
            )
        return results

    # ======================================================================
    # Stats
    # ======================================================================

    # ======================================================================
    # Convenience aliases and extended query methods
    # ======================================================================

    def get_file_record_by_path(self, rel_path: str) -> FileRecord | None:
        """Alias for :meth:`get_file` -- look up a FileRecord by relative path."""
        return self.get_file(rel_path)

    def get_file_record(self, file_id: int) -> FileRecord | None:
        """Alias for :meth:`get_file_by_id` -- look up a FileRecord by file_id."""
        return self.get_file_by_id(file_id)

    def get_all_file_records(self, include_deleted: bool = True) -> list[FileRecord]:
        """Return every FileRecord (live and deleted by default)."""
        return self.list_files(include_deleted=include_deleted)

    def get_deleted_file_ids(self) -> list[int]:
        """Return the file_ids of all soft-deleted files."""
        rows = self.conn.execute(
            "SELECT file_id FROM files WHERE is_deleted = 1"
        ).fetchall()
        return [int(r["file_id"]) for r in rows]

    def mark_file_deleted(self, file_id: int) -> None:
        """Alias for :meth:`mark_deleted`."""
        self.mark_deleted(file_id)

    def save_chunk(self, chunk: Chunk) -> int:
        """Alias for :meth:`insert_chunk` -- insert a chunk and return its id."""
        return self.insert_chunk(chunk)

    def get_chunk_by_hash(self, content_hash: str) -> Chunk | None:
        """
        Return the first chunk with *content_hash*, or None.

        Used for deduplication during ingestion: if a chunk with an identical
        content hash already exists in the index it does not need to be
        re-inserted.
        """
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_file_content(self, file_id: int) -> str | None:
        """
        Reconstruct the full text of *file_id* by concatenating its chunks
        in line order.

        Returns None if no chunks exist for the file.
        """
        chunks = self.get_chunks_for_file(file_id)
        if not chunks:
            return None
        return "\n".join(c.content for c in chunks)

    def save_usage_event(self, event: UsageEvent) -> None:
        """Alias for :meth:`record_usage`."""
        self.record_usage(event)

    def get_usage_events(
        self,
        event_types: list[str] | None = None,
    ) -> list[UsageEvent]:
        """
        Return UsageEvent rows, optionally filtered to *event_types*.

        Args:
            event_types: List of event_type strings to include.  Pass None
                         to return all event types.

        Returns:
            List of :class:`~mnemosyne.models.UsageEvent` objects.
        """
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            rows = self.conn.execute(
                f"""
                SELECT event_id, chunk_id, query_text, session_id,
                       event_type, timestamp
                FROM usage_events
                WHERE event_type IN ({placeholders})
                ORDER BY timestamp
                """,
                event_types,
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT event_id, chunk_id, query_text, session_id,
                       event_type, timestamp
                FROM usage_events
                ORDER BY timestamp
                """
            ).fetchall()

        return [
            UsageEvent(
                event_id=int(r["event_id"]),
                chunk_id=int(r["chunk_id"]),
                query_text=r["query_text"],
                session_id=r["session_id"],
                event_type=r["event_type"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def get_usage_events_for_chunk(self, chunk_id: int) -> list[UsageEvent]:
        """Return all UsageEvent rows for *chunk_id*."""
        rows = self.conn.execute(
            """
            SELECT event_id, chunk_id, query_text, session_id,
                   event_type, timestamp
            FROM usage_events
            WHERE chunk_id = ?
            ORDER BY timestamp
            """,
            (chunk_id,),
        ).fetchall()
        return [
            UsageEvent(
                event_id=int(r["event_id"]),
                chunk_id=int(r["chunk_id"]),
                query_text=r["query_text"],
                session_id=r["session_id"],
                event_type=r["event_type"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def get_usage_events_for_session(self, session_id: str) -> list[UsageEvent]:
        """Return all UsageEvent rows for *session_id*."""
        rows = self.conn.execute(
            """
            SELECT event_id, chunk_id, query_text, session_id,
                   event_type, timestamp
            FROM usage_events
            WHERE session_id = ?
            ORDER BY timestamp
            """,
            (session_id,),
        ).fetchall()
        return [
            UsageEvent(
                event_id=int(r["event_id"]),
                chunk_id=int(r["chunk_id"]),
                query_text=r["query_text"],
                session_id=r["session_id"],
                event_type=r["event_type"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ======================================================================
    # Stats / count helpers called by cli.py
    # ======================================================================

    def count_files(self, include_deleted: bool = False) -> int:
        """Return the number of tracked (non-deleted by default) files."""
        if include_deleted:
            row = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM files WHERE is_deleted = 0"
            ).fetchone()
        return int(row[0]) if row else 0

    def count_chunks(self) -> int:
        """Return the total number of chunks."""
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def total_tokens(self) -> int:
        """Return the sum of token_count across all chunks."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM chunks"
        ).fetchone()
        return int(row[0]) if row else 0

    def count_usage_events(self) -> int:
        """Return the total number of usage events."""
        row = self.conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()
        return int(row[0]) if row else 0

    def count_patterns(self) -> int:
        """Return the total number of task patterns."""
        row = self.conn.execute("SELECT COUNT(*) FROM task_patterns").fetchone()
        return int(row[0]) if row else 0

    def chunk_type_counts(self) -> dict[str, int]:
        """Return a mapping of chunk_type -> count."""
        rows = self.conn.execute(
            "SELECT chunk_type, COUNT(*) AS cnt FROM chunks GROUP BY chunk_type"
        ).fetchall()
        return {r["chunk_type"]: int(r["cnt"]) for r in rows}

    def language_counts(self) -> dict[str, int]:
        """Return a mapping of language -> file count (non-deleted files)."""
        rows = self.conn.execute(
            """
            SELECT language, COUNT(*) AS cnt
            FROM files
            WHERE is_deleted = 0
            GROUP BY language
            """
        ).fetchall()
        return {r["language"]: int(r["cnt"]) for r in rows}

    def get_cache_state_counts(self) -> dict[str, int]:
        """Return a mapping of ARC tier -> number of entries in cache_state."""
        rows = self.conn.execute(
            "SELECT tier, COUNT(*) AS cnt FROM cache_state GROUP BY tier"
        ).fetchall()
        return {r["tier"]: int(r["cnt"]) for r in rows}

    def clear_cache_state(self) -> None:
        """Delete all rows from cache_state."""
        with self.conn:
            self.conn.execute("DELETE FROM cache_state")

    def prune_cache_state(self) -> None:
        """
        Remove cache_state entries whose chunk_id no longer exists in chunks.

        This can happen after a GC pass where chunks were deleted but the
        cache_state table was not cleaned up yet.
        """
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM cache_state
                WHERE chunk_id NOT IN (
                    SELECT chunk_id FROM chunks
                    UNION ALL
                    SELECT chunk_id FROM doc_chunks
                )
                """
            )

    def prune_usage_events(self) -> None:
        """
        Remove usage_events rows whose chunk_id no longer exists in chunks.
        """
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM usage_events
                WHERE chunk_id NOT IN (
                    SELECT chunk_id FROM chunks
                    UNION ALL
                    SELECT chunk_id FROM doc_chunks
                )
                """
            )

    def get_top_accessed_chunks(self, limit: int = 100) -> list[Chunk]:
        """
        Return the *limit* most frequently accessed chunks based on
        usage_events, ordered by event count descending.

        Chunks with no usage events are excluded.
        """
        rows = self.conn.execute(
            """
            SELECT c.*, COUNT(u.event_id) AS event_count
            FROM chunks c
            JOIN usage_events u ON u.chunk_id = c.chunk_id
            GROUP BY c.chunk_id
            ORDER BY event_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    # ======================================================================
    # Vocabulary persistence (called by TFIDFBackend)
    # ======================================================================

    def save_vocabulary(self, payload: dict) -> None:
        """
        Persist a TF-IDF vocabulary payload.

        *payload* is a dict with keys ``"vocabulary"`` (term->df),
        ``"idf"`` (term->weight), and ``"total_docs"`` (int).
        Delegates to :meth:`update_vocabulary` using the stored doc-freq
        mapping and total_docs value.
        """
        vocab: dict[str, int] = payload.get("vocabulary", {})
        total_docs: int = payload.get("total_docs", 0)
        if vocab:
            self.update_vocabulary(vocab, total_docs)

    def load_vocabulary(self) -> dict | None:
        """
        Load the persisted TF-IDF vocabulary.

        Returns a dict with keys ``"vocabulary"`` (term->df), ``"idf"``
        (term->weight), and ``"total_docs"`` (int), or None if the
        vocabulary table is empty.
        """
        raw = self.get_vocabulary()
        if not raw:
            return None
        vocabulary: dict[str, int] = {}
        idf: dict[str, float] = {}
        total_docs: int = 0
        for term, (df, idf_val) in raw.items():
            vocabulary[term] = df
            idf[term] = idf_val
            total_docs = max(total_docs, df)
        # Recover total_docs from the vocabulary table directly
        row = self.conn.execute(
            "SELECT total_docs FROM vocabulary LIMIT 1"
        ).fetchone()
        if row:
            total_docs = int(row["total_docs"])
        return {"vocabulary": vocabulary, "idf": idf, "total_docs": total_docs}

    # ======================================================================
    # Stats
    # ======================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Return a summary of database contents.

        Returns:
            Dict with keys:
            ``files``, ``files_deleted``, ``chunks``, ``summaries``,
            ``usage_events``, ``vocabulary_size``, ``cache_entries``,
            ``task_patterns``, ``file_deltas``.
        """
        counts: dict[str, Any] = {}

        queries = {
            "files":          "SELECT COUNT(*) FROM files WHERE is_deleted = 0",
            "files_deleted":  "SELECT COUNT(*) FROM files WHERE is_deleted = 1",
            "chunks":         "SELECT COUNT(*) FROM chunks",
            "summaries":      "SELECT COUNT(*) FROM summaries",
            "usage_events":   "SELECT COUNT(*) FROM usage_events",
            "vocabulary_size": "SELECT COUNT(*) FROM vocabulary",
            "cache_entries":  "SELECT COUNT(*) FROM cache_state",
            "task_patterns":  "SELECT COUNT(*) FROM task_patterns",
            "file_deltas":    "SELECT COUNT(*) FROM file_deltas",
        }

        for key, sql in queries.items():
            row = self.conn.execute(sql).fetchone()
            counts[key] = int(row[0]) if row else 0

        # Token budget info
        row = self.conn.execute(
            "SELECT SUM(token_count) FROM chunks"
        ).fetchone()
        counts["total_tokens_indexed"] = int(row[0]) if row and row[0] else 0

        return counts
