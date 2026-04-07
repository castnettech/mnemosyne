# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document partition store for Mnemosyne.

Targets the ``doc_chunks``, ``doc_chunks_fts``, ``doc_sparse_embeddings``,
and ``doc_vocabulary`` tables.  Shares the same SQLite connection and
``files`` table as the code :class:`~mnemosyne.store.Store`.

This is intentionally NOT a subclass of Store.  The code Store has 49
methods tuned for code retrieval.  DocStore has ~12 methods tuned for
document content.  Separate classes, same database, zero coupling.
"""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from mnemosyne.models import Chunk, estimate_tokens

if TYPE_CHECKING:
    pass


class DocStore:
    """Document partition CRUD operations.

    Args:
        conn: An open SQLite connection (same DB as the code Store).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert a document chunk and return its chunk_id."""
        sql = """
            INSERT INTO doc_chunks
                (file_id, content_hash, chunk_type, line_start, line_end,
                 token_count, content, compressed, compression_ratio,
                 symbol_name, parent_chunk_id, page_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.conn:
            cur = self.conn.execute(sql, (
                chunk.file_id, chunk.content_hash, chunk.chunk_type,
                chunk.line_start, chunk.line_end, chunk.token_count,
                chunk.content, chunk.compressed, chunk.compression_ratio,
                chunk.symbol_name, chunk.parent_chunk_id, chunk.page_number,
            ))
        return cur.lastrowid  # type: ignore[return-value]

    def get_chunk(self, chunk_id: int) -> Chunk | None:
        """Return a document chunk by ID, or None."""
        row = self.conn.execute(
            "SELECT * FROM doc_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunk_by_hash(self, content_hash: str) -> Chunk | None:
        """Return first document chunk with *content_hash*, or None."""
        row = self.conn.execute(
            "SELECT * FROM doc_chunks WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_for_file(self, file_id: int) -> list[Chunk]:
        """Return all document chunks for *file_id*, ordered by line_start."""
        rows = self.conn.execute(
            "SELECT * FROM doc_chunks WHERE file_id = ? ORDER BY line_start",
            (file_id,),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def delete_chunks_for_file(self, file_id: int) -> None:
        """Delete all document chunks for *file_id*. FK cascades handle embeddings."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM doc_chunks WHERE file_id = ?", (file_id,)
            )

    def count_chunks(self) -> int:
        """Return total document chunk count."""
        row = self.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()
        return row[0]

    def total_tokens(self) -> int:
        """Return sum of token_count across all document chunks."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM doc_chunks"
        ).fetchone()
        return row[0]

    def chunk_type_counts(self) -> dict[str, int]:
        """Return {chunk_type: count} for document chunks."""
        rows = self.conn.execute(
            "SELECT chunk_type, COUNT(*) FROM doc_chunks GROUP BY chunk_type"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------
    # FTS5 search
    # ------------------------------------------------------------------

    def search_fts(self, query: str, limit: int = 60) -> list[tuple[int, float]]:
        """BM25 full-text search on the document partition.

        Returns:
            List of ``(chunk_id, bm25_score)`` pairs, best first.
        """
        if not query.strip():
            return []
        try:
            rows = self.conn.execute(
                "SELECT rowid, rank FROM doc_chunks_fts "
                "WHERE doc_chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), float(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Sparse embeddings (TF-IDF)
    # ------------------------------------------------------------------

    def insert_sparse_embedding(self, chunk_id: int, term_weights: dict) -> None:
        """Insert or replace TF-IDF term weights for a document chunk."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO doc_sparse_embeddings "
                "(chunk_id, term_weights, updated_at) "
                "VALUES (?, ?, datetime('now'))",
                (chunk_id, json.dumps(term_weights)),
            )

    def get_sparse_embedding(self, chunk_id: int) -> dict | None:
        """Return TF-IDF term weights for a document chunk, or None."""
        row = self.conn.execute(
            "SELECT term_weights FROM doc_sparse_embeddings WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def get_all_sparse_embeddings(self) -> list[tuple[int, dict]]:
        """Return all (chunk_id, term_weights) pairs from the doc partition."""
        rows = self.conn.execute(
            "SELECT chunk_id, term_weights FROM doc_sparse_embeddings"
        ).fetchall()
        return [(int(r[0]), json.loads(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Vocabulary (isolated IDF)
    # ------------------------------------------------------------------

    def update_vocabulary(self, vocab: dict[str, tuple[int, float, int]]) -> None:
        """Bulk-update the document vocabulary table.

        Args:
            vocab: ``{term: (doc_freq, idf, total_docs)}``
        """
        with self.conn:
            self.conn.execute("DELETE FROM doc_vocabulary")
            self.conn.executemany(
                "INSERT INTO doc_vocabulary (term, doc_freq, idf, total_docs) "
                "VALUES (?, ?, ?, ?)",
                [(t, df, idf, td) for t, (df, idf, td) in vocab.items()],
            )

    def get_vocabulary(self) -> dict[str, tuple[int, float, int]]:
        """Return the full document vocabulary as {term: (doc_freq, idf, total_docs)}."""
        rows = self.conn.execute(
            "SELECT term, doc_freq, idf, total_docs FROM doc_vocabulary"
        ).fetchall()
        return {r[0]: (r[1], float(r[2]), r[3]) for r in rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
