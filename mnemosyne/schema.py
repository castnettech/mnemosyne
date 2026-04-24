# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SQLite schema manager for Mnemosyne.

Responsibilities:
  - ``get_connection``  -- open / configure a SQLite connection (WAL, pragmas)
  - ``init_db``         -- idempotently create all tables, indexes, triggers, FTS5
  - ``migrate``         -- apply incremental schema upgrades keyed on schema_version

Schema version history:
  1  -- initial schema (all tables defined here)
  2  -- files.source_type column
  3  -- extraction metadata + chunks.page_number
  4  -- document partition (doc_chunks, doc_chunks_fts, doc_sparse_embeddings,
       doc_vocabulary)
  5  -- doc_embeddings table for the dense 128-dim hashed-TFIDF lane
       (hybrid retrieval + lightweight rerank, Wave 1D)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

CURRENT_SCHEMA_VERSION: Final[int] = 5

# ---------------------------------------------------------------------------
# DDL strings
# ---------------------------------------------------------------------------

# Each string is a standalone statement; we execute them one at a time so that
# if one fails the error message is unambiguous.

_DDL_STATEMENTS: list[str] = [
    # ------------------------------------------------------------------
    # files -- one row per tracked source file
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS files (
        file_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        rel_path      TEXT    NOT NULL UNIQUE,
        content_hash  TEXT    NOT NULL,
        size_bytes    INTEGER NOT NULL DEFAULT 0,
        language      TEXT,
        last_modified REAL    NOT NULL DEFAULT 0.0,
        last_indexed  TEXT,
        is_deleted    INTEGER NOT NULL DEFAULT 0,  -- BOOL (0/1)
        source_type   TEXT    DEFAULT 'file',      -- 'file' | 'schema' | 'document' | 'config_snapshot'
        extraction_method  TEXT,                    -- 'direct' | 'ocr_tesseract' | 'ocr_doctr'
        extraction_quality TEXT,                    -- 'good' | 'poor' | 'failed'
        page_count    INTEGER                      -- total pages for documents
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_files_rel_path    ON files (rel_path)",
    "CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_files_is_deleted   ON files (is_deleted)",
    "CREATE INDEX IF NOT EXISTS idx_files_source_type  ON files (source_type)",

    # ------------------------------------------------------------------
    # chunks -- content slices extracted from files
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id           INTEGER NOT NULL REFERENCES files (file_id) ON DELETE CASCADE,
        content_hash      TEXT    NOT NULL,
        chunk_type        TEXT    NOT NULL DEFAULT 'generic',
        line_start        INTEGER NOT NULL DEFAULT 0,
        line_end          INTEGER NOT NULL DEFAULT 0,
        token_count       INTEGER NOT NULL DEFAULT 0,
        content           TEXT    NOT NULL DEFAULT '',
        compressed        TEXT,
        compression_ratio REAL,
        symbol_name       TEXT,
        parent_chunk_id   INTEGER REFERENCES chunks (chunk_id) ON DELETE SET NULL,
        page_number       INTEGER                  -- 1-based page for documents
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_file_id      ON chunks (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type   ON chunks (chunk_type)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name  ON chunks (symbol_name)",

    # ------------------------------------------------------------------
    # chunks_fts -- FTS5 virtual table mirroring chunks.content
    # ------------------------------------------------------------------
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5 (
        content,
        tokenize = 'porter unicode61',
        content='chunks',
        content_rowid='chunk_id'
    )
    """,

    # Triggers keep chunks_fts in sync with chunks.
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ai
    AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_ad
    AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, content)
        VALUES ('delete', old.chunk_id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_fts_au
    AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts (chunks_fts, rowid, content)
        VALUES ('delete', old.chunk_id, old.content);
        INSERT INTO chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
    END
    """,

    # ------------------------------------------------------------------
    # embeddings -- dense vector storage (blob of float32 packed bytes)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id  INTEGER PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
        vector    BLOB    NOT NULL,  -- packed float32 little-endian bytes
        dim       INTEGER NOT NULL,
        model_tag TEXT    NOT NULL DEFAULT 'tfidf'
    )
    """,

    # ------------------------------------------------------------------
    # sparse_embeddings -- TF-IDF / BM25 term weights stored as JSON
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sparse_embeddings (
        chunk_id      INTEGER PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
        term_weights  TEXT    NOT NULL DEFAULT '{}',  -- JSON: {"term": weight, ...}
        updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ------------------------------------------------------------------
    # vocabulary -- global term statistics for IDF computation
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS vocabulary (
        term       TEXT    PRIMARY KEY,
        doc_freq   INTEGER NOT NULL DEFAULT 1,
        idf        REAL    NOT NULL DEFAULT 0.0,
        total_docs INTEGER NOT NULL DEFAULT 1
    )
    """,

    # ------------------------------------------------------------------
    # summaries -- multi-scope summaries (chunk / file / directory / project)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS summaries (
        summary_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        scope_type  TEXT    NOT NULL,          -- 'chunk'|'file'|'directory'|'project'
        scope_path  TEXT    NOT NULL,
        content     TEXT    NOT NULL DEFAULT '',
        token_count INTEGER NOT NULL DEFAULT 0,
        parent_id   INTEGER REFERENCES summaries (summary_id) ON DELETE SET NULL,
        version     INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_summaries_scope ON summaries (scope_type, scope_path)",
    "CREATE INDEX IF NOT EXISTS idx_summaries_parent ON summaries (parent_id)",

    # ------------------------------------------------------------------
    # usage_events -- query/chunk interaction log
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS usage_events (
        event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id    INTEGER NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
        query_text  TEXT,
        session_id  TEXT,
        event_type  TEXT    NOT NULL DEFAULT 'retrieved',  -- retrieved|selected|used|discarded
        timestamp   TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_chunk_id    ON usage_events (chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_usage_session_id  ON usage_events (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_usage_event_type  ON usage_events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_usage_timestamp   ON usage_events (timestamp)",

    # ------------------------------------------------------------------
    # cache_state -- persisted ARC cache tiers across sessions
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS cache_state (
        chunk_id      INTEGER PRIMARY KEY REFERENCES chunks (chunk_id) ON DELETE CASCADE,
        tier          TEXT    NOT NULL DEFAULT 'T1',  -- T1|T2|B1|B2
        access_count  INTEGER NOT NULL DEFAULT 0,
        last_accessed TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cache_tier ON cache_state (tier)",

    # ------------------------------------------------------------------
    # task_patterns -- query-signature -> chunk-id-list lookup
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS task_patterns (
        signature  TEXT    PRIMARY KEY,
        chunk_ids  TEXT    NOT NULL DEFAULT '[]',  -- JSON array of chunk_ids
        hit_count  INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ------------------------------------------------------------------
    # file_deltas -- change history between indexed versions of a file
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS file_deltas (
        delta_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL REFERENCES files (file_id) ON DELETE CASCADE,
        old_hash    TEXT    NOT NULL,
        new_hash    TEXT    NOT NULL,
        diff_text   TEXT    NOT NULL DEFAULT '',
        chunk_ids   TEXT    NOT NULL DEFAULT '[]',  -- JSON array of affected chunk_ids
        recorded_at TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_file_deltas_file_id     ON file_deltas (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_file_deltas_recorded_at ON file_deltas (recorded_at)",

    # ------------------------------------------------------------------
    # index_metadata -- key-value store for index configuration hashes
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS index_metadata (
        key         TEXT    PRIMARY KEY,
        value       TEXT    NOT NULL DEFAULT '',
        updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ==================================================================
    # DOCUMENT PARTITION -- isolated from code partition
    # ==================================================================

    # ------------------------------------------------------------------
    # doc_chunks -- document content slices (PDF, DOCX, CSV, plaintext)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS doc_chunks (
        chunk_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id           INTEGER NOT NULL REFERENCES files (file_id) ON DELETE CASCADE,
        content_hash      TEXT    NOT NULL,
        chunk_type        TEXT    NOT NULL DEFAULT 'paragraph',
        line_start        INTEGER NOT NULL DEFAULT 0,
        line_end          INTEGER NOT NULL DEFAULT 0,
        token_count       INTEGER NOT NULL DEFAULT 0,
        content           TEXT    NOT NULL DEFAULT '',
        compressed        TEXT,
        compression_ratio REAL,
        symbol_name       TEXT,
        parent_chunk_id   INTEGER REFERENCES doc_chunks (chunk_id) ON DELETE SET NULL,
        page_number       INTEGER
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doc_chunks_file_id      ON doc_chunks (file_id)",
    "CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_hash ON doc_chunks (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_type   ON doc_chunks (chunk_type)",

    # ------------------------------------------------------------------
    # doc_chunks_fts -- FTS5 for document content (separate from code FTS5)
    # ------------------------------------------------------------------
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts
    USING fts5 (
        content,
        tokenize = 'porter unicode61',
        content='doc_chunks',
        content_rowid='chunk_id'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_ai
    AFTER INSERT ON doc_chunks BEGIN
        INSERT INTO doc_chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_ad
    AFTER DELETE ON doc_chunks BEGIN
        INSERT INTO doc_chunks_fts (doc_chunks_fts, rowid, content)
        VALUES ('delete', old.chunk_id, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_au
    AFTER UPDATE ON doc_chunks BEGIN
        INSERT INTO doc_chunks_fts (doc_chunks_fts, rowid, content)
        VALUES ('delete', old.chunk_id, old.content);
        INSERT INTO doc_chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
    END
    """,

    # ------------------------------------------------------------------
    # doc_sparse_embeddings -- TF-IDF for documents (isolated IDF)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS doc_sparse_embeddings (
        chunk_id      INTEGER PRIMARY KEY REFERENCES doc_chunks (chunk_id) ON DELETE CASCADE,
        term_weights  TEXT    NOT NULL DEFAULT '{}',
        updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ------------------------------------------------------------------
    # doc_vocabulary -- document term statistics (isolated from code IDF)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS doc_vocabulary (
        term       TEXT    PRIMARY KEY,
        doc_freq   INTEGER NOT NULL DEFAULT 1,
        idf        REAL    NOT NULL DEFAULT 0.0,
        total_docs INTEGER NOT NULL DEFAULT 1
    )
    """,

    # ------------------------------------------------------------------
    # doc_embeddings -- dense lightweight vectors for doc chunks.
    #
    # The hashed_tfidf_v1 backend stores 128-dim int8 little-endian
    # bytes per chunk. A second row with a newer model_version can be
    # inserted without touching the old one; retrieval picks the
    # highest model_version with ``ORDER BY model_version DESC``.
    # model_id is kept alongside so a future ONNX dense backend can
    # live in the same table.
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS doc_embeddings (
        chunk_id       INTEGER NOT NULL REFERENCES doc_chunks (chunk_id) ON DELETE CASCADE,
        model_id       TEXT    NOT NULL DEFAULT 'hashed_tfidf_v1',
        model_version  INTEGER NOT NULL DEFAULT 1,
        dim            INTEGER NOT NULL DEFAULT 128,
        quantization   TEXT    NOT NULL DEFAULT 'int8',
        vector         BLOB    NOT NULL,
        updated_at     TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (chunk_id, model_id, model_version)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_chunk     ON doc_embeddings (chunk_id)",
    "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_model_ver ON doc_embeddings (model_id, model_version)",

    # ------------------------------------------------------------------
    # schema_version -- single-row version tracker for migrations
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER NOT NULL DEFAULT 1,
        applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
    )
    """,
]

# ---------------------------------------------------------------------------
# Migration registry
# ---------------------------------------------------------------------------
# Each entry is (from_version, to_version, list_of_sql_statements).
# Migrations are applied in ascending from_version order.

_MIGRATIONS: list[tuple[int, int, list[str]]] = [
    (1, 2, [
        "ALTER TABLE files ADD COLUMN source_type TEXT DEFAULT 'file'",
        "CREATE INDEX IF NOT EXISTS idx_files_source_type ON files (source_type)",
    ]),
    (2, 3, [
        "ALTER TABLE files ADD COLUMN extraction_method TEXT",
        "ALTER TABLE files ADD COLUMN extraction_quality TEXT",
        "ALTER TABLE files ADD COLUMN page_count INTEGER",
        "ALTER TABLE chunks ADD COLUMN page_number INTEGER",
    ]),
    (3, 4, [
        # Document partition tables
        """CREATE TABLE IF NOT EXISTS doc_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files (file_id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            chunk_type TEXT NOT NULL DEFAULT 'paragraph',
            line_start INTEGER NOT NULL DEFAULT 0,
            line_end INTEGER NOT NULL DEFAULT 0,
            token_count INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL DEFAULT '',
            compressed TEXT,
            compression_ratio REAL,
            symbol_name TEXT,
            parent_chunk_id INTEGER REFERENCES doc_chunks (chunk_id) ON DELETE SET NULL,
            page_number INTEGER
        )""",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_file_id ON doc_chunks (file_id)",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_hash ON doc_chunks (content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk_type ON doc_chunks (chunk_type)",
        # FTS5 for documents
        """CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts
        USING fts5 (content, tokenize = 'porter unicode61',
        content='doc_chunks', content_rowid='chunk_id')""",
        """CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_ai
        AFTER INSERT ON doc_chunks BEGIN
            INSERT INTO doc_chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
        END""",
        """CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_ad
        AFTER DELETE ON doc_chunks BEGIN
            INSERT INTO doc_chunks_fts (doc_chunks_fts, rowid, content)
            VALUES ('delete', old.chunk_id, old.content);
        END""",
        """CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_au
        AFTER UPDATE ON doc_chunks BEGIN
            INSERT INTO doc_chunks_fts (doc_chunks_fts, rowid, content)
            VALUES ('delete', old.chunk_id, old.content);
            INSERT INTO doc_chunks_fts (rowid, content) VALUES (new.chunk_id, new.content);
        END""",
        # Document sparse embeddings + vocabulary
        """CREATE TABLE IF NOT EXISTS doc_sparse_embeddings (
            chunk_id INTEGER PRIMARY KEY REFERENCES doc_chunks (chunk_id) ON DELETE CASCADE,
            term_weights TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""",
        """CREATE TABLE IF NOT EXISTS doc_vocabulary (
            term TEXT PRIMARY KEY,
            doc_freq INTEGER NOT NULL DEFAULT 1,
            idf REAL NOT NULL DEFAULT 0.0,
            total_docs INTEGER NOT NULL DEFAULT 1
        )""",
    ]),
    (4, 5, [
        # doc_embeddings -- dense 128-dim int8 hashed_tfidf_v1 vectors
        # for document chunks.  Enables the dense lane + lightweight
        # rerank in DocRetrievalEngine.  Composite PK lets a newer
        # model_version coexist with the old one; readers pick the
        # highest model_version at query time.
        """CREATE TABLE IF NOT EXISTS doc_embeddings (
            chunk_id INTEGER NOT NULL REFERENCES doc_chunks (chunk_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL DEFAULT 'hashed_tfidf_v1',
            model_version INTEGER NOT NULL DEFAULT 1,
            dim INTEGER NOT NULL DEFAULT 128,
            quantization TEXT NOT NULL DEFAULT 'int8',
            vector BLOB NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (chunk_id, model_id, model_version)
        )""",
        "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_chunk     ON doc_embeddings (chunk_id)",
        "CREATE INDEX IF NOT EXISTS idx_doc_embeddings_model_ver ON doc_embeddings (model_id, model_version)",
    ]),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a SQLite connection to *db_path* with recommended performance pragmas.

    Settings applied:
    - ``PRAGMA journal_mode = WAL``     -- concurrent readers, no read-lock on writes
    - ``PRAGMA synchronous = NORMAL``   -- safe compromise between fsync and speed
    - ``PRAGMA busy_timeout = 5000``    -- wait up to 5 s on lock contention
    - ``PRAGMA cache_size = -65536``    -- 64 MB page cache (negative = KiB)
    - ``PRAGMA mmap_size = 268435456``  -- 256 MB memory-mapped I/O
    - ``PRAGMA foreign_keys = ON``      -- enforce referential integrity
    - ``PRAGMA temp_store = MEMORY``    -- temp tables stay in RAM
    - ``row_factory = sqlite3.Row``     -- column-name access on result rows

    Args:
        db_path: Filesystem path to the SQLite database file. Parent directories
                 must already exist (use :func:`init_db` to ensure this).

    Returns:
        An open :class:`sqlite3.Connection`.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    pragmas = [
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
        "PRAGMA busy_timeout = 5000",       # 5 s wait on lock contention
        "PRAGMA cache_size = -65536",       # 64 MB
        "PRAGMA mmap_size = 268435456",     # 256 MB
        "PRAGMA foreign_keys = ON",
        "PRAGMA temp_store = MEMORY",
    ]
    for pragma in pragmas:
        conn.execute(pragma)
    conn.commit()
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create all Mnemosyne tables, indexes, triggers, and FTS5 virtual tables.

    This function is **idempotent** -- it uses ``CREATE TABLE IF NOT EXISTS``
    and ``CREATE INDEX IF NOT EXISTS`` throughout, so it is safe to call on an
    already-initialised database.

    After creating the schema, if ``schema_version`` is empty it inserts the
    initial version row and then calls :func:`migrate` to apply any pending
    upgrades.

    Args:
        conn: An open :class:`sqlite3.Connection` (ideally from
              :func:`get_connection`).
    """
    cur = conn.cursor()

    # Execute each DDL statement individually for clear error attribution.
    for stmt in _DDL_STATEMENTS:
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)

    # Seed the version row if the table is empty.
    row = cur.execute("SELECT COUNT(*) FROM schema_version").fetchone()
    if row[0] == 0:
        cur.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (CURRENT_SCHEMA_VERSION,),
        )

    conn.commit()

    # Apply any pending migrations.
    migrate(conn)


def migrate(conn: sqlite3.Connection) -> None:
    """
    Inspect the current ``schema_version`` and apply any pending migrations.

    Migrations are applied in order of their ``from_version`` number.
    Each migration runs inside its own transaction and bumps the version row
    on success.

    Args:
        conn: An open, initialised :class:`sqlite3.Connection`.
    """
    cur = conn.cursor()
    row = cur.execute(
        "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
    ).fetchone()

    if row is None:
        # schema_version table exists but is empty -- seed it.
        cur.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, datetime('now'))",
            (CURRENT_SCHEMA_VERSION,),
        )
        conn.commit()
        return

    current_version: int = row[0]

    for from_ver, to_ver, stmts in sorted(_MIGRATIONS, key=lambda m: m[0]):
        if current_version < from_ver:
            break  # Gap in migration chain -- stop safely.
        if current_version >= to_ver:
            continue  # Already applied.

        # Apply migration in a transaction.
        try:
            for stmt in stmts:
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.execute(
                "UPDATE schema_version SET version = ?, applied_at = datetime('now')",
                (to_ver,),
            )
            conn.commit()
            current_version = to_ver
        except sqlite3.Error:
            conn.rollback()
            raise

    # If any migration was applied, flag for one-time CLI upgrade hint.
    if current_version > row[0]:
        try:
            conn.execute(
                "INSERT INTO index_metadata (key, value, updated_at) "
                "VALUES ('upgrade_hint_pending', ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (str(current_version),),
            )
            conn.commit()
        except sqlite3.Error:
            pass  # Non-critical -- hint is cosmetic


def open_store(db_dir: str | Path) -> sqlite3.Connection:
    """
    Convenience helper: ensure *db_dir* exists, open the Mnemosyne database
    inside it, initialise the schema, and return the connection.

    The database file will be at ``<db_dir>/mnemosyne.db``.

    Args:
        db_dir: Directory that will contain the database file (created if absent).

    Returns:
        A fully configured and initialised :class:`sqlite3.Connection`.
    """
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_dir / "mnemosyne.db")
    init_db(conn)
    return conn
