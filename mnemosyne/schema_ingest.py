# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Schema ingestion orchestrator for Mnemosyne.

Reads database schema from DDL files, JSON snapshots, or YAML snapshots and
ingests them as first-class chunks in the Mnemosyne index.  Schema sources are
stored under synthetic ``__schema__/<env>/`` paths so they flow through the
standard retrieval pipeline (BM25, TF-IDF, symbol search, RRF fusion) without
any pipeline changes.

Security tiers:
  1 (default) -- DDL files only, no data values.
  2 (opt-in)  -- Config snapshots with automatic value redaction.
  3 (explicit) -- Live SQLite introspection (Phase 3, not yet implemented).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnemosyne.models import FileRecord, Chunk, estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.store import Store
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog
    from mnemosyne.config import Config


# ---------------------------------------------------------------------------
# Redaction patterns for Tier 2+ security
# ---------------------------------------------------------------------------

_REDACT_PATTERNS: list[re.Pattern[str]] = [
    # Email addresses
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # URLs with embedded credentials (user:pass@host)
    re.compile(r"://[^\s@]+@"),
    # Long base64-ish strings (likely API keys or tokens)
    re.compile(r"\b[A-Za-z0-9+/=]{40,}\b"),
    # Common secret prefixes
    re.compile(r"\b(?:sk|pk|api|key|token|secret|password)[-_]?[A-Za-z0-9]{16,}\b", re.I),
]


def _redact(text: str) -> tuple[str, int]:
    """Apply redaction patterns to *text*.

    Returns ``(redacted_text, redaction_count)``.
    """
    count = 0
    for pattern in _REDACT_PATTERNS:
        text, n = pattern.subn("[REDACTED]", text)
        count += n
    return text, count


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSON/YAML schema snapshot -> DDL text conversion
# ---------------------------------------------------------------------------


def _json_schema_to_ddl(data: dict[str, Any]) -> str:
    """Convert a JSON schema snapshot to DDL text for chunking.

    The JSON format is:
    ```json
    {
      "database": "myapp",
      "environment": "production",
      "tables": [
        {
          "name": "table_name",
          "columns": [
            {"name": "col", "type": "TEXT", "nullable": false, "default": "val",
             "primary_key": true}
          ],
          "indexes": ["idx_name"],
          "foreign_keys": [
            {"column": "user_id", "references": "users(id)"}
          ],
          "row_count": 42
        }
      ],
      "config_values": {
        "table_name": [
          {"col1": "val1", "col2": "val2"}
        ]
      }
    }
    ```
    """
    parts: list[str] = []
    db_name = data.get("database", "unknown")
    env = data.get("environment", "")

    if db_name or env:
        parts.append(f"-- Database: {db_name}")
        if env:
            parts.append(f"-- Environment: {env}")
        parts.append("")

    for table in data.get("tables", []):
        tname = table.get("name", "unknown")
        columns = table.get("columns", [])
        col_defs: list[str] = []

        for col in columns:
            cdef = f"    {col['name']} {col.get('type', 'TEXT')}"
            if col.get("primary_key"):
                cdef += " PRIMARY KEY"
            if col.get("nullable") is False:
                cdef += " NOT NULL"
            if "default" in col and col["default"] is not None:
                default_val = col["default"]
                if isinstance(default_val, str):
                    cdef += f" DEFAULT '{default_val}'"
                else:
                    cdef += f" DEFAULT {default_val}"
            col_defs.append(cdef)

        # Foreign keys as constraints
        for fk in table.get("foreign_keys", []):
            col_defs.append(
                f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['references']}"
            )

        parts.append(f"CREATE TABLE {tname} (")
        parts.append(",\n".join(col_defs))
        parts.append(");")

        # Row count as comment
        row_count = table.get("row_count")
        if row_count is not None:
            parts.append(f"-- Row count: {row_count}")

        # Indexes
        for idx in table.get("indexes", []):
            parts.append(f"CREATE INDEX {idx} ON {tname} (...);")

        parts.append("")

    # Config values as commented data (Tier 2+)
    config_values = data.get("config_values", {})
    if config_values:
        parts.append("-- Configuration values (environment-specific data)")
        for tname, rows in config_values.items():
            parts.append(f"-- Table: {tname}")
            for row in rows:
                row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                parts.append(f"--   {row_str}")
            parts.append("")

    return "\n".join(parts)


def _yaml_schema_to_ddl(data: dict[str, Any]) -> str:
    """Convert a YAML schema snapshot to DDL text.

    YAML format is identical to JSON format -- just parsed differently.
    """
    return _json_schema_to_ddl(data)


# ---------------------------------------------------------------------------
# SchemaIngester
# ---------------------------------------------------------------------------


class SchemaIngester:
    """Ingest database schema sources into the Mnemosyne index.

    Schema sources are stored as synthetic FileRecords with paths under
    ``__schema__/<env>/`` and ``source_type = 'schema'``.  Chunks flow
    through the standard chunking and embedding pipeline.

    Args:
        project_root: Absolute path to the project directory.
        config:       Mnemosyne Config instance.
        store:        Persistent Store instance.
        bloom:        BloomFilter for dedup.
        tfidf:        TF-IDF backend.
        audit:        AuditLog instance.
        dense:        Optional dense embedding backend.
    """

    def __init__(
        self,
        project_root: str,
        config: "Config",
        store: "Store",
        bloom: "BloomFilter",
        tfidf,
        audit: "AuditLog",
        dense=None,
    ) -> None:
        self.root = os.path.abspath(project_root)
        self.config = config
        self.store = store
        self.bloom = bloom
        self.tfidf = tfidf
        self.audit = audit
        self.dense = dense

    def ingest_from_file(
        self,
        source_path: str,
        env_tag: str = "",
        fmt: str = "auto",
    ) -> dict[str, int]:
        """Ingest a single schema source file.

        Args:
            source_path: Path to the DDL, JSON, or YAML file.
            env_tag:     Environment label (e.g. "prod", "dev").
            fmt:         Format hint: "ddl", "json", "yaml", or "auto" (detect
                         from extension).

        Returns:
            Stats dict with ``chunks_added``, ``chunks_deduped``,
            ``redactions``.
        """
        abs_path = os.path.abspath(source_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Schema source not found: {abs_path}")

        # Detect format
        if fmt == "auto":
            ext = os.path.splitext(abs_path)[1].lower()
            if ext in (".json",):
                fmt = "json"
            elif ext in (".yaml", ".yml"):
                fmt = "yaml"
            else:
                fmt = "ddl"

        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            raw_content = fh.read()

        return self._ingest_content(raw_content, source_path, env_tag, fmt)

    def introspect_sqlite(
        self,
        db_path: str,
        env_tag: str = "",
    ) -> dict[str, int]:
        """Introspect a local SQLite database and ingest its schema.

        Uses stdlib ``sqlite3`` PRAGMA queries to extract table definitions,
        indexes, and foreign keys, then converts them to DDL text and ingests
        through the standard pipeline.

        Args:
            db_path: Path to the SQLite database file.  Must be within the
                     project root (security constraint).
            env_tag: Environment label.

        Returns:
            Stats dict with ``chunks_added``, ``chunks_deduped``,
            ``redactions``, ``tables_found``.

        Raises:
            ValueError: If the path is outside the project root or points to
                        Mnemosyne's own database.
        """
        import sqlite3

        abs_db = os.path.abspath(db_path)

        # Security: path must be within project root
        try:
            common = os.path.commonpath([self.root, abs_db])
            if common != self.root:
                raise ValueError(
                    f"Database path {abs_db} is outside project root {self.root}"
                )
        except ValueError as exc:
            if "outside project root" in str(exc):
                raise
            raise ValueError(
                f"Database path {abs_db} is outside project root {self.root}"
            ) from exc

        # Security: reject Mnemosyne's own database
        mnemosyne_db = os.path.join(self.root, ".mnemosyne", "mnemosyne.db")
        if os.path.abspath(abs_db) == os.path.abspath(mnemosyne_db):
            raise ValueError("Cannot introspect Mnemosyne's own database")

        if not os.path.isfile(abs_db):
            raise FileNotFoundError(f"Database not found: {abs_db}")

        # Connect read-only
        conn = sqlite3.connect(f"file:{abs_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        try:
            ddl_parts: list[str] = []
            db_name = os.path.splitext(os.path.basename(abs_db))[0]
            ddl_parts.append(f"-- SQLite database: {db_name}")
            if env_tag:
                ddl_parts.append(f"-- Environment: {env_tag}")
            ddl_parts.append("")

            tables_found = 0

            # Get all user tables (skip sqlite_ internal tables)
            tables = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()

            for table in tables:
                tname = table["name"]
                create_sql = table["sql"]
                tables_found += 1

                if create_sql:
                    ddl_parts.append(f"{create_sql};")
                else:
                    # Generate DDL from PRAGMA for tables without stored SQL
                    cols = conn.execute(
                        f"PRAGMA table_info({tname})"
                    ).fetchall()
                    col_defs = []
                    for col in cols:
                        cdef = f"    {col['name']} {col['type'] or 'TEXT'}"
                        if col["notnull"]:
                            cdef += " NOT NULL"
                        if col["dflt_value"] is not None:
                            cdef += f" DEFAULT {col['dflt_value']}"
                        if col["pk"]:
                            cdef += " PRIMARY KEY"
                        col_defs.append(cdef)
                    ddl_parts.append(f"CREATE TABLE {tname} (")
                    ddl_parts.append(",\n".join(col_defs))
                    ddl_parts.append(");")

                # Foreign keys
                fks = conn.execute(
                    f"PRAGMA foreign_key_list({tname})"
                ).fetchall()
                for fk in fks:
                    ddl_parts.append(
                        f"-- FK: {tname}.{fk['from']} -> "
                        f"{fk['table']}.{fk['to']}"
                    )

                # Indexes for this table
                indexes = conn.execute(
                    f"PRAGMA index_list({tname})"
                ).fetchall()
                for idx in indexes:
                    idx_name = idx["name"]
                    unique = "UNIQUE " if idx["unique"] else ""
                    idx_cols = conn.execute(
                        f"PRAGMA index_info({idx_name})"
                    ).fetchall()
                    col_names = ", ".join(c["name"] for c in idx_cols if c["name"])
                    if col_names:
                        ddl_parts.append(
                            f"CREATE {unique}INDEX {idx_name} "
                            f"ON {tname} ({col_names});"
                        )

                # Row count
                try:
                    row_count = conn.execute(
                        f"SELECT COUNT(*) FROM [{tname}]"
                    ).fetchone()[0]
                    ddl_parts.append(f"-- Row count: {row_count}")
                except Exception:
                    pass

                ddl_parts.append("")

            # Views
            views = conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'view' ORDER BY name"
            ).fetchall()
            for view in views:
                if view["sql"]:
                    ddl_parts.append(f"{view['sql']};")
                    ddl_parts.append("")

        finally:
            conn.close()

        ddl_text = "\n".join(ddl_parts)
        if tables_found == 0:
            return {
                "chunks_added": 0,
                "chunks_deduped": 0,
                "redactions": 0,
                "tables_found": 0,
            }

        # Use the database filename as the source name
        source_name = os.path.basename(db_path)
        stats = self._ingest_content(ddl_text, source_name, env_tag, "ddl")
        stats["tables_found"] = tables_found
        return stats

    def ingest_from_config(self) -> dict[str, int]:
        """Ingest all schema sources defined in config.database section.

        Returns aggregated stats.
        """
        db_cfg = self.config.database
        sources = getattr(db_cfg, "schema_sources", []) or []
        env_tag = getattr(db_cfg, "environment_tag", "") or ""

        total_stats: dict[str, int] = {
            "sources_processed": 0,
            "sources_failed": 0,
            "chunks_added": 0,
            "chunks_deduped": 0,
            "redactions": 0,
        }

        for src in sources:
            abs_src = os.path.abspath(os.path.join(self.root, src))
            if os.path.isdir(abs_src):
                # Walk directory for schema files
                for dirpath, _, filenames in os.walk(abs_src):
                    for fname in filenames:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in (".sql", ".ddl", ".json", ".yaml", ".yml"):
                            fpath = os.path.join(dirpath, fname)
                            try:
                                stats = self.ingest_from_file(fpath, env_tag)
                                total_stats["chunks_added"] += stats["chunks_added"]
                                total_stats["chunks_deduped"] += stats["chunks_deduped"]
                                total_stats["redactions"] += stats["redactions"]
                                total_stats["sources_processed"] += 1
                            except Exception:
                                total_stats["sources_failed"] += 1
            elif os.path.isfile(abs_src):
                try:
                    stats = self.ingest_from_file(abs_src, env_tag)
                    total_stats["chunks_added"] += stats["chunks_added"]
                    total_stats["chunks_deduped"] += stats["chunks_deduped"]
                    total_stats["redactions"] += stats["redactions"]
                    total_stats["sources_processed"] += 1
                except Exception:
                    total_stats["sources_failed"] += 1

        return total_stats

    def _ingest_content(
        self,
        raw_content: str,
        source_path: str,
        env_tag: str,
        fmt: str,
    ) -> dict[str, int]:
        """Parse, optionally redact, chunk, and store schema content."""
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.hasher import content_hash as compute_content_hash

        security_tier = getattr(self.config.database, "security_tier", 1)

        # Parse structured formats to DDL text
        if fmt == "json":
            data = json.loads(raw_content)
            ddl_text = _json_schema_to_ddl(data)
            # Override env_tag from JSON if present and not explicitly set
            if not env_tag:
                env_tag = data.get("environment", "")
        elif fmt == "yaml":
            # Use a basic YAML parser (stdlib doesn't include yaml, but we
            # can handle simple cases or fall back to treating as DDL)
            try:
                import yaml  # type: ignore[import-untyped]
                data = yaml.safe_load(raw_content)
                ddl_text = _yaml_schema_to_ddl(data)
                if not env_tag:
                    env_tag = data.get("environment", "")
            except ImportError:
                # No yaml module available -- try JSON parsing as fallback
                try:
                    data = json.loads(raw_content)
                    ddl_text = _json_schema_to_ddl(data)
                except (json.JSONDecodeError, ValueError):
                    ddl_text = raw_content
        else:
            ddl_text = raw_content

        # Apply redaction for Tier 2+
        redactions = 0
        if security_tier >= 2:
            ddl_text, redactions = _redact(ddl_text)

        # Determine synthetic rel_path
        base_name = os.path.basename(source_path)
        name_stem = os.path.splitext(base_name)[0]
        env_part = env_tag if env_tag else "default"
        rel_path = f"__schema__/{env_part}/{name_stem}.ddl"

        # Create synthetic FileRecord
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content_hash_val = _content_hash(ddl_text)

        file_record = FileRecord(
            file_id=None,
            rel_path=rel_path,
            content_hash=content_hash_val,
            size_bytes=len(ddl_text.encode("utf-8")),
            language="sql_schema",
            last_modified=time.time(),
            last_indexed=now_iso,
            is_deleted=False,
        )
        file_id = self.store.upsert_file(file_record)

        # Set source_type to 'schema' (added in migration v2)
        try:
            self.store.conn.execute(
                "UPDATE files SET source_type = ? WHERE file_id = ?",
                ("schema", file_id),
            )
            self.store.conn.commit()
        except Exception:
            pass  # Column may not exist in older DBs

        # Delete old chunks for this synthetic file
        self.store.delete_chunks_for_file(file_id)

        # Chunk the DDL text
        chunker = get_chunker("sql_schema", self.config)
        candidates = chunker.chunk(ddl_text, "sql_schema")

        chunks_added = 0
        chunks_deduped = 0

        for cand in candidates:
            chunk_hash = compute_content_hash(cand.content)

            # Deduplicate
            existing = self.store.get_chunk_by_hash(chunk_hash)
            if existing is not None:
                chunks_deduped += 1
                continue

            token_count = estimate_tokens(cand.content)
            chunk = Chunk(
                chunk_id=None,
                file_id=file_id,
                content_hash=chunk_hash,
                chunk_type=cand.chunk_type,
                line_start=cand.line_start,
                line_end=cand.line_end,
                token_count=token_count,
                content=cand.content,
                compressed=None,
                compression_ratio=None,
                symbol_name=cand.symbol_name,
                parent_chunk_id=None,
            )
            chunk_id = self.store.save_chunk(chunk)

            # Build enriched text for embedding
            enriched = self._build_enriched_text(cand, rel_path, env_tag)

            # TF-IDF embedding
            try:
                terms = self.tfidf.embed(enriched)
                self.store.insert_sparse_embedding(chunk_id, terms)
            except Exception:
                pass

            # Dense embedding (optional)
            if self.dense is not None:
                try:
                    vec_bytes = self.dense.embed_to_bytes(enriched)
                    if vec_bytes:
                        self.store.insert_dense_embedding(
                            chunk_id, vec_bytes, dim=self.dense.dim
                        )
                except Exception:
                    pass

            self.bloom.add(chunk_hash)
            chunks_added += 1

        self.bloom.add(rel_path)

        # Audit log
        self.audit.log(
            "schema_ingest",
            rel_path=rel_path,
            chunks_added=chunks_added,
            chunks_deduped=chunks_deduped,
            redactions=redactions,
        )

        return {
            "chunks_added": chunks_added,
            "chunks_deduped": chunks_deduped,
            "redactions": redactions,
        }

    @staticmethod
    def _build_enriched_text(cand, rel_path: str, env_tag: str) -> str:
        """Build enriched text for embedding schema chunks."""
        parts: list[str] = []
        parts.append(f"# Schema: {rel_path}")
        if env_tag:
            parts.append(f"# Environment: {env_tag}")
        if cand.symbol_name:
            parts.append(f"# Object: {cand.symbol_name} ({cand.chunk_type})")
        parts.append(cand.content)
        return "\n".join(parts)

    def get_schema_stats(self) -> dict[str, Any]:
        """Return statistics about ingested schema sources."""
        rows = self.store.conn.execute(
            "SELECT rel_path, language FROM files "
            "WHERE rel_path LIKE '__schema__/%'"
        ).fetchall()

        envs: set[str] = set()
        source_count = 0
        for row in rows:
            source_count += 1
            parts = row["rel_path"].split("/")
            if len(parts) >= 2:
                envs.add(parts[1])

        chunk_count = self.store.conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.file_id "
            "WHERE f.rel_path LIKE '__schema__/%'"
        ).fetchone()[0]

        type_counts = {}
        type_rows = self.store.conn.execute(
            "SELECT c.chunk_type, COUNT(*) as cnt "
            "FROM chunks c JOIN files f ON c.file_id = f.file_id "
            "WHERE f.rel_path LIKE '__schema__/%' "
            "GROUP BY c.chunk_type"
        ).fetchall()
        for row in type_rows:
            type_counts[row["chunk_type"]] = row["cnt"]

        return {
            "schema_sources": source_count,
            "environments": sorted(envs),
            "total_chunks": chunk_count,
            "chunk_types": type_counts,
        }
