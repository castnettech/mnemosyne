# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Integration tests for schema ingestion (Phase 2).

Covers: JSON snapshot import, DDL file import, environment tagging,
FTS5 searchability of schema terms, symbol search for table names,
synthetic FileRecord creation, source_type migration, redaction,
and SchemaIngester stats.
"""

import json
import os
import tempfile
import unittest


def _setup_project():
    """Create a temp project with .mnemosyne/ initialized."""
    from mnemosyne.config import Config
    from mnemosyne.schema import open_store
    from mnemosyne.store import Store
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog
    from mnemosyne.embeddings import get_backend

    tmpdir = tempfile.mkdtemp()
    config = Config(root=tmpdir)

    db_dir = os.path.join(tmpdir, ".mnemosyne")
    os.makedirs(db_dir, exist_ok=True)
    conn = open_store(db_dir)
    store = Store(conn)
    bloom = BloomFilter()
    tfidf = get_backend(config, store)
    audit = AuditLog(os.path.join(db_dir, "audit.log"))

    return tmpdir, config, conn, store, bloom, tfidf, audit


# =========================================================================
# JSON snapshot import
# =========================================================================


class TestJsonSchemaImport(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_json_snapshot_creates_chunks(self):
        schema = {
            "database": "myapp",
            "environment": "production",
            "tables": [
                {
                    "name": "shipping_config",
                    "columns": [
                        {"name": "id", "type": "INTEGER", "primary_key": True},
                        {"name": "region", "type": "TEXT", "nullable": False},
                        {"name": "estimated_days", "type": "INTEGER", "default": 14},
                    ],
                    "indexes": ["idx_shipping_region"],
                    "row_count": 42,
                }
            ],
        }
        path = os.path.join(self.tmpdir, "schema.json")
        with open(path, "w") as f:
            json.dump(schema, f)

        stats = self.ingester.ingest_from_file(path, env_tag="prod")
        self.assertGreater(stats["chunks_added"], 0)

    def test_json_snapshot_with_config_values(self):
        schema = {
            "database": "myapp",
            "environment": "production",
            "tables": [
                {
                    "name": "settings",
                    "columns": [
                        {"name": "key", "type": "TEXT", "primary_key": True},
                        {"name": "value", "type": "TEXT"},
                    ],
                }
            ],
            "config_values": {
                "settings": [
                    {"key": "shipping_time", "value": "21"},
                    {"key": "max_retries", "value": "3"},
                ]
            },
        }
        path = os.path.join(self.tmpdir, "prod_config.json")
        with open(path, "w") as f:
            json.dump(schema, f)

        stats = self.ingester.ingest_from_file(path, env_tag="prod")
        self.assertGreater(stats["chunks_added"], 0)

    def test_env_tag_from_json(self):
        """Environment tag should be extracted from JSON if not explicitly set."""
        schema = {
            "database": "myapp",
            "environment": "staging",
            "tables": [
                {
                    "name": "t1",
                    "columns": [{"name": "id", "type": "INT"}],
                }
            ],
        }
        path = os.path.join(self.tmpdir, "staging.json")
        with open(path, "w") as f:
            json.dump(schema, f)

        self.ingester.ingest_from_file(path)

        # Check that the synthetic file path includes the env tag
        row = self.conn.execute(
            "SELECT rel_path FROM files WHERE rel_path LIKE '__schema__%'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn("staging", row["rel_path"])


# =========================================================================
# DDL file import
# =========================================================================


class TestDdlFileImport(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_ddl_file_import(self):
        ddl = (
            "CREATE TABLE users (\n"
            "    id SERIAL PRIMARY KEY,\n"
            "    name TEXT NOT NULL,\n"
            "    email TEXT UNIQUE\n"
            ");\n"
            "\n"
            "CREATE INDEX idx_users_email ON users (email);\n"
        )
        path = os.path.join(self.tmpdir, "schema.sql")
        with open(path, "w") as f:
            f.write(ddl)

        stats = self.ingester.ingest_from_file(path, env_tag="dev")
        self.assertGreater(stats["chunks_added"], 0)

    def test_synthetic_rel_path_format(self):
        ddl = "CREATE TABLE orders (id INT PRIMARY KEY);\n"
        path = os.path.join(self.tmpdir, "orders.sql")
        with open(path, "w") as f:
            f.write(ddl)

        self.ingester.ingest_from_file(path, env_tag="prod")

        row = self.conn.execute(
            "SELECT rel_path, language FROM files WHERE rel_path LIKE '__schema__%'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["rel_path"], "__schema__/prod/orders.ddl")
        self.assertEqual(row["language"], "sql_schema")


# =========================================================================
# FTS5 searchability
# =========================================================================


class TestSchemaFtsSearch(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_table_name_searchable_via_fts5(self):
        ddl = "CREATE TABLE shipping_config (id INT, region TEXT, estimated_days INT);\n"
        path = os.path.join(self.tmpdir, "shipping.sql")
        with open(path, "w") as f:
            f.write(ddl)

        self.ingester.ingest_from_file(path, env_tag="prod")

        # FTS5 search for table name
        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("shipping_config",),
        ).fetchall()
        self.assertGreater(len(rows), 0)

    def test_column_name_searchable_via_fts5(self):
        ddl = "CREATE TABLE orders (id INT, estimated_days INT, total NUMERIC);\n"
        path = os.path.join(self.tmpdir, "orders.sql")
        with open(path, "w") as f:
            f.write(ddl)

        self.ingester.ingest_from_file(path, env_tag="dev")

        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("estimated_days",),
        ).fetchall()
        self.assertGreater(len(rows), 0)


# =========================================================================
# Symbol search for table names
# =========================================================================


class TestSchemaSymbolSearch(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_table_name_stored_as_symbol(self):
        ddl = "CREATE TABLE user_roles (id INT, role TEXT);\n"
        path = os.path.join(self.tmpdir, "roles.sql")
        with open(path, "w") as f:
            f.write(ddl)

        self.ingester.ingest_from_file(path, env_tag="prod")

        rows = self.conn.execute(
            "SELECT symbol_name, chunk_type FROM chunks WHERE symbol_name = ?",
            ("user_roles",),
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_type"], "table_ddl")


# =========================================================================
# Environment tagging and multi-env
# =========================================================================


class TestEnvironmentTagging(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_same_schema_different_envs(self):
        ddl = "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);\n"

        for env in ("prod", "dev"):
            path = os.path.join(self.tmpdir, f"config_{env}.sql")
            with open(path, "w") as f:
                f.write(ddl)
            self.ingester.ingest_from_file(path, env_tag=env)

        # Should have two separate file entries
        rows = self.conn.execute(
            "SELECT rel_path FROM files WHERE rel_path LIKE '__schema__%'"
        ).fetchall()
        paths = [r["rel_path"] for r in rows]
        self.assertEqual(len(paths), 2)
        self.assertTrue(any("prod" in p for p in paths))
        self.assertTrue(any("dev" in p for p in paths))


# =========================================================================
# Schema migration v2
# =========================================================================


class TestSchemaMigration(unittest.TestCase):

    def test_source_type_column_exists(self):
        tmpdir, config, conn, store, bloom, tfidf, audit = _setup_project()
        # The migration should have added source_type
        row = conn.execute(
            "PRAGMA table_info(files)"
        ).fetchall()
        col_names = [r["name"] for r in row]
        self.assertIn("source_type", col_names)
        conn.close()

    def test_source_type_default_is_file(self):
        tmpdir, config, conn, store, bloom, tfidf, audit = _setup_project()
        from mnemosyne.models import FileRecord
        record = FileRecord(
            file_id=None,
            rel_path="test.py",
            content_hash="abc123",
            size_bytes=100,
            language="python",
            last_modified=0.0,
        )
        file_id = store.upsert_file(record)
        row = conn.execute(
            "SELECT source_type FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        self.assertEqual(row["source_type"], "file")
        conn.close()


# =========================================================================
# Redaction (Tier 2)
# =========================================================================


class TestRedaction(unittest.TestCase):

    def test_email_redacted(self):
        from mnemosyne.schema_ingest import _redact
        text = "DEFAULT 'admin@example.com'"
        redacted, count = _redact(text)
        self.assertNotIn("admin@example.com", redacted)
        self.assertIn("[REDACTED]", redacted)
        self.assertGreater(count, 0)

    def test_url_with_credentials_redacted(self):
        from mnemosyne.schema_ingest import _redact
        text = "postgresql://user:pass@localhost:5432/db"
        redacted, count = _redact(text)
        self.assertNotIn("user:pass@", redacted)
        self.assertGreater(count, 0)

    def test_plain_text_not_redacted(self):
        from mnemosyne.schema_ingest import _redact
        text = "CREATE TABLE users (id INT PRIMARY KEY)"
        redacted, count = _redact(text)
        self.assertEqual(text, redacted)
        self.assertEqual(count, 0)


# =========================================================================
# Schema stats
# =========================================================================


class TestSchemaStats(unittest.TestCase):

    def setUp(self):
        self.tmpdir, self.config, self.conn, self.store, self.bloom, self.tfidf, self.audit = _setup_project()
        from mnemosyne.schema_ingest import SchemaIngester
        self.ingester = SchemaIngester(
            project_root=self.tmpdir,
            config=self.config,
            store=self.store,
            bloom=self.bloom,
            tfidf=self.tfidf,
            audit=self.audit,
        )

    def tearDown(self):
        self.conn.close()

    def test_stats_empty(self):
        stats = self.ingester.get_schema_stats()
        self.assertEqual(stats["schema_sources"], 0)
        self.assertEqual(stats["total_chunks"], 0)
        self.assertEqual(stats["environments"], [])

    def test_stats_after_ingest(self):
        ddl = "CREATE TABLE t1 (id INT);\nCREATE TABLE t2 (id INT);\n"
        path = os.path.join(self.tmpdir, "tables.sql")
        with open(path, "w") as f:
            f.write(ddl)

        self.ingester.ingest_from_file(path, env_tag="prod")
        stats = self.ingester.get_schema_stats()
        self.assertEqual(stats["schema_sources"], 1)
        self.assertIn("prod", stats["environments"])
        self.assertGreater(stats["total_chunks"], 0)
        self.assertIn("table_ddl", stats["chunk_types"])


if __name__ == "__main__":
    unittest.main()
