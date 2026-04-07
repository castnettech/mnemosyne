# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for SQLite introspection and schema ingestion security (Phase 3).

Covers: SQLite PRAGMA introspection, DDL generation from live databases,
path containment security, Mnemosyne DB rejection, FTS5 searchability
of introspected schemas, foreign key detection, index extraction, and
view handling.
"""

import os
import sqlite3
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


def _create_test_db(tmpdir: str, name: str = "test.db") -> str:
    """Create a test SQLite database with sample tables."""
    db_path = os.path.join(tmpdir, name)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    """)
    conn.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            total REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX idx_orders_user ON orders (user_id)")
    conn.execute("""
        CREATE VIEW active_orders AS
        SELECT o.*, u.name as user_name
        FROM orders o JOIN users u ON o.user_id = u.id
    """)
    # Insert some test data
    conn.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@test.com')")
    conn.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@test.com')")
    conn.execute("INSERT INTO orders (user_id, total) VALUES (1, 99.99)")
    conn.execute("INSERT INTO orders (user_id, total) VALUES (1, 49.99)")
    conn.execute("INSERT INTO orders (user_id, total) VALUES (2, 25.00)")
    conn.commit()
    conn.close()
    return db_path


# =========================================================================
# SQLite introspection -- basic
# =========================================================================


class TestSqliteIntrospection(unittest.TestCase):

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
        self.test_db = _create_test_db(self.tmpdir)

    def tearDown(self):
        self.conn.close()

    def test_introspect_creates_chunks(self):
        stats = self.ingester.introspect_sqlite(self.test_db, env_tag="dev")
        self.assertGreater(stats["chunks_added"], 0)
        self.assertEqual(stats["tables_found"], 2)  # users + orders

    def test_introspect_finds_tables(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="local")

        # Check that table names are stored as symbols
        rows = self.conn.execute(
            "SELECT symbol_name FROM chunks WHERE chunk_type = 'table_ddl'"
        ).fetchall()
        names = {r["symbol_name"] for r in rows}
        self.assertIn("users", names)
        self.assertIn("orders", names)

    def test_introspect_captures_indexes(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="dev")

        rows = self.conn.execute(
            "SELECT content FROM chunks WHERE chunk_type = 'index_ddl'"
        ).fetchall()
        index_content = " ".join(r["content"] for r in rows)
        self.assertIn("idx_orders_user", index_content)

    def test_introspect_captures_views(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="dev")

        rows = self.conn.execute(
            "SELECT symbol_name FROM chunks WHERE chunk_type = 'view_ddl'"
        ).fetchall()
        names = {r["symbol_name"] for r in rows}
        self.assertIn("active_orders", names)

    def test_introspect_includes_row_counts(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="dev")

        # Row counts should appear as comments in the DDL content
        rows = self.conn.execute(
            "SELECT content FROM chunks"
        ).fetchall()
        all_content = " ".join(r["content"] for r in rows)
        self.assertIn("Row count:", all_content)

    def test_introspect_fts5_searchable(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="dev")

        # Table name should be FTS-searchable
        rows = self.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
            ("users",),
        ).fetchall()
        self.assertGreater(len(rows), 0)

    def test_introspect_with_env_tag(self):
        self.ingester.introspect_sqlite(self.test_db, env_tag="production")

        row = self.conn.execute(
            "SELECT rel_path FROM files WHERE rel_path LIKE '__schema__%'"
        ).fetchone()
        self.assertIn("production", row["rel_path"])


# =========================================================================
# Security: path containment
# =========================================================================


class TestSqliteIntrospectionSecurity(unittest.TestCase):

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

    def test_rejects_path_outside_project_root(self):
        outside_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        outside_db.close()
        try:
            with self.assertRaises(ValueError) as ctx:
                self.ingester.introspect_sqlite(outside_db.name, env_tag="dev")
            self.assertIn("outside project root", str(ctx.exception))
        finally:
            os.unlink(outside_db.name)

    def test_rejects_mnemosyne_own_database(self):
        mnemosyne_db = os.path.join(self.tmpdir, ".mnemosyne", "mnemosyne.db")
        with self.assertRaises(ValueError) as ctx:
            self.ingester.introspect_sqlite(mnemosyne_db, env_tag="dev")
        self.assertIn("own database", str(ctx.exception))

    def test_rejects_nonexistent_database(self):
        fake_path = os.path.join(self.tmpdir, "nonexistent.db")
        with self.assertRaises(FileNotFoundError):
            self.ingester.introspect_sqlite(fake_path, env_tag="dev")


# =========================================================================
# Empty and edge-case databases
# =========================================================================


class TestSqliteEdgeCases(unittest.TestCase):

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

    def test_empty_database(self):
        db_path = os.path.join(self.tmpdir, "empty.db")
        conn = sqlite3.connect(db_path)
        conn.close()

        stats = self.ingester.introspect_sqlite(db_path, env_tag="dev")
        self.assertEqual(stats["tables_found"], 0)
        self.assertEqual(stats["chunks_added"], 0)

    def test_database_with_foreign_keys(self):
        db_path = os.path.join(self.tmpdir, "fk_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE child (id INTEGER PRIMARY KEY, "
            "parent_id INTEGER REFERENCES parent(id))"
        )
        conn.commit()
        conn.close()

        self.ingester.introspect_sqlite(db_path, env_tag="dev")
        rows = self.conn.execute(
            "SELECT content FROM chunks"
        ).fetchall()
        all_content = " ".join(r["content"] for r in rows)
        self.assertIn("FK:", all_content)


if __name__ == "__main__":
    unittest.main()
