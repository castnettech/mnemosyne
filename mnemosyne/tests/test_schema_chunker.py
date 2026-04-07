# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the SQL DDL-aware SchemaChunker.

Covers: CREATE TABLE/INDEX/VIEW boundary detection, symbol_name extraction,
multi-statement files, ALTER TABLE, PostgreSQL/MySQL/SQLite dialect handling,
comment preservation, oversized statement splitting, and interstitial block
emission.
"""

import unittest


def _default_config():
    """Return a Config with default settings from a temp directory."""
    import tempfile
    from mnemosyne.config import Config
    return Config(root=tempfile.mkdtemp())


# =========================================================================
# SchemaChunker -- basic DDL detection
# =========================================================================


class TestSchemaChunkerBasic(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_single_create_table(self):
        sql = (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "table_ddl")
        self.assertEqual(chunks[0].symbol_name, "users")
        self.assertIn("CREATE TABLE", chunks[0].content)

    def test_create_index(self):
        sql = "CREATE INDEX idx_users_name ON users (name);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "index_ddl")
        self.assertEqual(chunks[0].symbol_name, "idx_users_name")

    def test_create_unique_index(self):
        sql = "CREATE UNIQUE INDEX idx_users_email ON users (email);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "index_ddl")
        self.assertEqual(chunks[0].symbol_name, "idx_users_email")

    def test_create_view(self):
        sql = (
            "CREATE VIEW active_users AS\n"
            "SELECT * FROM users WHERE active = true;\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "view_ddl")
        self.assertEqual(chunks[0].symbol_name, "active_users")

    def test_create_materialized_view(self):
        sql = (
            "CREATE MATERIALIZED VIEW mv_stats AS\n"
            "SELECT count(*) FROM orders;\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "view_ddl")
        self.assertEqual(chunks[0].symbol_name, "mv_stats")

    def test_alter_table(self):
        sql = "ALTER TABLE users ADD COLUMN email TEXT;\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "table_ddl")
        self.assertEqual(chunks[0].symbol_name, "users")

    def test_empty_source(self):
        self.assertEqual(self.chunker.chunk(""), [])
        self.assertEqual(self.chunker.chunk("   \n  "), [])


# =========================================================================
# SchemaChunker -- multi-statement files
# =========================================================================


class TestSchemaChunkerMultiStatement(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_two_tables(self):
        sql = (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY\n"
            ");\n"
            "\n"
            "CREATE TABLE orders (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    user_id INTEGER REFERENCES users(id)\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        ddl_chunks = [c for c in chunks if c.chunk_type == "table_ddl"]
        self.assertEqual(len(ddl_chunks), 2)
        self.assertEqual(ddl_chunks[0].symbol_name, "users")
        self.assertEqual(ddl_chunks[1].symbol_name, "orders")

    def test_table_and_index(self):
        sql = (
            "CREATE TABLE products (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    name TEXT NOT NULL\n"
            ");\n"
            "\n"
            "CREATE INDEX idx_products_name ON products (name);\n"
        )
        chunks = self.chunker.chunk(sql)
        types = [c.chunk_type for c in chunks if c.chunk_type != "block"]
        self.assertIn("table_ddl", types)
        self.assertIn("index_ddl", types)

    def test_interstitial_comments_become_blocks(self):
        sql = (
            "-- Schema for the user module\n"
            "-- Version 2.0\n"
            "\n"
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        block_chunks = [c for c in chunks if c.chunk_type == "block"]
        ddl_chunks = [c for c in chunks if c.chunk_type == "table_ddl"]
        self.assertEqual(len(ddl_chunks), 1)
        self.assertEqual(len(block_chunks), 1)
        self.assertIn("Schema for the user module", block_chunks[0].content)


# =========================================================================
# SchemaChunker -- dialect handling
# =========================================================================


class TestSchemaChunkerDialects(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_if_not_exists(self):
        sql = "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "settings")

    def test_schema_qualified_name(self):
        sql = "CREATE TABLE public.users (id SERIAL PRIMARY KEY);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "users")

    def test_temporary_table(self):
        sql = "CREATE TEMPORARY TABLE tmp_calc (val INTEGER);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "tmp_calc")

    def test_or_replace_view(self):
        sql = "CREATE OR REPLACE VIEW v_orders AS SELECT * FROM orders;\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "view_ddl")
        self.assertEqual(chunks[0].symbol_name, "v_orders")

    def test_quoted_identifier(self):
        sql = 'CREATE TABLE "user-data" (id INTEGER);\n'
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "user-data")

    def test_case_insensitive(self):
        sql = "create table Shipping_Config (id integer primary key);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "Shipping_Config")
        self.assertEqual(chunks[0].chunk_type, "table_ddl")

    def test_postgresql_serial_and_constraints(self):
        sql = (
            "CREATE TABLE orders (\n"
            "    id SERIAL PRIMARY KEY,\n"
            "    user_id INTEGER NOT NULL REFERENCES users(id),\n"
            "    total NUMERIC(10, 2) DEFAULT 0.00,\n"
            "    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "orders")
        self.assertIn("SERIAL", chunks[0].content)

    def test_create_index_concurrently(self):
        sql = "CREATE INDEX CONCURRENTLY idx_active ON users (active);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "idx_active")

    def test_no_trailing_semicolon(self):
        """Files without a trailing semicolon should still parse."""
        sql = "CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "config")

    def test_unlogged_table(self):
        sql = "CREATE UNLOGGED TABLE audit_log (id SERIAL, msg TEXT);\n"
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].symbol_name, "audit_log")


# =========================================================================
# SchemaChunker -- comment handling
# =========================================================================


class TestSchemaChunkerComments(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_line_comments_preserved(self):
        sql = (
            "CREATE TABLE users (\n"
            "    -- primary identifier\n"
            "    id INTEGER PRIMARY KEY\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertIn("-- primary identifier", chunks[0].content)

    def test_block_comments_preserved(self):
        sql = (
            "CREATE TABLE users (\n"
            "    /* auto-incrementing ID */\n"
            "    id INTEGER PRIMARY KEY\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertIn("/* auto-incrementing ID */", chunks[0].content)

    def test_semicolon_inside_string_not_statement_end(self):
        sql = (
            "CREATE TABLE logs (\n"
            "    id INTEGER PRIMARY KEY,\n"
            "    msg TEXT DEFAULT 'no; data'\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        ddl_chunks = [c for c in chunks if c.chunk_type == "table_ddl"]
        self.assertEqual(len(ddl_chunks), 1)
        self.assertIn("'no; data'", ddl_chunks[0].content)


# =========================================================================
# SchemaChunker -- line numbers
# =========================================================================


class TestSchemaChunkerLineNumbers(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_line_numbers_single_statement(self):
        sql = (
            "CREATE TABLE users (\n"
            "    id INTEGER PRIMARY KEY\n"
            ");\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(chunks[0].line_start, 1)
        self.assertGreaterEqual(chunks[0].line_end, 3)

    def test_line_numbers_with_leading_comments(self):
        sql = (
            "-- header comment\n"
            "-- another line\n"
            "\n"
            "CREATE TABLE t1 (id INT);\n"
        )
        chunks = self.chunker.chunk(sql)
        ddl = [c for c in chunks if c.chunk_type == "table_ddl"]
        self.assertEqual(len(ddl), 1)
        self.assertEqual(ddl[0].line_start, 4)


# =========================================================================
# SchemaChunker -- no DDL fallback
# =========================================================================


class TestSchemaChunkerFallback(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_no_ddl_returns_single_block(self):
        sql = (
            "-- Just some comments\n"
            "SELECT * FROM users;\n"
            "INSERT INTO logs VALUES (1, 'test');\n"
        )
        chunks = self.chunker.chunk(sql)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "block")


# =========================================================================
# SchemaChunker -- PostgreSQL dollar-quoted strings
# =========================================================================


class TestSchemaChunkerDollarQuoting(unittest.TestCase):

    def setUp(self):
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        self.chunker = SchemaChunker(_default_config())

    def test_function_with_dollar_quoted_body(self):
        sql = (
            "CREATE FUNCTION increment(i INTEGER) RETURNS INTEGER AS $$\n"
            "BEGIN\n"
            "    RETURN i + 1;\n"
            "END;\n"
            "$$ LANGUAGE plpgsql;\n"
        )
        chunks = self.chunker.chunk(sql)
        func_chunks = [c for c in chunks if c.symbol_name == "increment"]
        self.assertEqual(len(func_chunks), 1)
        # The whole function body should be in one chunk
        self.assertIn("RETURN i + 1", func_chunks[0].content)


# =========================================================================
# SchemaChunker -- registry dispatch
# =========================================================================


class TestSchemaChunkerRegistry(unittest.TestCase):

    def test_sql_dispatches_to_schema_chunker(self):
        from mnemosyne.chunkers import get_chunker, detect_language
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        config = _default_config()
        lang = detect_language("schema.sql")
        self.assertEqual(lang, "sql")
        chunker = get_chunker(lang, config)
        self.assertIsInstance(chunker, SchemaChunker)

    def test_sql_schema_dispatches_to_schema_chunker(self):
        from mnemosyne.chunkers import get_chunker
        from mnemosyne.chunkers.schema_chunker import SchemaChunker
        config = _default_config()
        chunker = get_chunker("sql_schema", config)
        self.assertIsInstance(chunker, SchemaChunker)


if __name__ == "__main__":
    unittest.main()
