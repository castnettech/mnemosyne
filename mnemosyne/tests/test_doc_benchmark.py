# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document partition benchmark -- regression gate for Tier 0 ingestion.

Builds a synthetic mixed-content project (code + docs + schema), indexes
it through both partitions, runs 12 ground-truth questions, and asserts
retrieval quality baselines.  Catches regressions in extractors, document
chunking, DocStore, DocRetrievalEngine, and partition isolation.
"""

from __future__ import annotations

import json
import os
import time
import zipfile

import pytest

from mnemosyne.config import Config
from mnemosyne.schema import open_store
from mnemosyne.store import Store
from mnemosyne.doc_store import DocStore
from mnemosyne.bloom import BloomFilter
from mnemosyne.audit import AuditLog
from mnemosyne.embeddings import get_backend
from mnemosyne.ingest import Ingester
from mnemosyne.retrieval import RetrievalEngine
from mnemosyne.doc_retrieval import DocRetrievalEngine
from mnemosyne.analytics import Analytics
from mnemosyne.prefetch import Prefetcher


# ---------------------------------------------------------------------------
# Synthetic project builder
# ---------------------------------------------------------------------------

def _make_docx(path: str, paragraphs: list[str]) -> None:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    parts = ['<?xml version="1.0" encoding="UTF-8"?>']
    parts.append(f'<w:document xmlns:w="{ns}"><w:body>')
    for para in paragraphs:
        parts.append(f"<w:p><w:r><w:t>{para}</w:t></w:r></w:p>")
    parts.append("</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", "\n".join(parts))
        ct = '<?xml version="1.0" encoding="UTF-8"?>'
        ct += '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        ct += '<Default Extension="xml" ContentType="application/xml"/>'
        ct += "</Types>"
        zf.writestr("[Content_Types].xml", ct)


def build_acme_project(root) -> None:
    """Build the acme-analytics synthetic project at *root*."""
    src = root / "src"
    schema = root / "schema"
    docs = root / "docs"
    for d in (src, schema, docs):
        d.mkdir(parents=True, exist_ok=True)

    # --- Python source (code partition) ---
    (src / "analytics.py").write_text('''\
"""Analytics engine for the Acme platform."""

from dataclasses import dataclass


class AnalyticsEngine:
    """Core analytics query and aggregation engine."""

    def __init__(self, pool):
        self.pool = pool

    def query(self, metric_name: str, start_date: str, end_date: str) -> list[dict]:
        """Filter metrics by name and date range, return matching rows.

        Executes a parameterized SQL query against the metrics table,
        applies date range filtering, and returns results sorted by
        timestamp descending.
        """
        sql = (
            "SELECT * FROM metrics "
            "WHERE name = ? AND recorded_at BETWEEN ? AND ? "
            "ORDER BY recorded_at DESC"
        )
        return self.pool.execute(sql, (metric_name, start_date, end_date))

    def aggregate(self, metric_name: str, period: str = "daily") -> dict:
        """Aggregate metric values by time period.

        Groups by day, week, or month and computes sum, avg, min, max
        for the specified metric.
        """
        group_expr = {
            "daily": "DATE(recorded_at)",
            "weekly": "strftime('%Y-%W', recorded_at)",
            "monthly": "strftime('%Y-%m', recorded_at)",
        }.get(period, "DATE(recorded_at)")
        sql = (
            f"SELECT {group_expr} AS period, "
            "SUM(value) AS total, AVG(value) AS average, "
            "MIN(value) AS minimum, MAX(value) AS maximum "
            f"FROM metrics WHERE name = ? GROUP BY {group_expr}"
        )
        return self.pool.execute(sql, (metric_name,))

    def export_csv(self, metric_name: str, output_path: str) -> int:
        """Export metric data to a CSV file. Returns row count."""
        rows = self.query(metric_name, "1970-01-01", "2099-12-31")
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "value", "recorded_at"])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)
''')

    (src / "models.py").write_text('''\
"""Data model classes for the Acme analytics platform."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metric:
    """A single metric measurement.

    Attributes:
        metric_id: Unique identifier.
        name: Metric name (e.g., 'cpu_usage', 'revenue').
        value: Numeric measurement value.
        recorded_at: ISO-8601 timestamp.
        tags: Optional key-value metadata.
    """
    metric_id: int
    name: str
    value: float
    recorded_at: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration.

    Attributes:
        dashboard_id: Unique identifier.
        title: Display title.
        owner_id: User who created the dashboard.
        widgets: List of widget configurations.
        refresh_interval_sec: Auto-refresh interval in seconds.
    """
    dashboard_id: int
    title: str
    owner_id: int
    widgets: list[dict[str, Any]] = field(default_factory=list)
    refresh_interval_sec: int = 60


@dataclass
class Alert:
    """Alert rule definition.

    Attributes:
        alert_id: Unique identifier.
        metric_name: Metric to monitor.
        threshold: Value that triggers the alert.
        operator: Comparison operator (gt, lt, eq, gte, lte).
        notification_channel: Where to send alerts (email, slack, pager).
    """
    alert_id: int
    metric_name: str
    threshold: float
    operator: str = "gt"
    notification_channel: str = "email"
''')

    (src / "database.py").write_text('''\
"""Database connection pool and query execution."""

import sqlite3
from contextlib import contextmanager


class ConnectionPool:
    """Manages a pool of database connections with timeout handling.

    The pool maintains up to max_connections concurrent connections.
    Connections that exceed timeout_sec are automatically closed and
    recycled on the next request.
    """

    def __init__(self, db_url: str, max_connections: int = 10, timeout_sec: int = 30):
        self.db_url = db_url
        self.max_connections = max_connections
        self.timeout_sec = timeout_sec
        self._connections: list[sqlite3.Connection] = []

    @contextmanager
    def get_connection(self):
        """Acquire a connection from the pool."""
        if self._connections:
            conn = self._connections.pop()
        else:
            conn = sqlite3.connect(self.db_url, timeout=self.timeout_sec)
            conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            if len(self._connections) < self.max_connections:
                self._connections.append(conn)
            else:
                conn.close()

    def execute(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as dicts."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE and return affected row count."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def run_migration(self, ddl_path: str) -> None:
        """Execute a DDL file to set up or migrate the database schema."""
        with open(ddl_path, "r") as f:
            ddl = f.read()
        with self.get_connection() as conn:
            conn.executescript(ddl)

    def close_all(self) -> None:
        """Close all pooled connections."""
        for conn in self._connections:
            conn.close()
        self._connections.clear()
''')

    # --- SQL Schema (code partition via SchemaChunker) ---
    (schema / "schema.sql").write_text('''\
-- Acme Analytics Database Schema

CREATE TABLE users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT    NOT NULL UNIQUE,
    full_name   TEXT    NOT NULL,
    role        TEXT    NOT NULL DEFAULT 'viewer',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE metrics (
    metric_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    value       REAL    NOT NULL,
    recorded_at TEXT    NOT NULL,
    source      TEXT    DEFAULT 'manual',
    tags        TEXT    DEFAULT '{}'
);
CREATE INDEX idx_metrics_name ON metrics (name);
CREATE INDEX idx_metrics_recorded_at ON metrics (recorded_at);

CREATE TABLE dashboards (
    dashboard_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT    NOT NULL,
    owner_id            INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    widgets             TEXT    NOT NULL DEFAULT '[]',
    refresh_interval_sec INTEGER NOT NULL DEFAULT 60,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_dashboards_owner ON dashboards (owner_id);

CREATE TABLE alerts (
    alert_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name          TEXT    NOT NULL,
    threshold            REAL    NOT NULL,
    operator             TEXT    NOT NULL DEFAULT 'gt',
    notification_channel TEXT    NOT NULL DEFAULT 'email',
    owner_id             INTEGER REFERENCES users (user_id) ON DELETE SET NULL,
    enabled              INTEGER NOT NULL DEFAULT 1,
    created_at           TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_alerts_metric ON alerts (metric_name);

CREATE TABLE deployment_log (
    deploy_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    version      TEXT    NOT NULL,
    environment  TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    deployed_by  INTEGER REFERENCES users (user_id),
    started_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    duration_ms  INTEGER
);
''')

    # --- DOCX architecture doc (doc partition) ---
    _make_docx(str(docs / "architecture.docx"), [
        "Acme Analytics Platform Architecture",
        "The system consists of three main layers: ingestion, processing, and presentation.",
        "Ingestion Layer: Metrics flow from application servers via a collector agent "
        "that pushes data to the central metrics table every 60 seconds. Each metric "
        "carries a name, numeric value, timestamp, source identifier, and optional tags.",
        "Processing Layer: The AnalyticsEngine class handles query execution and "
        "aggregation. It connects to PostgreSQL through a ConnectionPool that manages "
        "up to 10 concurrent connections with a 30-second timeout. Queries are "
        "parameterized to prevent SQL injection.",
        "Presentation Layer: Dashboards are configured per-user with customizable "
        "widgets. Each widget queries a specific metric with a time range. The "
        "refresh interval defaults to 60 seconds but can be configured per dashboard.",
        "Alerting: Alert rules monitor named metrics against thresholds. When a "
        "metric exceeds its threshold, notifications are sent via the configured "
        "channel (email, Slack, or PagerDuty). Alerts support gt, lt, eq, gte, lte operators.",
        "Database: All data is stored in PostgreSQL with row-level security enabled. "
        "The schema includes users, metrics, dashboards, alerts, and deployment_log tables. "
        "Foreign keys enforce referential integrity between dashboards/alerts and users.",
        "Deployment: Releases follow a blue-green deployment strategy. The deployment_log "
        "table tracks every release with version, environment, status, and duration.",
    ])

    # --- CSV deployment history (doc partition) ---
    lines = ["date,version,environment,status,duration_ms"]
    statuses = ["success", "success", "success", "failed", "success",
                "success", "rollback", "success", "success", "success"]
    for i in range(50):
        day = f"2026-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}"
        ver = f"2.{i // 10}.{i % 10}"
        env = ["prod", "staging", "dev"][i % 3]
        status = statuses[i % len(statuses)]
        dur = 120000 + (i * 1000) if status == "success" else 5000 + (i * 100)
        lines.append(f"{day},{ver},{env},{status},{dur}")
    (docs / "deployments.csv").write_text("\n".join(lines) + "\n")

    # --- Server log (doc partition) ---
    log_lines = []
    for i in range(200):
        ts = f"2026-04-06 {10 + (i // 60):02d}:{i % 60:02d}:00"
        if i % 50 == 0:
            log_lines.append(f"{ts} ERROR ConnectionTimeout: database pool exhausted after 30s")
        elif i % 30 == 0:
            log_lines.append(f"{ts} WARN SlowQuery: SELECT * FROM metrics took 2.3s")
        elif i % 20 == 0:
            log_lines.append(f"{ts} WARN HighMemory: heap usage at 85% (4.2GB/5GB)")
        else:
            log_lines.append(f"{ts} INFO RequestComplete: GET /api/metrics 200 45ms")
    (docs / "runbook.log").write_text("\n".join(log_lines) + "\n")

    # --- INI config (doc partition) ---
    (docs / "settings.ini").write_text("""\
[database]
host = db.acme-internal.com
port = 5432
name = acme_analytics
pool_size = 10
timeout_sec = 30
ssl_mode = require

[cache]
backend = redis
host = cache.acme-internal.com
port = 6379
ttl_seconds = 300
max_memory_mb = 512

[monitoring]
enabled = true
metrics_endpoint = http://prometheus:9090/api/v1/write
alert_email = ops-team@acme.com
alert_slack_channel = #ops-alerts
cpu_threshold_percent = 80
memory_threshold_percent = 85
latency_threshold_ms = 500
check_interval_sec = 60
""")

    # --- README (code partition, existing text chunker) ---
    (root / "README.md").write_text("""\
# Acme Analytics Platform

Internal analytics platform for monitoring business and infrastructure metrics.

## Features

- Real-time metric ingestion from application servers
- Configurable dashboards with auto-refresh widgets
- Alert rules with multi-channel notifications (email, Slack, PagerDuty)
- SQL-based query engine with date range filtering and aggregation
- Connection pooling with automatic timeout and recycling
- Blue-green deployment tracking with rollback support

## Quick Start

Configure database connection in `docs/settings.ini`, run the schema
migration via `database.py`, then start the analytics engine.
""")


# ---------------------------------------------------------------------------
# Benchmark runner (extends benchmark_suite patterns)
# ---------------------------------------------------------------------------

def _setup_engines(project_root: str) -> tuple:
    """Set up both code and doc engines for benchmark. Returns
    (code_engine, doc_engine, config, store, doc_store).
    """
    project_root = os.path.abspath(project_root)
    mnemosyne_dir = os.path.join(project_root, ".mnemosyne")

    # Clean slate
    db_path = os.path.join(mnemosyne_dir, "mnemosyne.db")
    bloom_path = os.path.join(mnemosyne_dir, "bloom.bin")
    for f in (db_path, bloom_path):
        if os.path.isfile(f):
            os.remove(f)
    os.makedirs(mnemosyne_dir, exist_ok=True)

    config = Config(root=project_root)
    config.embedding.tfidf_min_df = 1

    conn = open_store(mnemosyne_dir)
    store = Store(conn)
    doc_store = DocStore(conn)
    bloom = BloomFilter()
    tfidf = get_backend(config, store)
    doc_tfidf = get_backend(config, store=None)
    audit = AuditLog(os.path.join(mnemosyne_dir, "audit.log"))

    ingester = Ingester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf_backend=tfidf,
        audit=audit,
        doc_store=doc_store,
        doc_tfidf=doc_tfidf,
    )
    stats = ingester.ingest(full=True)

    # Code engine
    analytics = Analytics(store, config)
    analytics.start_session()
    prefetcher = Prefetcher(store)
    code_engine = RetrievalEngine(
        store=store, tfidf_backend=tfidf, config=config,
        analytics=analytics, prefetcher=prefetcher,
    )

    # Doc engine
    doc_engine = DocRetrievalEngine(
        doc_store=doc_store, tfidf_backend=doc_tfidf, config=config,
    )

    return code_engine, doc_engine, config, store, doc_store, stats


def run_question(code_engine, doc_engine, q_data: dict) -> dict:
    """Run a single question against the appropriate engine(s)."""
    partition = q_data.get("partition", "code")
    query_text = q_data["question"]
    budget = q_data.get("budget", 2000)

    t0 = time.perf_counter()

    if partition == "docs":
        results = doc_engine.query(query_text=query_text, budget=budget)
    elif partition == "all":
        code_results = code_engine.query(
            query_text=query_text, budget=budget // 2, use_compression=True,
        )
        doc_results = doc_engine.query(
            query_text=query_text, budget=budget // 2,
        )
        results = code_results + doc_results
    else:
        results = code_engine.query(
            query_text=query_text, budget=budget, use_compression=True,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    retrieved_files = sorted(set(
        r.file_path.replace("\\", "/") for r in results
    ))
    gt_files = [f.replace("\\", "/") for f in q_data["ground_truth_files"]]

    # Hit@3: any gold file in top 3 result files
    top3_files = []
    seen = set()
    for r in results:
        fp = r.file_path.replace("\\", "/")
        if fp not in seen:
            seen.add(fp)
            top3_files.append(fp)
            if len(top3_files) >= 3:
                break
    hit_at_3 = 1.0 if any(gf in top3_files for gf in gt_files) else 0.0

    # File recall: fraction of gold files found anywhere in results
    gt_found = sum(1 for gf in gt_files if gf in retrieved_files)
    file_recall = gt_found / len(gt_files) if gt_files else 0.0

    # File precision: fraction of retrieved files that are gold
    gt_set = set(gt_files)
    correct = sum(1 for rf in retrieved_files if rf in gt_set)
    file_precision = correct / len(retrieved_files) if retrieved_files else 0.0

    return {
        "id": q_data["id"],
        "partition": partition,
        "hit_at_3": hit_at_3,
        "file_recall": file_recall,
        "file_precision": file_precision,
        "retrieved_files": retrieved_files,
        "ground_truth_files": gt_files,
        "elapsed_ms": elapsed_ms,
        "result_count": len(results),
    }


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def acme_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("acme_analytics")
    build_acme_project(root)
    return root


@pytest.fixture(scope="module")
def benchmark_results(acme_project):
    """Run the full benchmark and return all question results."""
    code_engine, doc_engine, config, store, doc_store, ingest_stats = \
        _setup_engines(str(acme_project))

    questions_file = os.path.join(
        os.path.dirname(__file__),
        "benchmark_questions", "acme_analytics.json",
    )
    with open(questions_file, "r") as f:
        data = json.load(f)

    results = {}
    for q in data["questions"]:
        qr = run_question(code_engine, doc_engine, q)
        results[q["id"]] = qr

    results["_ingest_stats"] = ingest_stats
    results["_code_chunks"] = store.count_chunks()
    results["_doc_chunks"] = doc_store.count_chunks()
    return results


# ---------------------------------------------------------------------------
# Ingestion gate
# ---------------------------------------------------------------------------

class TestIngestion:
    """Verify both partitions received chunks."""

    def test_code_partition_has_chunks(self, benchmark_results):
        assert benchmark_results["_code_chunks"] > 0, \
            "Code partition should have chunks from src/*.py"

    def test_doc_partition_has_chunks(self, benchmark_results):
        assert benchmark_results["_doc_chunks"] > 0, \
            "Doc partition should have chunks from docs/*"

    def test_ingest_zero_failures(self, benchmark_results):
        stats = benchmark_results["_ingest_stats"]
        assert stats["files_failed"] == 0

    def test_all_files_indexed(self, benchmark_results):
        stats = benchmark_results["_ingest_stats"]
        assert stats["files_indexed"] >= 8, \
            f"Expected >= 8 files indexed, got {stats['files_indexed']}"


# ---------------------------------------------------------------------------
# Document retrieval gate
# ---------------------------------------------------------------------------

class TestDocRetrieval:
    """Document partition queries must find their gold files."""

    def test_d01_architecture_docx(self, benchmark_results):
        r = benchmark_results["D01"]
        assert r["file_recall"] >= 1.0, \
            f"D01: architecture.docx not found. Got: {r['retrieved_files']}"

    def test_d02_deployments_csv(self, benchmark_results):
        r = benchmark_results["D02"]
        assert r["file_recall"] >= 1.0, \
            f"D02: deployments.csv not found. Got: {r['retrieved_files']}"

    def test_d03_runbook_log(self, benchmark_results):
        r = benchmark_results["D03"]
        assert r["file_recall"] >= 1.0, \
            f"D03: runbook.log not found. Got: {r['retrieved_files']}"

    def test_d04_settings_ini(self, benchmark_results):
        r = benchmark_results["D04"]
        assert r["file_recall"] >= 1.0, \
            f"D04: settings.ini not found. Got: {r['retrieved_files']}"

    def test_doc_hit_at_3(self, benchmark_results):
        doc_ids = ["D01", "D02", "D03", "D04"]
        hits = sum(benchmark_results[qid]["hit_at_3"] for qid in doc_ids)
        avg = hits / len(doc_ids)
        assert avg >= 0.75, f"Doc hit@3 avg {avg:.2f} < 0.75"


# ---------------------------------------------------------------------------
# Schema retrieval gate
# ---------------------------------------------------------------------------

class TestSchemaRetrieval:
    """Schema queries must find the SQL file in the code partition."""

    def test_s01_metrics_table(self, benchmark_results):
        r = benchmark_results["S01"]
        assert r["file_recall"] >= 1.0, \
            f"S01: schema.sql not found. Got: {r['retrieved_files']}"

    def test_s02_foreign_keys(self, benchmark_results):
        r = benchmark_results["S02"]
        assert r["file_recall"] >= 1.0, \
            f"S02: schema.sql not found. Got: {r['retrieved_files']}"


# ---------------------------------------------------------------------------
# Code regression gate
# ---------------------------------------------------------------------------

class TestCodeRegression:
    """Code retrieval must not degrade from adding document support."""

    def test_c01_analytics_query(self, benchmark_results):
        r = benchmark_results["C01"]
        assert r["hit_at_3"] >= 1.0, \
            f"C01: analytics.py not in top 3. Got: {r['retrieved_files']}"

    def test_c02_models(self, benchmark_results):
        r = benchmark_results["C02"]
        assert r["hit_at_3"] >= 1.0, \
            f"C02: models.py not in top 3. Got: {r['retrieved_files']}"

    def test_c03_connection_pool(self, benchmark_results):
        r = benchmark_results["C03"]
        assert r["hit_at_3"] >= 1.0, \
            f"C03: database.py not in top 3. Got: {r['retrieved_files']}"


# ---------------------------------------------------------------------------
# Cross-partition gate
# ---------------------------------------------------------------------------

class TestCrossPartition:
    """Cross-partition queries must find files from both sides."""

    def test_x01_finds_code(self, benchmark_results):
        r = benchmark_results["X01"]
        code_files = {"src/analytics.py", "src/database.py"}
        found = set(r["retrieved_files"]) & code_files
        assert len(found) >= 1, \
            f"X01: no code files found. Got: {r['retrieved_files']}"

    def test_x02_finds_docs(self, benchmark_results):
        r = benchmark_results["X02"]
        assert r["file_recall"] >= 0.5, \
            f"X02: file recall {r['file_recall']:.2f} < 0.5. Got: {r['retrieved_files']}"


# ---------------------------------------------------------------------------
# Partition isolation gate
# ---------------------------------------------------------------------------

class TestPartitionIsolation:
    """Document queries must NOT return code files and vice versa."""

    def test_doc_query_returns_no_python(self, benchmark_results):
        for qid in ["D01", "D02", "D03", "D04"]:
            r = benchmark_results[qid]
            py_files = [f for f in r["retrieved_files"] if f.endswith(".py")]
            assert len(py_files) == 0, \
                f"{qid}: doc query returned Python files: {py_files}"

    def test_code_query_returns_no_docs(self, benchmark_results):
        for qid in ["C01", "C02", "C03"]:
            r = benchmark_results[qid]
            doc_files = [f for f in r["retrieved_files"]
                         if f.endswith((".csv", ".docx", ".log", ".ini"))]
            assert len(doc_files) == 0, \
                f"{qid}: code query returned doc files: {doc_files}"
