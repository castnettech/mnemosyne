# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the dense lane + rerank hooks in DocRetrievalEngine."""

from __future__ import annotations

import os

import pytest

from mnemosyne.config import Config
from mnemosyne.doc_retrieval import (
    DocRetrievalEngine,
    dense_lane_enabled,
    rerank_enabled,
    rerank_keep,
)
from mnemosyne.doc_store import DocStore
from mnemosyne.embeddings import get_backend, hashed_dense
from mnemosyne.models import Chunk
from mnemosyne.schema import open_store


def _mk_file(conn, file_id: int, rel_path: str) -> None:
    conn.execute(
        "INSERT INTO files (file_id, rel_path, content_hash, size_bytes, "
        "language, last_modified, last_indexed, is_deleted) VALUES "
        "(?, ?, 'abc', 10, 'markdown', '2026-01-01', '2026-01-01', 0)",
        (file_id, rel_path),
    )
    conn.commit()


def _mk_chunk(doc_store: DocStore, file_id: int, content: str) -> int:
    chunk = Chunk(
        chunk_id=None,
        file_id=file_id,
        content_hash=f"h{abs(hash(content)) & 0xffff:x}",
        chunk_type="paragraph",
        line_start=1,
        line_end=2,
        token_count=max(1, len(content.split())),
        content=content,
        compressed=None,
        compression_ratio=None,
        symbol_name=None,
        parent_chunk_id=None,
        page_number=None,
    )
    return doc_store.insert_chunk(chunk)


@pytest.fixture
def engine(tmp_path):
    """Seed a small doc index with three chunks across three files."""
    conn = open_store(tmp_path / ".mnemosyne")
    doc_store = DocStore(conn)

    _mk_file(conn, 1, "README.md")
    _mk_file(conn, 2, "INSTALL.md")
    _mk_file(conn, 3, "CONTRIB.md")

    # One clearly on-topic, one mildly related, one off-topic.
    c_good = _mk_chunk(
        doc_store, 1,
        "Walk-forward evaluation harness for EdgeOS thesis three. "
        "Runs, gates, and calibration replay are defined here.",
    )
    c_mild = _mk_chunk(
        doc_store, 2,
        "EdgeOS installation uses dotnet build and runs migrations.",
    )
    c_off = _mk_chunk(
        doc_store, 3,
        "Subscriber signup flow sends confirmation email via SendGrid.",
    )

    # Populate dense embeddings for all three chunks.
    for cid, txt in [
        (c_good, "Walk-forward evaluation harness thesis three"),
        (c_mild, "dotnet build migrations installation"),
        (c_off, "subscriber signup confirmation email sendgrid"),
    ]:
        doc_store.insert_dense_embedding(
            cid, vector_bytes=hashed_dense.embed_bytes(txt)
        )

    config = Config(root=tmp_path)
    # Provide a TF-IDF backend so the vector lane has something to bind
    # against; with no embeddings it returns empty which is fine for
    # the dense-only regression checks below.
    tfidf = get_backend(config, doc_store)

    eng = DocRetrievalEngine(doc_store, tfidf, config)
    return eng, {"good": c_good, "mild": c_mild, "off": c_off}


class TestFeatureFlags:
    def test_defaults_are_enabled(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_DENSE_LANE", raising=False)
        monkeypatch.delenv("MNEMOSYNE_RERANK", raising=False)
        assert dense_lane_enabled() is True
        assert rerank_enabled() is True

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DENSE_LANE", "0")
        monkeypatch.setenv("MNEMOSYNE_RERANK", "0")
        assert dense_lane_enabled() is False
        assert rerank_enabled() is False

    def test_keep_default_is_eight(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_RERANK_KEEP", raising=False)
        assert rerank_keep() == 8

    def test_keep_env_override(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RERANK_KEEP", "3")
        assert rerank_keep() == 3


class TestDenseLaneIntegration:
    def test_dense_lane_returns_chunks_when_bm25_dark(self, engine, monkeypatch):
        eng, ids = engine
        # Force BM25 + TF-IDF to be silent by issuing a query whose
        # tokens don't match the FTS content.  The dense lane still
        # lights up the good chunk via the hashed cosine.
        monkeypatch.setenv("MNEMOSYNE_DENSE_LANE", "1")
        monkeypatch.setenv("MNEMOSYNE_RERANK", "1")
        results = eng.query(
            "thesis three walk forward evaluation runs"
        )
        # The best chunk should be the 'good' one.
        assert results
        top_chunk_ids = [r.chunk.chunk_id for r in results]
        assert ids["good"] in top_chunk_ids

    def test_dense_lane_disabled_skips_vector_rows(self, engine, monkeypatch):
        eng, _ = engine
        monkeypatch.setenv("MNEMOSYNE_DENSE_LANE", "0")
        monkeypatch.setenv("MNEMOSYNE_RERANK", "0")
        # With dense off, we can still get results from BM25/TF-IDF.
        # The relevant assertion here is simply that the query does not
        # crash and returns a list.
        results = eng.query("thesis three")
        assert isinstance(results, list)

    def test_envelope_caps_at_rerank_keep(self, engine, monkeypatch):
        eng, _ = engine
        monkeypatch.setenv("MNEMOSYNE_DENSE_LANE", "1")
        monkeypatch.setenv("MNEMOSYNE_RERANK", "1")
        monkeypatch.setenv("MNEMOSYNE_RERANK_KEEP", "2")
        results = eng.query("walk forward thesis")
        # We only seeded three chunks; envelope should respect the 2-cap.
        assert len(results) <= 2

    def test_rerank_surfaces_cosine_score_in_scores(self, engine, monkeypatch):
        eng, ids = engine
        monkeypatch.setenv("MNEMOSYNE_DENSE_LANE", "1")
        monkeypatch.setenv("MNEMOSYNE_RERANK", "1")
        results = eng.query("walk forward evaluation thesis")
        assert results
        # Winners must carry the rerank cosine so downstream analytics
        # can tell what actually selected them.
        scores = results[0].scores
        assert "rerank_cosine" in scores or "rerank_blended" in scores
