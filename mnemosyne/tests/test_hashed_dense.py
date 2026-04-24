# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the hashed-TFIDF dense embedder and its DocStore CRUD."""

from __future__ import annotations

import math

import pytest

from mnemosyne.config import Config
from mnemosyne.doc_store import DocStore
from mnemosyne.embeddings import hashed_dense
from mnemosyne.models import Chunk
from mnemosyne.schema import open_store


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / ".mnemosyne")
    # Create a fake file record so doc_chunks FK is satisfied.
    c.execute(
        "INSERT INTO files (file_id, rel_path, content_hash, size_bytes, "
        "language, last_modified, last_indexed, is_deleted) VALUES "
        "(1, 'README.md', 'deadbeef', 120, 'markdown', '2026-01-01', "
        "'2026-01-01', 0)"
    )
    c.commit()
    return c


@pytest.fixture
def doc_store(conn):
    return DocStore(conn)


def _mk_chunk(doc_store, content: str, chunk_id: int | None = None) -> int:
    chunk = Chunk(
        chunk_id=None,
        file_id=1,
        content_hash=f"h{hash(content) & 0xffff:x}",
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


class TestEmbedText:
    def test_empty_input_returns_zero_vector(self):
        vec = hashed_dense.embed_bytes("")
        assert len(vec) == hashed_dense.DIM
        assert vec == b"\x00" * hashed_dense.DIM

    def test_dim_and_quantization_constants_stable(self):
        # These constants are part of the storage contract with the
        # schema migration + the muses turn-embedding lane.  Changing
        # them without a schema bump would break existing indexes.
        assert hashed_dense.DIM == 128
        assert hashed_dense.QUANTIZATION == "int8"
        assert hashed_dense.MODEL_ID == "hashed_tfidf_v1"
        assert hashed_dense.MODEL_VERSION == 1

    def test_nonempty_text_produces_nonzero_bytes(self):
        vec = hashed_dense.embed_bytes(
            "EdgeOS truth-first learning Brain calibration"
        )
        assert len(vec) == 128
        assert any(b != 0 for b in vec)

    def test_decode_round_trip_is_unit_normalised(self):
        floats = hashed_dense.embed_floats(
            "walk forward evaluation thesis three"
        )
        # L2 norm of the float vector should be ~1 (or 0 for empty).
        norm = math.sqrt(sum(x * x for x in floats))
        assert 0.99 <= norm <= 1.01 or norm == 0.0

    def test_cosine_is_commutative_and_self_similar(self):
        a = hashed_dense.embed_floats("option calibration snapshot")
        assert hashed_dense.cosine(a, a) > 0.99
        b = hashed_dense.embed_floats("option calibration snapshot")
        assert abs(hashed_dense.cosine(a, b) - 1.0) < 1e-6

    def test_cosine_separates_related_and_unrelated_text(self):
        a = hashed_dense.embed_floats(
            "walk forward evaluation harness thesis three"
        )
        related = hashed_dense.embed_floats(
            "thesis three walk forward evaluation runs"
        )
        unrelated = hashed_dense.embed_floats(
            "regbrief monthly digest subscriber email pipeline"
        )
        # Cosine of related should be strictly larger than cosine of
        # unrelated against the same anchor.
        assert hashed_dense.cosine(a, related) > hashed_dense.cosine(
            a, unrelated
        )

    def test_zero_norm_vector_yields_zero_cosine(self):
        zero = [0.0] * hashed_dense.DIM
        a = hashed_dense.embed_floats("edge scorer options strategy")
        assert hashed_dense.cosine(zero, a) == 0.0
        assert hashed_dense.cosine(a, zero) == 0.0


class TestDocStoreDenseCrud:
    def test_insert_and_get_roundtrip(self, doc_store):
        cid = _mk_chunk(doc_store, "first chunk")
        v = hashed_dense.embed_bytes("first chunk")
        doc_store.insert_dense_embedding(cid, vector_bytes=v)
        got = doc_store.get_dense_embedding(cid)
        assert got is not None
        vec_bytes, dim = got
        assert dim == 128
        assert vec_bytes == v

    def test_batch_fetch_returns_only_requested(self, doc_store):
        c1 = _mk_chunk(doc_store, "alpha")
        c2 = _mk_chunk(doc_store, "beta")
        c3 = _mk_chunk(doc_store, "gamma")
        for cid, txt in [(c1, "alpha"), (c2, "beta"), (c3, "gamma")]:
            doc_store.insert_dense_embedding(
                cid, vector_bytes=hashed_dense.embed_bytes(txt)
            )

        batch = doc_store.get_dense_embeddings_batch([c1, c3])
        assert set(batch.keys()) == {c1, c3}
        assert c2 not in batch
        assert len(batch[c1][0]) == 128

    def test_batch_fetch_all_when_no_ids(self, doc_store):
        c1 = _mk_chunk(doc_store, "first")
        c2 = _mk_chunk(doc_store, "second")
        for cid, txt in [(c1, "first"), (c2, "second")]:
            doc_store.insert_dense_embedding(
                cid, vector_bytes=hashed_dense.embed_bytes(txt)
            )
        batch = doc_store.get_dense_embeddings_batch()
        assert set(batch.keys()) == {c1, c2}

    def test_count_dense_embeddings(self, doc_store):
        c1 = _mk_chunk(doc_store, "alpha")
        c2 = _mk_chunk(doc_store, "beta")
        doc_store.insert_dense_embedding(
            c1, vector_bytes=hashed_dense.embed_bytes("alpha")
        )
        assert doc_store.count_dense_embeddings() == 1
        doc_store.insert_dense_embedding(
            c2, vector_bytes=hashed_dense.embed_bytes("beta")
        )
        assert doc_store.count_dense_embeddings() == 2

    def test_newer_model_version_wins_on_read(self, doc_store):
        cid = _mk_chunk(doc_store, "shared")
        v1 = hashed_dense.embed_bytes("shared")
        v2 = hashed_dense.embed_bytes("SHARED LATER")
        doc_store.insert_dense_embedding(
            cid, vector_bytes=v1, model_version=1
        )
        doc_store.insert_dense_embedding(
            cid, vector_bytes=v2, model_version=2
        )
        # Single-id getter should return the newer row.
        got_single = doc_store.get_dense_embedding(cid)
        assert got_single is not None
        assert got_single[0] == v2

        # Batch getter should too.
        batch = doc_store.get_dense_embeddings_batch([cid])
        assert batch[cid][0] == v2

    def test_iter_chunks_missing_dense(self, doc_store):
        c_with = _mk_chunk(doc_store, "has dense")
        c_without = _mk_chunk(doc_store, "no dense")
        doc_store.insert_dense_embedding(
            c_with, vector_bytes=hashed_dense.embed_bytes("has dense")
        )
        missing = [cid for cid, _ in
                   doc_store.iter_chunks_missing_dense()]
        assert c_with not in missing
        assert c_without in missing
