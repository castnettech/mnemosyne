# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the generic cosine-over-BLOB vector search helper.

The scoring assertions need numpy (an optional, lazily-imported dependency),
so they are gated.  The pure-Python identifier guard and the empty/zero-norm
short-circuit run unconditionally.
"""

from __future__ import annotations

import sqlite3
import struct

import pytest

from mnemosyne.embeddings import vector_search as vs

try:
    import numpy as _np  # noqa: F401

    _HAVE_NUMPY = True
except ImportError:
    _HAVE_NUMPY = False

requires_numpy = pytest.mark.skipif(
    not _HAVE_NUMPY, reason="numpy not installed (optional dense dependency)"
)


# ---------------------------------------------------------------------------
# Encoding helpers for hand-built fixture vectors
# ---------------------------------------------------------------------------


def _int8_blob(floats: list[float]) -> bytes:
    q = [max(-127, min(127, int(round(x * 127)))) for x in floats]
    return struct.pack(f"{len(q)}b", *q)


def _float32_blob(floats: list[float]) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


# ---------------------------------------------------------------------------
# Pure-logic tests -- always run
# ---------------------------------------------------------------------------


class TestIdentifierGuard:
    def test_accepts_plain_identifiers(self):
        assert vs._is_safe_identifier("turn_embeddings")
        assert vs._is_safe_identifier("_private")
        assert vs._is_safe_identifier("col123")

    def test_rejects_injection_attempts(self):
        assert not vs._is_safe_identifier("")
        assert not vs._is_safe_identifier("1col")
        assert not vs._is_safe_identifier("a b")
        assert not vs._is_safe_identifier("t;DROP TABLE x")
        assert not vs._is_safe_identifier("col-name")
        assert not vs._is_safe_identifier('col"name')

    def test_over_table_rejects_bad_identifier(self):
        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError):
            vs.cosine_topk_over_table(
                conn,
                [0.1, 0.2],
                table="t; DROP TABLE x",
                id_column="id",
                vector_column="vec",
                dim=2,
            )

    def test_over_table_rejects_bad_encoding(self):
        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError):
            vs.cosine_topk_over_table(
                conn,
                [0.1, 0.2],
                table="t",
                id_column="id",
                vector_column="vec",
                dim=2,
                encoding="float16",
            )


class TestEmptyShortCircuits:
    def test_empty_query_returns_empty(self):
        assert vs.cosine_topk([], [(1, b"\x01\x02")], dim=2) == []

    def test_empty_candidate_set_returns_empty(self):
        conn = sqlite3.connect(":memory:")
        out = vs.cosine_topk_over_table(
            conn,
            [0.1, 0.2],
            table="t",
            id_column="id",
            vector_column="vec",
            dim=2,
            candidate_ids=[],
        )
        assert out == []


# ---------------------------------------------------------------------------
# Scoring tests -- require numpy
# ---------------------------------------------------------------------------


@requires_numpy
class TestCosineTopK:
    def test_ranks_by_similarity_int8(self):
        # 3-dim hand-built vectors; query points along axis 0.
        query = [1.0, 0.0, 0.0]
        rows = [
            (10, _int8_blob([1.0, 0.0, 0.0])),   # identical   -> ~1.0
            (20, _int8_blob([0.0, 1.0, 0.0])),   # orthogonal  -> ~0.0
            (30, _int8_blob([0.7, 0.7, 0.0])),   # 45 degrees  -> ~0.7
        ]
        out = vs.cosine_topk(query, rows, dim=3, encoding="int8", top_k=3)
        ids = [r[0] for r in out]
        assert ids == [10, 30, 20]
        assert out[0][1] > out[1][1] > out[2][1]
        assert out[0][1] == pytest.approx(1.0, abs=0.02)

    def test_topk_truncates(self):
        query = [1.0, 0.0]
        rows = [(i, _int8_blob([1.0, 0.0])) for i in range(5)]
        out = vs.cosine_topk(query, rows, dim=2, encoding="int8", top_k=2)
        assert len(out) == 2

    def test_float32_encoding(self):
        query = [1.0, 0.0, 0.0]
        rows = [
            (1, _float32_blob([1.0, 0.0, 0.0])),
            (2, _float32_blob([-1.0, 0.0, 0.0])),  # opposite -> -1.0
        ]
        out = vs.cosine_topk(query, rows, dim=3, encoding="float32", top_k=2)
        assert out[0][0] == 1
        assert out[0][1] == pytest.approx(1.0, abs=1e-5)
        assert out[1][1] == pytest.approx(-1.0, abs=1e-5)

    def test_wrong_length_blob_is_skipped(self):
        query = [1.0, 0.0, 0.0]
        rows = [
            (1, _int8_blob([1.0, 0.0, 0.0])),  # dim 3 -- kept
            (2, _int8_blob([1.0, 0.0])),       # dim 2 -- skipped
        ]
        out = vs.cosine_topk(query, rows, dim=3, encoding="int8", top_k=5)
        assert [r[0] for r in out] == [1]

    def test_zero_norm_stored_vector_skipped(self):
        query = [1.0, 0.0, 0.0]
        rows = [
            (1, _int8_blob([0.0, 0.0, 0.0])),  # zero vector -- skipped
            (2, _int8_blob([1.0, 0.0, 0.0])),
        ]
        out = vs.cosine_topk(query, rows, dim=3, encoding="int8", top_k=5)
        assert [r[0] for r in out] == [2]

    def test_query_wrong_dim_returns_empty(self):
        out = vs.cosine_topk(
            [1.0, 0.0], [(1, _int8_blob([1.0, 0.0, 0.0]))], dim=3
        )
        assert out == []


@requires_numpy
class TestCosineTopKOverTable:
    @pytest.fixture
    def conn(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE turn_vectors (turn_id INTEGER PRIMARY KEY, vec BLOB)")
        rows = [
            (101, _int8_blob([1.0, 0.0, 0.0])),
            (102, _int8_blob([0.0, 1.0, 0.0])),
            (103, _int8_blob([0.6, 0.8, 0.0])),
        ]
        c.executemany("INSERT INTO turn_vectors VALUES (?, ?)", rows)
        c.commit()
        return c

    def test_full_scan_ranks_correctly(self, conn):
        out = vs.cosine_topk_over_table(
            conn,
            [1.0, 0.0, 0.0],
            table="turn_vectors",
            id_column="turn_id",
            vector_column="vec",
            dim=3,
            top_k=3,
        )
        assert [r[0] for r in out] == [101, 103, 102]

    def test_candidate_filter_restricts_scan(self, conn):
        out = vs.cosine_topk_over_table(
            conn,
            [1.0, 0.0, 0.0],
            table="turn_vectors",
            id_column="turn_id",
            vector_column="vec",
            dim=3,
            candidate_ids=[102, 103],
        )
        ids = [r[0] for r in out]
        assert 101 not in ids
        assert set(ids) == {102, 103}
        assert ids[0] == 103  # closer to the query than 102

    def test_missing_table_returns_empty(self, conn):
        out = vs.cosine_topk_over_table(
            conn,
            [1.0, 0.0, 0.0],
            table="turn_vectors",  # valid name, but query nonexistent column
            id_column="turn_id",
            vector_column="does_not_exist",
            dim=3,
        )
        assert out == []
