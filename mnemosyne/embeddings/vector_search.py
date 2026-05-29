# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Generic brute-force cosine search over packed vector BLOBs.

This module is storage-agnostic and partition-agnostic.  It scores a single
query vector against a set of stored vectors -- each stored as a packed binary
BLOB in some column of some table -- and returns the top-k by cosine
similarity.  Callers parameterize the table, the id column, the vector column,
and the on-disk encoding (int8 or float32), so the same routine serves any
partition that keeps dense vectors as a SQLite BLOB.

Design notes
------------
- Pure brute force + numpy.  No ANN index, no sqlite extension, no loadable
  module.  At the scales this engine targets (tens of thousands of short
  records) a linear scan in numpy is fast and has zero external moving parts.
- Two layers are exposed:
    * :func:`cosine_topk` -- the pure-compute core.  Takes a query vector and
      an iterable of ``(id, vector_bytes)`` pairs.  No database knowledge, so
      it is trivially unit-testable and reusable from any backing store.
    * :func:`cosine_topk_over_table` -- a thin SQLite convenience wrapper that
      SELECTs ``(id_column, vector_column)`` from ``table`` (optionally
      restricted to a candidate id set) and feeds the rows to
      :func:`cosine_topk`.
- Vectors are assumed L2-normalized at store time (the int8 and float32
  encoders in this package both normalize before packing).  The query is
  re-normalized defensively here so a caller passing an un-normalized query
  still gets correct cosine ordering.
- Encoding contract:
    * ``"int8"``   -> 1 byte per dim, signed, value/127.0 maps back to floats.
    * ``"float32"`` -> 4 bytes per dim, little-endian IEEE-754.
  A stored BLOB whose length does not match ``dim`` for the chosen encoding is
  skipped (it belongs to a different model_version / dim).
- numpy is imported lazily inside the functions, mirroring the rest of the
  embeddings package, so importing this module never hard-requires numpy.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as _np


# ---------------------------------------------------------------------------
# Encoding contract
# ---------------------------------------------------------------------------

#: Bytes-per-dimension for each supported on-disk encoding.
_BYTES_PER_DIM = {"int8": 1, "float32": 4}


def _decode_blob(vec_bytes: bytes, dim: int, encoding: str, np):
    """Decode a stored BLOB into a float32 numpy vector, or ``None``.

    Returns ``None`` when the BLOB length does not match ``dim`` for the
    encoding (a vector from a different model/dim -- skip it rather than crash).
    """
    bpd = _BYTES_PER_DIM.get(encoding)
    if bpd is None:
        raise ValueError(f"unsupported encoding: {encoding!r}")
    if not vec_bytes or len(vec_bytes) != dim * bpd:
        return None
    if encoding == "int8":
        arr = np.frombuffer(vec_bytes, dtype=np.int8).astype(np.float32)
        arr = arr / 127.0
    else:  # float32
        # .copy() so the array owns writable memory (frombuffer is read-only).
        arr = np.frombuffer(vec_bytes, dtype="<f4").astype(np.float32).copy()
    return arr


def cosine_topk(
    query_vec: Sequence[float],
    rows: Iterable[tuple[int, bytes]],
    *,
    dim: int,
    encoding: str = "int8",
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """Score ``query_vec`` against stored vector BLOBs; return top-k.

    Args:
        query_vec: The query vector (length ``dim``).  Re-normalized here, so it
            need not be pre-normalized.
        rows: Iterable of ``(id, vector_bytes)`` pairs.  ``id`` is opaque to this
            function (chunk id, turn id, row id -- caller's choice).
        dim: Expected vector dimensionality.  BLOBs of any other length are
            skipped.
        encoding: ``"int8"`` (1 byte/dim) or ``"float32"`` (4 bytes/dim).
        top_k: Maximum number of results to return.

    Returns:
        A list of ``(id, similarity)`` tuples sorted by similarity descending,
        truncated to ``top_k``.  Empty when the query is empty/zero-norm, numpy
        is unavailable, or no stored vector matches ``dim``.
    """
    if not query_vec:
        return []
    try:
        import numpy as np
    except ImportError:  # numpy is an optional, lazy dependency here
        return []

    q = np.asarray(query_vec, dtype=np.float32)
    if q.shape[0] != dim:
        return []
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return []
    q = q / q_norm

    results: list[tuple[int, float]] = []
    for row_id, vec_bytes in rows:
        v = _decode_blob(vec_bytes, dim, encoding, np)
        if v is None:
            continue
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0.0:
            continue
        sim = float(np.dot(q, v) / v_norm)
        results.append((int(row_id), sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def cosine_topk_over_table(
    conn: sqlite3.Connection,
    query_vec: Sequence[float],
    *,
    table: str,
    id_column: str,
    vector_column: str,
    dim: int,
    encoding: str = "int8",
    top_k: int = 20,
    candidate_ids: Sequence[int] | None = None,
) -> list[tuple[int, float]]:
    """Brute-force cosine top-k over a SQLite table of vector BLOBs.

    Generic over the table / id column / vector column so any partition that
    stores dense vectors as a BLOB can call it without bespoke SQL.

    Args:
        conn: An open ``sqlite3.Connection``.
        query_vec: The query vector (length ``dim``); re-normalized internally.
        table: Table name holding the vectors.
        id_column: Name of the integer id column to return.
        vector_column: Name of the BLOB column holding packed vectors.
        dim: Expected vector dimensionality.
        encoding: ``"int8"`` or ``"float32"`` (see :func:`cosine_topk`).
        top_k: Maximum results.
        candidate_ids: If given, only these ids are scored (pre-filtered by a
            lexical lane, say).  ``None`` scans the whole table.

    Returns:
        ``[(id, similarity), ...]`` descending by similarity, length <= top_k.

    Security / robustness:
        ``table``, ``id_column``, and ``vector_column`` are validated as plain
        SQL identifiers (alphanumeric + underscore) before interpolation, since
        SQLite cannot parameterize identifiers.  Anything else raises
        ``ValueError`` -- callers pass their own schema names, never user input,
        but this keeps the helper safe by construction.
    """
    for ident in (table, id_column, vector_column):
        if not _is_safe_identifier(ident):
            raise ValueError(f"unsafe SQL identifier: {ident!r}")
    if encoding not in _BYTES_PER_DIM:
        raise ValueError(f"unsupported encoding: {encoding!r}")

    if candidate_ids is not None and len(candidate_ids) == 0:
        return []

    sql = f"SELECT {id_column}, {vector_column} FROM {table}"
    params: tuple = ()
    if candidate_ids is not None:
        placeholders = ",".join("?" for _ in candidate_ids)
        sql += f" WHERE {id_column} IN ({placeholders})"
        params = tuple(int(cid) for cid in candidate_ids)

    try:
        cursor = conn.execute(sql, params)
        rows = ((row[0], row[1]) for row in cursor.fetchall())
    except sqlite3.Error:
        return []

    return cosine_topk(
        query_vec, rows, dim=dim, encoding=encoding, top_k=top_k
    )


def _is_safe_identifier(name: str) -> bool:
    """True if *name* is a bare SQL identifier (letters/digits/underscore)."""
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)


__all__ = [
    "cosine_topk",
    "cosine_topk_over_table",
]
