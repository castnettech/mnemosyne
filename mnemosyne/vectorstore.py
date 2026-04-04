# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
In-memory ANN vector store abstraction for Mnemosyne.

Provides a brute-force cosine-similarity search over sparse TF-IDF vectors.
The design is intentionally simple and dependency-free: it holds all vectors
in a plain dict and performs exhaustive linear scan at query time.

For corpora where the number of indexed chunks stays under ~50 000 this is
fast enough (each search is O(k * |query_terms|) via the inverted index in
:class:`~mnemosyne.embeddings.tfidf_backend.TFIDFBackend`).  If a future
deployment requires sub-linear ANN, the same interface can be backed by an
HNSW or IVF implementation by swapping the ``search`` method.

Persistence is delegated to *store* through two optional methods:
- ``store.load_vectors() -> dict[int, dict[str, float]]``
- ``store.save_vectors(mapping: dict[int, dict[str, float]]) -> None``
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # store type is intentionally left untyped to avoid circular imports


class VectorStore:
    """
    Brute-force sparse-vector store.

    All vectors are sparse dicts ``{term: weight}`` as produced by
    :meth:`~mnemosyne.embeddings.tfidf_backend.TFIDFBackend.embed`.
    Cosine similarity is computed inline without needing the embedding backend
    -- the backend is only needed to produce the query vector before calling
    :meth:`search`.
    """

    def __init__(self, store=None) -> None:
        """
        Args:
            store: Optional persistent store object.  If provided,
                   :meth:`load_all` will populate the in-memory dict from it.
        """
        self.store = store
        self._vectors: dict[int, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunk_id: int, vector: dict[str, float]) -> None:
        """
        Insert or replace the vector for *chunk_id*.

        Args:
            chunk_id: Integer primary key of the chunk.
            vector:   Sparse TF-IDF vector ``{term: weight}``.
        """
        self._vectors[chunk_id] = vector

    def remove(self, chunk_id: int) -> None:
        """
        Remove the vector for *chunk_id* if it exists.

        Silently a no-op when *chunk_id* is not present.

        Args:
            chunk_id: Integer primary key of the chunk to remove.
        """
        self._vectors.pop(chunk_id, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: dict[str, float],
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Return the *top_k* most similar chunks by cosine similarity.

        Only chunks that share at least one term with *query_vector* receive a
        non-zero score, so the scan is effectively sparse.

        Args:
            query_vector: Sparse TF-IDF vector for the query (same space as
                          stored vectors).
            top_k:        Maximum number of results.

        Returns:
            List of ``(chunk_id, similarity)`` tuples sorted by descending
            similarity.  May be shorter than *top_k*.
        """
        if not query_vector or not self._vectors:
            return []

        query_norm = math.sqrt(sum(v * v for v in query_vector.values()))
        if query_norm == 0.0:
            return []

        scores: list[tuple[int, float]] = []

        for chunk_id, doc_vec in self._vectors.items():
            # Dot product over shared terms only -- O(|query|) per doc when
            # query is much smaller than the vocabulary.
            common = set(query_vector.keys()) & set(doc_vec.keys())
            if not common:
                continue

            dot = sum(query_vector[t] * doc_vec[t] for t in common)
            doc_norm = math.sqrt(sum(v * v for v in doc_vec.values()))

            if doc_norm == 0.0:
                continue

            cosine = dot / (query_norm * doc_norm)
            scores.append((chunk_id, cosine))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """
        Populate the in-memory dict from the persistent store.

        Calls ``store.load_vectors()`` if the store exposes that method.
        The return value must be a ``dict[int, dict[str, float]]`` or None.

        Silently a no-op when no store is attached or the method is absent.
        """
        if self.store is None:
            return
        load_fn = getattr(self.store, "load_vectors", None)
        if load_fn is None:
            return
        payload = load_fn()
        if payload:
            # Ensure keys are int (they may come back as str from JSON storage)
            self._vectors = {int(k): v for k, v in payload.items()}

    def save_all(self) -> None:
        """
        Flush the in-memory dict to the persistent store.

        Calls ``store.save_vectors(mapping)`` if the store exposes that method.
        Silently a no-op when no store is attached.
        """
        if self.store is None:
            return
        save_fn = getattr(self.store, "save_vectors", None)
        if save_fn is None:
            return
        save_fn(self._vectors)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of vectors currently held in the store."""
        return len(self._vectors)

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, chunk_id: int) -> bool:
        return chunk_id in self._vectors
