# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pure-Python TF-IDF sparse embedding backend for Mnemosyne.

No external dependencies.  All maths uses the Python standard library
(``math``, ``re``, ``collections``).

Design notes
------------
- **Augmented TF**: ``0.5 + 0.5 * (count / max_count)`` prevents bias toward
  long documents while still rewarding repeated occurrence within a chunk.
- **Smoothed IDF**: ``log((N+1) / (df+1)) + 1`` avoids division-by-zero and
  gives non-zero weight to terms that appear in every document.
- **Inverted index**: built lazily and used for sub-linear query time — only
  terms actually present in the query are consulted.
- **Identifier splitting**: camelCase and snake_case tokens are expanded into
  their component parts, giving better recall for code search.
- **Vocabulary persistence**: delegated to *store* (injected); the backend
  only calls ``store.save_vocabulary`` / ``store.load_vocabulary`` with a
  plain-dict payload so it is decoupled from the store's internal format.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from mnemosyne.embeddings.stemmer import stem as _porter_stem

if TYPE_CHECKING:
    from mnemosyne.config import Config

logger = logging.getLogger(__name__)


# Common English stopwords filtered from tokenisation to prevent prose-heavy
# documents (HTML, Markdown) from inflating similarity scores with generic terms.
# Must stay in sync with retrieval._STOPWORDS — both lists filter the same
# natural-language noise.  Missing entries cause token leaks (e.g. "is" from
# camelCase-split "isNegated" inflating scores for any is* identifier).
_STOPWORDS: frozenset[str] = frozenset({
    "the", "be", "to", "of", "and", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "no", "just",
    "him", "know", "take", "into", "your", "some", "could", "them",
    "see", "other", "than", "then", "now", "its", "also", "after",
    "use", "how", "our", "any", "these", "most", "may", "did", "does",
    "are", "is", "was", "were", "been", "has", "had", "being",
})


class TFIDFBackend:
    """
    TF-IDF sparse vector embedding and retrieval backend.

    Lifecycle::

        backend = TFIDFBackend(config, store)
        backend.build_vocabulary(list_of_texts)     # once, or after re-index
        vectors = [(id, backend.embed(text)) for id, text in chunks]
        backend.build_inverted_index(vectors)        # once per index build
        results = backend.search("query text", top_k=10)

    The :meth:`embed` method can also be called independently to get a sparse
    vector for any text (e.g. for similarity comparisons).
    """

    # Vocabulary payload key used when persisting to/from the store
    _VOCAB_KEY = "tfidf_vocabulary"

    # Regex pattern used by tokenize() — kept as a class constant so the
    # hash can reference it deterministically.
    _TOKEN_PATTERN: str = r"[a-zA-Z_][a-zA-Z0-9_]{2,}"
    _MIN_SUBTOKEN_LEN: int = 2

    def __init__(self, config: "Config", store=None) -> None:
        self.max_features: int = config.embedding.tfidf_max_features
        self.min_df: int = config.embedding.tfidf_min_df
        self.store = store

        self.vocabulary: dict[str, int] = {}          # term -> document_frequency
        self.idf: dict[str, float] = {}               # term -> idf_weight
        self.total_docs: int = 0
        self._inverted_index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._vocabulary_stale: bool = False

        # Attempt to load a previously persisted vocabulary.
        if store is not None:
            self._load_vocabulary()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Backend identifier string."""
        return "tfidf"

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """
        Extract normalised term tokens from *text*.

        Behaviour:
        - Extract alphanumeric + underscore tokens of at least 3 characters.
        - Convert to lowercase.
        - Split camelCase tokens into their components (``getUserById`` →
          ``getuser``, ``get``, ``user``, ``by``, ``id``).
        - Split snake_case tokens as well (already covered by ``_`` separator).
        - Deduplicate expanded parts, but preserve original tokens too so that
          exact-match queries still score highly.

        Returns:
            List of lowercase string tokens (may contain duplicates for
            frequency counting purposes).
        """
        raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
        expanded: list[str] = []
        for tok in raw_tokens:
            expanded.append(tok)
            # Split camelCase: insert separator before each uppercase run
            # Note: tok is already lowercased, so we work on original casing
            # We re-extract from text to get original casing for splitting.
            pass  # camelCase splitting handled below on original tokens

        # Re-tokenize on original text to get proper casing for camelCase split
        raw_original = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text)
        result: list[str] = []
        for tok in raw_original:
            lower_tok = tok.lower()
            if lower_tok in _STOPWORDS:
                continue
            result.append(lower_tok)
            # Split camelCase: MyClass -> my, class; getUserById -> get, user, by, id
            # Insert underscores before uppercase letters that follow lowercase letters
            snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", tok)
            # Also split on existing underscores
            parts = snake.lower().split("_")
            if len(parts) > 1:
                result.extend(p for p in parts if len(p) >= 2 and p not in _STOPWORDS)

        # Apply Porter stemming so TF-IDF aligns with BM25/FTS5 porter tokeniser.
        return [_porter_stem(tok) for tok in result]

    # ------------------------------------------------------------------
    # Tokenizer version tracking
    # ------------------------------------------------------------------

    def _compute_tokenizer_hash(self) -> str:
        """
        Return a deterministic SHA-256 hex digest encoding the current
        tokenizer configuration: sorted stopwords, token regex pattern,
        and minimum sub-token length.

        A change in any of these parameters invalidates previously built
        vocabulary/index data.
        """
        canonical_parts = [
            "stopwords:" + ",".join(sorted(_STOPWORDS)),
            "pattern:" + self._TOKEN_PATTERN,
            "min_subtoken_len:" + str(self._MIN_SUBTOKEN_LEN),
            "stemmer:porter",
        ]
        canonical = "\n".join(canonical_parts)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    def build_vocabulary(self, texts: list[str]) -> None:
        """
        Build the TF-IDF vocabulary from a corpus of *texts*.

        Only terms appearing in at least ``min_df`` documents are retained.
        The vocabulary is then capped at ``max_features`` terms, preferring the
        rarest terms (highest IDF) because they are most discriminative.

        Args:
            texts: List of document strings (one per chunk or file).
        """
        self.total_docs = len(texts)
        doc_freqs: Counter[str] = Counter()

        for text in texts:
            # Use a set to count each term once per document (document frequency)
            terms_in_doc = set(self.tokenize(text))
            for term in terms_in_doc:
                doc_freqs[term] += 1

        # Filter: must appear in at least min_df documents
        filtered: dict[str, int] = {
            term: df for term, df in doc_freqs.items() if df >= self.min_df
        }

        # Cap at max_features, keeping the rarest terms (lowest document frequency
        # = highest IDF = most discriminative for retrieval).
        sorted_terms = sorted(filtered.items(), key=lambda x: x[1])
        top_terms = sorted_terms[: self.max_features]

        self.vocabulary = dict(top_terms)
        self._compute_idf()
        self._vocabulary_stale = False

        # Persist the updated vocabulary and tokenizer hash
        self._save_vocabulary()
        if self.store is not None:
            set_fn = getattr(self.store, "set_index_metadata", None)
            if set_fn is not None:
                set_fn("tokenizer_hash", self._compute_tokenizer_hash())

    def _compute_idf(self) -> None:
        """Compute smoothed IDF weights for all vocabulary terms."""
        N = max(1, self.total_docs)
        self.idf = {
            term: math.log((N + 1) / (df + 1)) + 1.0
            for term, df in self.vocabulary.items()
        }

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> dict[str, float]:
        """
        Compute a sparse TF-IDF vector for *text*.

        Only vocabulary terms are included in the output vector.  Terms not
        in the vocabulary are silently ignored.

        The **augmented TF** formula ``0.5 + 0.5 * (count / max_count)``
        normalises for document length while preserving relative frequencies.

        Args:
            text: Any string (query, chunk content, etc.).

        Returns:
            Sparse vector as ``{term: weight}`` dict.  Empty dict if *text*
            contains no vocabulary terms.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        tf: Counter[str] = Counter(tokens)
        max_tf = max(tf.values())

        vector: dict[str, float] = {}
        for term, count in tf.items():
            if term in self.idf:
                augmented_tf = 0.5 + 0.5 * (count / max_tf)
                vector[term] = augmented_tf * self.idf[term]

        return vector

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similarity(
        self,
        vec_a: dict[str, float],
        vec_b: dict[str, float],
    ) -> float:
        """
        Cosine similarity between two sparse TF-IDF vectors.

        Returns a value in ``[0.0, 1.0]``.  Returns ``0.0`` when either
        vector is empty or when there are no common terms.

        Args:
            vec_a: Sparse vector (term -> weight) from :meth:`embed`.
            vec_b: Sparse vector (term -> weight) from :meth:`embed`.

        Returns:
            Float cosine similarity.
        """
        if not vec_a or not vec_b:
            return 0.0

        common_terms = set(vec_a.keys()) & set(vec_b.keys())
        if not common_terms:
            return 0.0

        dot_product = sum(vec_a[t] * vec_b[t] for t in common_terms)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Inverted index
    # ------------------------------------------------------------------

    def build_inverted_index(
        self, chunk_vectors: list[tuple[int, dict[str, float]]]
    ) -> None:
        """
        Build an in-memory inverted index for fast query-time retrieval.

        The index maps each vocabulary term to the list of (chunk_id, weight)
        pairs where the term appears, enabling sub-linear search: only chunks
        that share at least one term with the query are scored.

        Args:
            chunk_vectors: List of ``(chunk_id, sparse_vector)`` tuples as
                           returned by iterating :meth:`embed` over the corpus.
        """
        self._inverted_index.clear()
        for chunk_id, vector in chunk_vectors:
            for term, weight in vector.items():
                self._inverted_index[term].append((chunk_id, weight))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Retrieve the top-*k* most relevant chunk IDs for *query*.

        Uses the inverted index to accumulate partial dot-product scores
        efficiently — only terms present in the query are looked up.  The
        query vector norm is constant across all candidates so it is omitted
        from the ranking comparison (relative order is preserved).

        If the vocabulary was built with a different tokenizer version
        (``_vocabulary_stale is True``), returns an empty list and logs a
        rebuild instruction.

        Args:
            query:  Raw query string.
            top_k:  Maximum number of results to return.

        Returns:
            List of ``(chunk_id, score)`` tuples sorted by descending score.
            May be shorter than *top_k* if fewer matches exist.
        """
        if self._vocabulary_stale:
            logger.warning(
                "Index built with different tokenizer version. "
                "Run `mnemosyne ingest --full` to rebuild."
            )
            return []

        query_vec = self.embed(query)
        if not query_vec:
            return []

        # Accumulate dot-product contributions via the inverted index.
        scores: dict[int, float] = defaultdict(float)
        for term, q_weight in query_vec.items():
            for chunk_id, d_weight in self._inverted_index.get(term, []):
                scores[chunk_id] += q_weight * d_weight

        # Sort descending by accumulated score and return top_k.
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return ranked

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_vocabulary(self) -> None:
        """
        Attempt to load a persisted vocabulary from the store.

        The store is expected to expose a ``load_vocabulary()`` method that
        returns a dict with keys ``"vocabulary"`` (term->df), ``"idf"``
        (term->weight), and ``"total_docs"`` (int).  If the method is absent
        or returns None, loading is silently skipped.

        After loading, the stored tokenizer hash is compared against the
        current hash.  If they differ, ``_vocabulary_stale`` is set to True
        and a warning is logged.
        """
        if self.store is None:
            return
        load_fn = getattr(self.store, "load_vocabulary", None)
        if load_fn is None:
            return
        payload = load_fn()
        if not payload:
            return
        self.vocabulary = payload.get("vocabulary", {})
        self.idf = payload.get("idf", {})
        self.total_docs = payload.get("total_docs", 0)

        # Check tokenizer version consistency.
        get_fn = getattr(self.store, "get_index_metadata", None)
        if get_fn is not None:
            stored_hash = get_fn("tokenizer_hash")
            if stored_hash is not None:
                current_hash = self._compute_tokenizer_hash()
                if stored_hash != current_hash:
                    self._vocabulary_stale = True
                    logger.warning(
                        "Tokenizer configuration has changed since the index "
                        "was built (stored=%s, current=%s). Search results "
                        "will be empty until the index is rebuilt.",
                        stored_hash[:12],
                        current_hash[:12],
                    )

    def _save_vocabulary(self) -> None:
        """
        Persist the current vocabulary to the store (if available).

        The store is expected to expose a ``save_vocabulary(payload: dict)``
        method.  If absent, the call is silently skipped.
        """
        if self.store is None:
            return
        save_fn = getattr(self.store, "save_vocabulary", None)
        if save_fn is None:
            return
        payload = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "total_docs": self.total_docs,
        }
        save_fn(payload)
