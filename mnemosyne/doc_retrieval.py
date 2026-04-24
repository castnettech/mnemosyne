# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document retrieval engine for Mnemosyne.

Purpose-built retrieval pipeline for the document partition.  Signals:

  - BM25 via SQLite FTS5
  - TF-IDF sparse cosine
  - Hashed dense cosine (128-dim int8 ``hashed_tfidf_v1`` vectors, the
    same shape the turn lane writes)

Pipeline:

    1. Run all three signals (dense optional, feature-flagged).
    2. Fuse with RRF -- equal weight when dense is on.
    3. Take the top-N fused pool and rerank by dense cosine -- "poor
       man's cross-encoder".  Shipping now > perfect later.
    4. Cost-model density rank + budget cut on the rerank survivors.

Feature flags (environment variables, default ON):

    MNEMOSYNE_DENSE_LANE ("0" disables)
    MNEMOSYNE_RERANK     ("0" disables)

Both default to enabled; set to "0" to roll back without code changes.

Reuses :func:`~mnemosyne.ranking.rrf_fuse` and
:func:`~mnemosyne.ranking.budget_cut` from the shared ranking module.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import TYPE_CHECKING

from mnemosyne.embeddings import hashed_dense
from mnemosyne.models import QueryResult
from mnemosyne.ranking import budget_cut, rrf_fuse

if TYPE_CHECKING:
    from mnemosyne.config import Config
    from mnemosyne.doc_store import DocStore
    from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

logger = logging.getLogger(__name__)

# FTS5 special chars
_FTS5_SPECIAL = re.compile(r'["\'\(\)\*\:\^\.\,\;\-\?\!\[\]\{\}\<\>\~\`\#\@\&\$\%\+\=\/\\]')

# Lighter stopword set for document queries -- keep more natural language
# terms than the code pipeline does, since document queries are prose.
_DOC_STOPWORDS: frozenset[str] = frozenset({
    "the", "be", "to", "of", "and", "in", "that", "have", "it",
    "for", "not", "on", "with", "as", "do", "at", "this", "but",
    "by", "from", "or", "an", "will", "all", "would", "there",
    "their", "so", "up", "out", "if", "about",
    "are", "is", "was", "were", "been", "has", "had", "being",
})

# ---------------------------------------------------------------------------
# Feature-flag helpers
# ---------------------------------------------------------------------------

# Default envelope cap after rerank.  Each chunk is typically 150-500
# tokens, so 8 chunks sits well under the 4k retrieval budget and
# dramatically beats the pre-rerank 26-37 flood.
_DEFAULT_RERANK_KEEP = 8

# Pre-rerank pool size.  RRF returns up to ~ max_results * 3 candidates
# per lane.  We cap the rerank cosine pass at 50 so latency stays
# sub-millisecond even on cold caches.
_DEFAULT_RERANK_POOL = 50


def _env_flag(name: str, default: bool = True) -> bool:
    """Parse an ``MNEMOSYNE_*`` env flag.  ``"0"`` disables, anything else
    falls back to *default*.  Empty / unset -> default.
    """
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val not in ("0", "false", "False", "no", "off")


def dense_lane_enabled() -> bool:
    """True when the dense lane is active for doc retrieval."""
    return _env_flag("MNEMOSYNE_DENSE_LANE", default=True)


def rerank_enabled() -> bool:
    """True when the post-fusion rerank pass is active."""
    return _env_flag("MNEMOSYNE_RERANK", default=True)


def rerank_keep() -> int:
    """Maximum envelope size after rerank.  Env override:
    ``MNEMOSYNE_RERANK_KEEP`` (defaults to 8).
    """
    raw = os.getenv("MNEMOSYNE_RERANK_KEEP")
    if not raw:
        return _DEFAULT_RERANK_KEEP
    try:
        v = int(raw)
        if v <= 0:
            return _DEFAULT_RERANK_KEEP
        return v
    except ValueError:
        return _DEFAULT_RERANK_KEEP


def rerank_pool() -> int:
    """Pre-rerank fused pool cap.  Env override:
    ``MNEMOSYNE_RERANK_POOL`` (defaults to 50).
    """
    raw = os.getenv("MNEMOSYNE_RERANK_POOL")
    if not raw:
        return _DEFAULT_RERANK_POOL
    try:
        v = int(raw)
        if v <= 0:
            return _DEFAULT_RERANK_POOL
        return v
    except ValueError:
        return _DEFAULT_RERANK_POOL


def _escape_fts5_doc(query: str) -> str:
    """Escape FTS5 special chars for document search."""
    cleaned = _FTS5_SPECIAL.sub(" ", query).strip()
    terms = [t for t in cleaned.split()
             if len(t) >= 2 and t.lower() not in _DOC_STOPWORDS]
    if not terms:
        return cleaned
    return " OR ".join(terms)


class DocRetrievalEngine:
    """Document partition retrieval -- BM25 + TF-IDF + dense, RRF + rerank.

    Args:
        doc_store:     :class:`~mnemosyne.doc_store.DocStore` instance.
        tfidf_backend: TF-IDF backend trained on document vocabulary.
        config:        Mnemosyne config.
    """

    def __init__(
        self,
        doc_store: "DocStore",
        tfidf_backend: "TFIDFBackend",
        config: "Config",
    ) -> None:
        self.store = doc_store
        self.tfidf = tfidf_backend
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        budget: int | None = None,
        session_id: str | None = None,
    ) -> list[QueryResult]:
        """Execute document retrieval and return budget-fitted results.

        Args:
            query_text: Natural language query.
            budget:     Max tokens to return (default from config).
            session_id: Unused (reserved for future usage tracking).

        Returns:
            Ordered list of :class:`~mnemosyne.models.QueryResult`.
        """
        budget = budget if budget is not None else self.config.retrieval.token_budget
        max_results = self.config.retrieval.max_results

        # ---------- Signal 1: BM25 via FTS5
        bm25_results = self._bm25_search(query_text, max_results * 3)

        # ---------- Signal 2: TF-IDF vector search
        vector_results = self._vector_search(query_text, max_results * 3)

        # ---------- Signal 3: Dense hashed-TFIDF cosine (optional)
        dense_enabled = dense_lane_enabled()
        dense_results: list[tuple[int, float]] = []
        if dense_enabled:
            # Pre-filter the scan to BM25 + TF-IDF candidates to keep
            # latency bounded when the index is large.  If both lanes
            # returned nothing, fall back to a full scan so a pure-
            # semantic hit can still surface.
            candidate_ids: list[int] = []
            seen: set[int] = set()
            for cid, _ in bm25_results:
                if cid not in seen:
                    seen.add(cid)
                    candidate_ids.append(cid)
            for cid, _ in vector_results:
                if cid not in seen:
                    seen.add(cid)
                    candidate_ids.append(cid)
            dense_results = self._dense_search(
                query_text,
                candidate_ids=candidate_ids or None,
                limit=max_results * 3,
            )

        if not bm25_results and not vector_results and not dense_results:
            return []

        # ---------- Fusion (RRF, equal weights across active lanes)
        score_lists: dict[str, list[tuple[int, float]]] = {}
        weights: dict[str, float] = {}

        if bm25_results:
            score_lists["bm25"] = bm25_results
            weights["bm25"] = 1.0
        if vector_results:
            score_lists["vector"] = vector_results
            weights["vector"] = 1.0
        if dense_results:
            score_lists["dense"] = dense_results
            weights["dense"] = 1.0

        if weights:
            # Normalise equal weights so RRF contributions stay comparable
            # to the pre-wave behaviour when only two lanes were active.
            w = 1.0 / len(weights)
            for k in weights:
                weights[k] = w

        fused = rrf_fuse(score_lists, weights)

        # ---------- Rerank (lightweight cosine rerank on fused pool)
        reranked = fused
        if rerank_enabled():
            reranked = self._rerank_pool(
                fused,
                query_text,
                pool=rerank_pool(),
                keep=rerank_keep(),
            )

        # ---------- Cost-model density re-ranking
        ranked = self._density_rank(reranked)

        # ---------- Budget cut (reuse shared budget_cut, no compressor for docs)
        selected = budget_cut(ranked, budget, compressor=None, store=self.store)

        # Build QueryResult objects
        results: list[QueryResult] = []
        for chunk, source_scores in selected:
            file_rec = self._get_file_record(chunk.file_id)
            rel_path = file_rec.rel_path if file_rec else ""
            results.append(QueryResult(
                chunk=chunk,
                file_path=rel_path,
                scores=source_scores,
            ))

        return results

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _bm25_search(
        self, query_text: str, limit: int,
    ) -> list[tuple[int, float]]:
        """BM25 full-text search on doc_chunks_fts."""
        safe_query = _escape_fts5_doc(query_text)
        if not safe_query.strip():
            return []
        raw = self.store.search_fts(safe_query, limit=limit)
        if not raw:
            return []
        # Normalize scores to [0, 1]
        max_abs = max(abs(s) for _, s in raw) or 1.0
        return [(cid, abs(score) / max_abs) for cid, score in raw]

    def _vector_search(
        self, query_text: str, limit: int,
    ) -> list[tuple[int, float]]:
        """TF-IDF cosine similarity on doc partition embeddings."""
        results = self.tfidf.search(query_text, top_k=limit)
        return [(cid, score) for cid, score in results if score > 0]

    def _dense_search(
        self,
        query_text: str,
        candidate_ids: list[int] | None,
        limit: int,
    ) -> list[tuple[int, float]]:
        """Hashed-TFIDF dense cosine lane.

        Uses the same 128-dim ``hashed_tfidf_v1`` vectors that the turn
        lane writes on every capture.  When *candidate_ids* is provided
        we only score the pre-filter; otherwise we full-scan the stored
        vectors (bounded by the table size).
        """
        q_vec = hashed_dense.embed_floats(query_text)
        if not any(abs(x) > 0 for x in q_vec):
            return []

        vectors = self.store.get_dense_embeddings_batch(
            chunk_ids=candidate_ids or None,
        )
        if not vectors:
            return []

        scored: list[tuple[int, float]] = []
        for cid, (vec_bytes, dim) in vectors.items():
            if dim <= 0:
                continue
            doc_vec = hashed_dense.decode_int8(vec_bytes)
            sim = hashed_dense.cosine(q_vec, doc_vec)
            if sim > 0:
                scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def _rerank_pool(
        self,
        fused: list[tuple[int, float, dict]],
        query_text: str,
        pool: int,
        keep: int,
    ) -> list[tuple[int, float, dict]]:
        """Rerank the top-*pool* fused candidates by dense cosine.

        Lightweight substitute for a cross-encoder: we score each
        candidate's stored ``hashed_tfidf_v1`` vector against the query
        vector.  Candidates without a stored vector keep their original
        RRF rank by falling back to the fused score so nothing drops
        silently.  The survivors are capped at *keep*.
        """
        if not fused:
            return fused
        head = fused[:pool]
        tail = fused[pool:]

        q_vec = hashed_dense.embed_floats(query_text)
        q_nonzero = any(abs(x) > 0 for x in q_vec)

        head_ids = [cid for cid, _, _ in head]
        vectors = self.store.get_dense_embeddings_batch(chunk_ids=head_ids)

        # (blended, rrf_score, original_index, chunk_id, rrf_for_output,
        #  scores_dict) -- original_index keeps sort stable for ties.
        scored: list[tuple[float, float, int, int, float, dict]] = []
        for idx, (cid, rrf_score, scores_dict) in enumerate(head):
            sim = 0.0
            if q_nonzero and cid in vectors:
                vec_bytes, _ = vectors[cid]
                doc_vec = hashed_dense.decode_int8(vec_bytes)
                sim = hashed_dense.cosine(q_vec, doc_vec)
            # Blend rerank cosine with RRF so chunks with a missing dense
            # vector still get ranked relative to the rest of the pool.
            # 0.7 * cosine + 0.3 * rrf tracks "lambda=0.7 relevance-
            # skewed MMR" per brief's option (b).
            blended = 0.7 * sim + 0.3 * rrf_score
            new_scores = dict(scores_dict)
            new_scores["rerank_cosine"] = round(sim, 6)
            new_scores["rerank_blended"] = round(blended, 6)
            scored.append((blended, rrf_score, -idx, cid, rrf_score, new_scores))

        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        kept = [(cid, rrf, sc) for _, _, _, cid, rrf, sc in scored[:keep]]

        # Preserve the non-fit tail so budget_cut can still reach for
        # them if a must-cite candidate happened to land at rank 51+.
        # They are appended after the rerank winners in original order.
        return kept + tail

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _density_rank(
        self, fused: list[tuple[int, float, dict]],
    ) -> list[tuple[int, float, dict]]:
        """Re-rank by value density: relevance / log(token_count).

        No code-specific penalties.  No boilerplate detection.
        Documents are ranked purely on relevance per token.
        """
        enriched: list[tuple[float, int, float, dict]] = []
        for chunk_id, rrf_score, source_scores in fused:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue
            tokens = max(1, chunk.token_count)
            density = rrf_score / (1.0 + math.log1p(tokens))
            enriched.append((density, chunk_id, rrf_score, source_scores))

        enriched.sort(key=lambda x: x[0], reverse=True)
        return [(cid, rrf, scores) for _, cid, rrf, scores in enriched]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_file_record(self, file_id: int):
        """Look up file record from the shared files table."""
        row = self.store.conn.execute(
            "SELECT * FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        if row is None:
            return None
        from mnemosyne.models import FileRecord
        keys = row.keys()
        return FileRecord(
            file_id=row["file_id"],
            rel_path=row["rel_path"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            language=row["language"],
            last_modified=row["last_modified"],
            last_indexed=row["last_indexed"],
            is_deleted=bool(row["is_deleted"]),
            source_type=row["source_type"] if "source_type" in keys else "file",
            extraction_method=row["extraction_method"] if "extraction_method" in keys else None,
            extraction_quality=row["extraction_quality"] if "extraction_quality" in keys else None,
            page_count=row["page_count"] if "page_count" in keys else None,
        )
