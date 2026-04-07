# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Document retrieval engine for Mnemosyne.

A purpose-built retrieval pipeline for the document partition.  Uses
BM25 (FTS5) and TF-IDF as the two signals, fused via RRF.  No symbol
matching, no import graph, no code boilerplate detection, no extension
penalties.  Clean prose retrieval.

Reuses :func:`~mnemosyne.ranking.rrf_fuse` and
:func:`~mnemosyne.ranking.budget_cut` from the shared ranking module.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from mnemosyne.models import QueryResult, Chunk, estimate_tokens
from mnemosyne.ranking import rrf_fuse, budget_cut

if TYPE_CHECKING:
    from mnemosyne.doc_store import DocStore
    from mnemosyne.config import Config
    from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

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


def _escape_fts5_doc(query: str) -> str:
    """Escape FTS5 special chars for document search."""
    cleaned = _FTS5_SPECIAL.sub(" ", query).strip()
    terms = [t for t in cleaned.split()
             if len(t) >= 2 and t.lower() not in _DOC_STOPWORDS]
    if not terms:
        return cleaned
    return " OR ".join(terms)


class DocRetrievalEngine:
    """Document partition retrieval -- BM25 + TF-IDF, RRF fusion.

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

        # Signal 1: BM25 via FTS5
        bm25_results = self._bm25_search(query_text, max_results * 3)

        # Signal 2: TF-IDF vector search
        vector_results = self._vector_search(query_text, max_results * 3)

        if not bm25_results and not vector_results:
            return []

        # RRF fusion -- two signals, equal weight
        score_lists: dict[str, list[tuple[int, float]]] = {}
        weights: dict[str, float] = {}

        if bm25_results:
            score_lists["bm25"] = bm25_results
            weights["bm25"] = 0.5
        if vector_results:
            score_lists["vector"] = vector_results
            weights["vector"] = 0.5

        fused = rrf_fuse(score_lists, weights)

        # Cost-model re-ranking: simple value-per-token density
        ranked = self._density_rank(fused)

        # Budget cut (reuse shared budget_cut, no compressor for docs)
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
