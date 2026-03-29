# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hybrid retrieval orchestrator for Mnemosyne.

Combines BM25 full-text search (SQLite FTS5), TF-IDF vector search, usage
frequency scores, and pre-fetch boosting via Reciprocal Rank Fusion (RRF).
Results are re-ranked by cost-model value density and cut to a token budget.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from mnemosyne.models import QueryResult, estimate_tokens
from mnemosyne.ranking import rrf_fuse, cost_model_score, budget_cut

if TYPE_CHECKING:
    from mnemosyne.store import Store
    from mnemosyne.analytics import Analytics
    from mnemosyne.prefetch import Prefetcher


# FTS5 special characters that must be escaped in query strings
_FTS5_SPECIAL = re.compile(r'["\'\(\)\*\:\^]')

# Common English stopwords that inflate BM25 scores for prose-heavy files
# (HTML, Markdown) without contributing retrieval signal for code search.
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


def _escape_fts5(query: str) -> str:
    """Escape characters that have special meaning in FTS5 MATCH queries.

    Joins terms with OR for broader recall (implicit AND is too strict
    for natural language queries against code).  Stopwords are removed
    so that common English words do not inflate scores for prose-heavy
    documents (HTML, Markdown) that share terms with the query.
    """
    cleaned = _FTS5_SPECIAL.sub(" ", query).strip()
    terms = [t for t in cleaned.split()
             if len(t) >= 2 and t.lower() not in _STOPWORDS]
    if not terms:
        return cleaned
    return " OR ".join(terms)


class RetrievalEngine:
    """
    Hybrid search orchestrator.

    Pipeline on each :meth:`query` call:

    1. BM25 search via SQLite FTS5.
    2. TF-IDF vector search.
    3. Usage frequency scores (decay-weighted).
    4. Pre-fetch boosting (pattern-matched chunk IDs promoted to rank 1).
    5. RRF fusion across all signals.
    6. Cost-model re-ranking (value density = rrf_score / token_count).
    7. Budget cutting with optional compression fallback.
    8. Usage-event recording.

    Args:
        store:       Persistent :class:`~mnemosyne.store.Store` instance.
        tfidf_backend: TF-IDF backend with a ``search(query, top_k) ->
                       list[(chunk_id, score)]`` method.
        config:      Mnemosyne :class:`~mnemosyne.config.Config` instance.
        analytics:   Optional :class:`~mnemosyne.analytics.Analytics` instance.
        prefetcher:  Optional :class:`~mnemosyne.prefetch.Prefetcher` instance.
    """

    def __init__(
        self,
        store: "Store",
        tfidf_backend,
        config,
        analytics: "Analytics | None" = None,
        prefetcher: "Prefetcher | None" = None,
    ) -> None:
        self.store = store
        self.tfidf = tfidf_backend
        self.config = config
        self.analytics = analytics
        self.prefetcher = prefetcher

        # Ensure the TF-IDF inverted index is populated from persisted
        # embeddings so vector search works immediately.
        self._ensure_inverted_index()

    def _ensure_inverted_index(self) -> None:
        """Load sparse embeddings into the TF-IDF inverted index if empty."""
        if getattr(self.tfidf, "_inverted_index", None):
            return
        build_fn = getattr(self.tfidf, "build_inverted_index", None)
        if build_fn is None:
            return
        try:
            all_embs = self.store.get_all_sparse_embeddings()
            if all_embs:
                build_fn(all_embs)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        budget: int | None = None,
        session_id: str | None = None,
        use_compression: bool = True,
    ) -> list[QueryResult]:
        """
        Execute hybrid retrieval and return budget-fitted results.

        Args:
            query_text:      The user or agent query string.
            budget:          Maximum total tokens to return.  Defaults to
                             ``config.retrieval.token_budget``.
            session_id:      Session identifier used for usage event logging.
            use_compression: Try compressed chunk content when a chunk is
                             too large for the remaining budget.

        Returns:
            Ordered list of :class:`~mnemosyne.models.QueryResult`, highest
            relevance first.
        """
        budget = budget if budget is not None else self.config.retrieval.token_budget

        # 1. BM25 search via FTS5
        bm25_results = self._bm25_search(query_text)

        # 2. Vector search via TF-IDF (keyword-based)
        vector_results = self._vector_search(query_text)

        # 2b. Symbol name search — match query terms against chunk symbol_name
        symbol_results = self._symbol_search(query_text)

        # 3. Usage frequency scores
        usage_scores = self._usage_scores() if self.analytics else {}

        # 4. Pre-fetch boosting
        prefetch_ids = self._prefetch_check(query_text) if self.prefetcher else set()

        # 5. RRF fusion — all signals combined
        # Strip chunk_type from symbol triples for RRF (expects 2-tuples)
        symbol_pairs = [(cid, score) for cid, score, _ in symbol_results] if symbol_results else []
        fused = self._rrf_fuse(
            bm25_results, vector_results, usage_scores, prefetch_ids,
            symbol_results=symbol_pairs,
        )

        # 5a. Symbol match multiplier — purely additive over v0.3.0.
        #     All symbol matches get the same 3x boost as before.
        #     PascalCase class-name mode adds a 4x boost for class-type
        #     chunks only — this is the ONLY new behaviour.
        if symbol_results:
            symbol_info: dict[int, tuple[float, str]] = {}
            for cid, score, ctype in symbol_results:
                if score >= 0.5:
                    symbol_info[cid] = (score, ctype)

            # Detect class-name query terms -> class-name mode.
            # Triggers on PascalCase (e.g., "AsyncClient") or TitleCase
            # (e.g., "Timeout", "Response") — both are class conventions.
            query_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query_text)
            class_name_mode = any(
                t[0].isupper() and re.search(r"[a-z]", t) and (
                    re.search(r"[A-Z]", t[1:])  # PascalCase: AsyncClient
                    or t[0].isupper() and t[1:].islower() and len(t) >= 4  # TitleCase: Timeout
                )
                for t in query_tokens
            )

            if symbol_info:
                boosted = []
                for cid, rrf, scores in fused:
                    if cid in symbol_info:
                        _, ctype = symbol_info[cid]
                        # 3x for all (preserves v0.3.0); 4x for class + PascalCase
                        boost = 4.0 if (class_name_mode and ctype == "class") else 3.0
                        new_rrf = rrf * boost
                        new_scores = dict(scores)
                        new_scores["rrf"] = new_rrf
                        boosted.append((cid, new_rrf, new_scores))
                    else:
                        boosted.append((cid, rrf, scores))
                fused = sorted(boosted, key=lambda x: x[1], reverse=True)

        # 5b. Filename-match boost: if query terms appear in a file's name
        #     or path, boost all chunks from that file.
        fused = self._filename_boost(fused, query_text)

        # 5c. File-level filter: keep only chunks from the top-scoring files.
        fused = self._file_level_filter(fused)

        # 5d. Import/namespace graph: inject up to 2 additional files that
        #     are dependencies of the surviving files.  Runs AFTER the filter
        #     so core keyword-matched results are never displaced.
        fused = self._import_graph_boost(fused)

        # 6. Cost-model re-ranking (value per token)
        ranked = self._cost_model_rank(fused)

        # 7. Budget cutting with optional compression
        compressor = None
        if use_compression:
            try:
                from mnemosyne.compress import Compressor
                compressor = Compressor(self.config)
            except ImportError:
                pass

        results = self._budget_cut(ranked, budget, compressor)

        # 7b. Staleness detection — annotate results whose source file
        #     has changed on disk since the last index run.
        self._annotate_staleness(results)

        # 8. Record usage events
        if self.analytics and session_id:
            self._record_events(results, query_text, session_id)

        # 9. Record pre-fetch pattern
        if self.prefetcher and results:
            selected_ids = [r.chunk.chunk_id for r in results if r.chunk.chunk_id is not None]
            self.prefetcher.record_pattern(query_text, selected_ids)

        return results

    # ------------------------------------------------------------------
    # Search signals
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str) -> list[tuple[int, float]]:
        """
        Run a BM25 query via SQLite FTS5.

        FTS5 special characters are escaped before the query is issued.
        Results are normalised to [0, 1] relative to the top score.

        Returns:
            List of ``(chunk_id, normalised_bm25_score)`` pairs, at most
            ``max_results * 3`` entries.
        """
        safe_query = _escape_fts5(query)
        if not safe_query:
            return []

        limit = self.config.retrieval.max_results * 3
        try:
            raw_results: list[tuple[int, float]] = self.store.search_fts(safe_query, limit=limit)
        except Exception:
            return []

        if not raw_results:
            return []

        # FTS5 bm25() returns negative values (lower = better).
        # Convert to positive scores and normalise.
        # FTS5 scores are typically negative; abs gives us a positive magnitude.
        scores = [(cid, abs(score)) for cid, score in raw_results]
        max_score = max(s for _, s in scores) if scores else 1.0
        if max_score == 0.0:
            max_score = 1.0
        return [(cid, score / max_score) for cid, score in scores]

    def _vector_search(self, query: str) -> list[tuple[int, float]]:
        """
        Run TF-IDF vector search.

        Returns:
            List of ``(chunk_id, similarity_score)`` pairs.
        """
        try:
            return self.tfidf.search(query, top_k=self.config.retrieval.max_results * 3)
        except Exception:
            return []

    def _symbol_search(self, query: str) -> list[tuple[int, float, str]]:
        """
        Search for chunks whose ``symbol_name`` matches query terms.

        Extracts camelCase/snake_case identifiers from the query and checks
        them against the ``symbol_name`` column.  Matching chunks get a
        normalised score of 1.0 (exact match) or 0.5 (partial/prefix match).

        Returns:
            List of ``(chunk_id, score, chunk_type)`` triples.
        """
        # Extract potential identifiers from the query
        identifiers = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query)
        if not identifiers:
            return []

        # Normalise: collect lowercase versions and camelCase parts
        search_terms: set[str] = set()
        for ident in identifiers:
            lower = ident.lower()
            if lower not in _STOPWORDS:
                search_terms.add(lower)
            # Also split camelCase
            parts = re.sub(r"([a-z])([A-Z])", r"\1_\2", ident).lower().split("_")
            for p in parts:
                if len(p) >= 3 and p not in _STOPWORDS:
                    search_terms.add(p)

        if not search_terms:
            return []

        # Query the database for chunks with matching symbol names
        results: list[tuple[int, float, str]] = []
        try:
            rows = self.store.conn.execute(
                "SELECT chunk_id, symbol_name, chunk_type FROM chunks WHERE symbol_name IS NOT NULL"
            ).fetchall()
        except Exception:
            return []

        for row in rows:
            symbol = row["symbol_name"].lower()
            # Split the symbol name into components too
            symbol_parts = re.sub(r"([a-z])([A-Z])", r"\1_\2", row["symbol_name"]).lower().split("_")

            # Check for exact match
            if symbol in search_terms:
                results.append((int(row["chunk_id"]), 1.0, row["chunk_type"] or ""))
                continue

            # Check for prefix/component overlap
            overlap = search_terms & set(symbol_parts)
            if overlap:
                # Score by fraction of matching components
                score = len(overlap) / max(len(symbol_parts), len(search_terms)) * 0.8
                results.append((int(row["chunk_id"]), score, row["chunk_type"] or ""))

        return results

    def _usage_scores(self) -> dict[int, float]:
        """
        Retrieve exponentially-decayed usage scores from Analytics.

        Returns:
            Mapping of ``chunk_id -> decay_score``.
        """
        try:
            return self.analytics.get_usage_scores()
        except Exception:
            return {}

    def _prefetch_check(self, query: str) -> set[int]:
        """
        Look up the pre-fetch pattern for this query.

        Returns:
            Set of chunk IDs that should be boosted.
        """
        try:
            return self.prefetcher.get_prefetch_ids(query)
        except Exception:
            return set()

    # ------------------------------------------------------------------
    # Fusion and ranking
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        bm25: list[tuple[int, float]],
        vector: list[tuple[int, float]],
        usage: dict[int, float],
        prefetch_ids: set[int],
        symbol_results: list[tuple[int, float]] | None = None,
    ) -> list[tuple[int, float, dict]]:
        """
        Fuse all search signals via Reciprocal Rank Fusion.

        Signals: BM25, TF-IDF vector, dense semantic (hybrid mode),
        symbol name matches, usage scores, pre-fetch boost.
        """
        cfg = self.config.retrieval
        weights = {
            "bm25": cfg.bm25_weight,
            "vector": cfg.vector_weight,
            "usage": cfg.usage_weight,
        }

        # Convert usage dict to a ranked list
        usage_list: list[tuple[int, float]] = sorted(
            usage.items(), key=lambda x: x[1], reverse=True
        )

        score_lists: dict[str, list[tuple[int, float]]] = {
            "bm25": bm25,
            "vector": vector,
            "usage": usage_list,
        }

        # Symbol name matches — highest-confidence signal
        if symbol_results:
            score_lists["symbol"] = symbol_results
            weights["symbol"] = cfg.bm25_weight * 1.5

        # Inject pre-fetch boost: add prefetch_ids to every list at top with
        # a synthetic high score so they rank first in the fused result.
        if prefetch_ids:
            boost_score = 2.0  # above any normalised score
            prefetch_list = [(cid, boost_score) for cid in prefetch_ids]
            score_lists["prefetch"] = prefetch_list
            weights["prefetch"] = max(cfg.bm25_weight, cfg.vector_weight)

        return rrf_fuse(score_lists, weights, k=60)

    def _filename_boost(
        self,
        fused: list[tuple[int, float, dict]],
        query_text: str,
    ) -> list[tuple[int, float, dict]]:
        """
        Boost chunks from files whose name/path matches query terms.

        When a query contains a meaningful keyword that appears in a file's
        basename — e.g. "score" matches ``scorer.js`` — all chunks from that
        file receive a 1.5x RRF score multiplier.  Stopwords and short terms
        are filtered to prevent false positives.

        Uses stem-prefix matching: a match occurs when a query term and a
        filename component share a common prefix of >= 5 characters AND the
        prefix covers at least 60% of the shorter string.  Files not already
        in the fused results (i.e. missed by BM25 and TF-IDF) are never
        injected — only existing results are boosted.
        """
        # Extract meaningful query terms: min 5 chars, not stopwords
        query_lower = query_text.lower()
        raw_terms = set(re.findall(r"[a-z]{5,}", query_lower))
        terms = raw_terms - _STOPWORDS
        if not terms:
            return fused

        # Collect file_ids already present in fused results
        fused_file_ids: set[int] = set()
        for chunk_id, _, _ in fused:
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                fused_file_ids.add(chunk.file_id)

        # Build a set of boosted file_ids using stem-prefix matching.
        # Only consider files that are ALREADY in the fused results — a
        # filename substring match alone is insufficient evidence to inject
        # files that neither BM25 nor TF-IDF retrieved.
        boosted_file_ids: set[int] = set()

        for rec in self.store.list_files(include_deleted=False):
            if rec.file_id not in fused_file_ids:
                continue

            # Split filename into meaningful components
            basename = rec.rel_path.rsplit("/", 1)[-1].lower()
            # Remove extension and split on separators (-, _, .)
            name_no_ext = basename.rsplit(".", 1)[0] if "." in basename else basename
            components = re.split(r"[-_.]", name_no_ext)

            for term in terms:
                for comp in components:
                    # Both term and component must be >= 5 chars
                    if len(comp) < 5:
                        continue
                    prefix_len = 5
                    if comp[:prefix_len] == term[:prefix_len] or term[:prefix_len] == comp[:prefix_len]:
                        # Require the matched prefix to cover >= 60% of
                        # the shorter string
                        shorter_len = min(len(term), len(comp))
                        if prefix_len / shorter_len >= 0.6:
                            boosted_file_ids.add(rec.file_id)
                            break
                else:
                    continue
                break

        if not boosted_file_ids:
            return fused

        result: list[tuple[int, float, dict]] = []

        for chunk_id, rrf_score, source_scores in fused:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue

            if chunk.file_id in boosted_file_ids:
                boosted_score = rrf_score * 1.5
                boosted_sources = dict(source_scores)
                boosted_sources["rrf"] = boosted_score
                result.append((chunk_id, boosted_score, boosted_sources))
            else:
                result.append((chunk_id, rrf_score, source_scores))

        # Re-sort by boosted RRF score
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _import_graph_boost(
        self,
        fused: list[tuple[int, float, dict]],
    ) -> list[tuple[int, float, dict]]:
        """
        Inject files connected to already-retrieved files via dependency edges.

        Runs AFTER the file-level filter so keyword-matched results are never
        displaced.  Uses available file slots (max_files - current count).

        Scans three reference types (general, not codebase-specific):

        1. **ES6 imports / CommonJS require** — static module references
        2. **Runtime namespace access** — ``Namespace.Module`` patterns where
           the property maps to an indexed file's basename
        3. **Path references** — quoted strings matching indexed file paths
           (catches shell scripts referencing HTML, config referencing scripts)

        Source files are prioritised over test files in the injection queue.
        """
        import os

        all_files = {f.file_id: f.rel_path for f in self.store.list_files(include_deleted=False)}
        path_to_id = {v: k for k, v in all_files.items()}

        # Basename → file_id for namespace resolution
        basename_to_id: dict[str, int] = {}
        for fid, path in all_files.items():
            name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            basename_to_id[name] = fid

        # Filename fragments for path-reference detection
        filename_to_id: dict[str, int] = {}
        for fid, path in all_files.items():
            fname = path.rsplit("/", 1)[-1]
            filename_to_id[fname] = fid

        _import_re = re.compile(
            r"""(?:import\s+.*?from\s+['"]([^'"]+)['"]"""
            r"""|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
        )
        _namespace_re = re.compile(
            r"""(?:var|let|const|=)\s+\w+\s*=\s*(\w+)\.([A-Z]\w+)\b"""
            r"""|(\w+)\.([A-Z]\w+)\.\w+\("""
        )
        # Quoted strings that look like file paths
        _path_ref_re = re.compile(r"""['"]([^'"]*?/?\w+\.\w{1,5})['"]""")

        # Identify files currently in fused results
        fused_file_ids: set[int] = set()
        for cid, _, _ in fused:
            chunk = self.store.get_chunk(cid)
            if chunk:
                fused_file_ids.add(chunk.file_id)

        # Collect forward dependencies from fused files
        connected_file_ids: set[int] = set()
        for fid in fused_file_ids:
            for chunk in self.store.get_chunks_for_file(fid):
                # Static imports
                for m in _import_re.finditer(chunk.content):
                    raw = m.group(1) or m.group(2)
                    if not raw or raw.startswith("http"):
                        continue
                    source_dir = os.path.dirname(all_files.get(fid, ""))
                    candidate = os.path.normpath(os.path.join(source_dir, raw)).replace("\\", "/")
                    for suffix in ("", ".js", ".ts", "/index.js"):
                        full = candidate + suffix
                        if full in path_to_id:
                            connected_file_ids.add(path_to_id[full])
                            break

                # Runtime namespace reads
                for m in _namespace_re.finditer(chunk.content):
                    prop = (m.group(2) or m.group(4) or "").lower()
                    if prop and prop in basename_to_id:
                        target_fid = basename_to_id[prop]
                        if target_fid != fid:
                            connected_file_ids.add(target_fid)

                # Path references (catches serve.sh → index.html, etc.)
                for m in _path_ref_re.finditer(chunk.content):
                    ref = m.group(1)
                    fname = ref.rsplit("/", 1)[-1]
                    if fname in filename_to_id:
                        target_fid = filename_to_id[fname]
                        if target_fid != fid:
                            connected_file_ids.add(target_fid)

        connected_file_ids -= fused_file_ids
        if not connected_file_ids:
            return fused

        # Read configurable injection threshold (minimum references to inject)
        import_inject_threshold = getattr(
            self.config.retrieval, "import_inject_threshold", 2
        )

        # Dynamic cap: fill available slots up to 2
        max_inject = min(2, max(0, 10 - len(fused_file_ids)))
        if max_inject == 0:
            return fused

        # Prioritise by: (1) source over test, (2) most-referenced first.
        # A file referenced by 3 fused files is more likely relevant than
        # one referenced by only 1.
        ref_counts: dict[int, int] = {}
        for fid_c in connected_file_ids:
            ref_counts[fid_c] = ref_counts.get(fid_c, 0)
        # Recount from the scanning loop above — track per-file hit count
        for fid_fused in fused_file_ids:
            for chunk in self.store.get_chunks_for_file(fid_fused):
                for m in _namespace_re.finditer(chunk.content):
                    prop = (m.group(2) or m.group(4) or "").lower()
                    if prop and prop in basename_to_id:
                        t = basename_to_id[prop]
                        if t in connected_file_ids:
                            ref_counts[t] = ref_counts.get(t, 0) + 1
                for m in _import_re.finditer(chunk.content):
                    raw = m.group(1) or m.group(2)
                    if not raw or raw.startswith("http"):
                        continue
                    source_dir = os.path.dirname(all_files.get(fid_fused, ""))
                    candidate = os.path.normpath(os.path.join(source_dir, raw)).replace("\\", "/")
                    for suffix in ("", ".js", ".ts", "/index.js"):
                        full = candidate + suffix
                        if full in path_to_id and path_to_id[full] in connected_file_ids:
                            ref_counts[path_to_id[full]] = ref_counts.get(path_to_id[full], 0) + 1
                            break

        # Filter: only inject files referenced at least import_inject_threshold
        # times.  A single reference is weakly correlated with the query.
        eligible = {
            fid for fid in connected_file_ids
            if ref_counts.get(fid, 0) >= import_inject_threshold
        }
        if not eligible:
            return fused

        def _sort_key(fid: int) -> tuple[int, int]:
            path = all_files.get(fid, "")
            is_test = 1 if ("test" in path.lower()) else 0
            return (is_test, -ref_counts.get(fid, 0))

        sorted_connected = sorted(eligible, key=_sort_key)

        result = list(fused)
        sorted_scores = sorted((s for _, s, _ in result), reverse=True)
        top_score = sorted_scores[0] if sorted_scores else 1.0
        p75_score = sorted_scores[max(0, len(sorted_scores) // 4)] if sorted_scores else 1.0

        injected = 0
        for fid in sorted_connected:
            if injected >= max_inject:
                break
            # High-confidence injections (referenced 3+ times) get near-top
            # score so they survive budget_cut.  Others get p75 score at a
            # lower multiplier so budget_cut can prune them if not truly
            # relevant.
            rc = ref_counts.get(fid, 0)
            score = top_score * 0.9 if rc >= 3 else p75_score * 0.7
            for chunk in self.store.get_chunks_for_file(fid):
                result.append((chunk.chunk_id, score, {"dep_graph": 1.0, "rrf": score}))
            injected += 1

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _file_level_filter(
        self,
        fused: list[tuple[int, float, dict]],
    ) -> list[tuple[int, float, dict]]:
        """
        Keep only chunks from the top *max_files* files by aggregate RRF score.

        ``max_files`` is determined adaptively based on the number of unique
        files in the fused results: ``min(max(3, n_files // 3), 8)``.  This
        scales from 3 (small projects) to 8 (large corpora).  If
        ``config.retrieval.max_files`` is set to a positive value, that
        override is used instead.

        Returns:
            Filtered list of ``(chunk_id, rrf_score, source_scores)``.
        """
        from collections import defaultdict

        # Accumulate total RRF score per file_id
        file_scores: dict[int, float] = defaultdict(float)
        chunk_file_map: dict[int, int] = {}

        for chunk_id, rrf_score, _ in fused:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue
            file_scores[chunk.file_id] += rrf_score
            chunk_file_map[chunk_id] = chunk.file_id

        # Determine max_files: config override or adaptive formula
        configured = getattr(self.config.retrieval, "max_files", 0)
        if configured and configured > 0:
            max_files = configured
        else:
            n_unique = len(file_scores)
            max_files = min(max(3, n_unique // 3), 8)

        # Select the top-scoring files
        top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        top_file_ids = {fid for fid, _ in top_files[:max_files]}

        return [
            (cid, rrf, scores) for cid, rrf, scores in fused
            if chunk_file_map.get(cid) in top_file_ids
        ]

    def _cost_model_rank(
        self,
        fused: list[tuple[int, float, dict]],
    ) -> list[tuple[int, float, dict]]:
        """
        Re-rank fused results by value density with boilerplate penalty.

        Chunks with a higher relevance-per-token come first.  Chunks whose
        content is mostly boilerplate (detected by density.py) receive an
        automatic penalty so prose-heavy HTML/legal text is demoted without
        project-specific ignore patterns.

        Returns:
            Re-sorted list of ``(chunk_id, rrf_score, source_scores)``.
        """
        from mnemosyne.density import detect_boilerplate_patterns, repetition_score

        enriched: list[tuple[float, int, float, dict]] = []
        for chunk_id, rrf_score, source_scores in fused:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue

            # Compute boilerplate ratio for this chunk
            lines = chunk.content.splitlines()
            bp_ratio = 0.0
            if lines:
                bp_runs = detect_boilerplate_patterns(lines)
                bp_line_count = sum(end - start + 1 for start, end, _ in bp_runs)
                rep = repetition_score(lines)
                bp_ratio = min(1.0, (bp_line_count / len(lines)) + rep * 0.3)

            # Structured code = named symbol or function/class chunk type
            is_code = (
                chunk.symbol_name is not None
                or chunk.chunk_type in ("function", "class", "imports")
            )

            # Look up file metadata for path-based signals
            file_rec = self.store.get_file_record(chunk.file_id)
            rel_path = file_rec.rel_path if file_rec else ""

            # Penalize non-code file types (HTML, CSS, Markdown, etc.)
            if not is_code and file_rec:
                ext = rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
                if ext in ("html", "css", "md", "txt"):
                    bp_ratio = max(bp_ratio, 0.85)

            # Penalize test files — they exercise production code and have
            # broad vocabulary overlap, but are secondary to source files
            # for most queries.  General signal, not project-specific.
            if rel_path.startswith("tests/") or rel_path.startswith("test/"):
                bp_ratio = max(bp_ratio, 0.5)

            density = cost_model_score(
                chunk_id, rrf_score, chunk.token_count,
                boilerplate_ratio=bp_ratio,
                is_structured_code=is_code,
            )
            enriched.append((density, chunk_id, rrf_score, source_scores))

        enriched.sort(key=lambda x: x[0], reverse=True)
        return [(cid, rrf, scores) for _, cid, rrf, scores in enriched]

    def _budget_cut(
        self,
        ranked: list[tuple[int, float, dict]],
        budget: int,
        compressor,
    ) -> list[QueryResult]:
        """
        Select chunks greedily within *budget* tokens.

        Builds ``QueryResult`` objects including file path, scores, and
        delta/compression markers.

        Returns:
            List of :class:`~mnemosyne.models.QueryResult` in value-density
            order.
        """
        selected_pairs = budget_cut(ranked, budget, compressor=compressor, store=self.store)
        results: list[QueryResult] = []

        for chunk, source_scores in selected_pairs:
            file_record = self.store.get_file_record(chunk.file_id)
            file_path = file_record.rel_path if file_record else f"file:{chunk.file_id}"
            results.append(
                QueryResult(
                    chunk=chunk,
                    file_path=file_path,
                    scores=source_scores,
                    is_delta=False,
                    delta_text=None,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Usage event recording
    # ------------------------------------------------------------------

    def _annotate_staleness(self, results: list[QueryResult]) -> None:
        """
        Check each result's source file against disk and set staleness flags.

        For each unique ``file_id`` in *results*, the stored ``last_modified``
        timestamp is compared to the current ``os.path.getmtime()`` of the
        file on disk.  Results are annotated in-place:

        - ``is_stale=True, stale_reason="file modified since last index"``
          when the disk mtime differs from the stored value.
        - ``is_stale=True, stale_reason="file no longer exists on disk"``
          when the file has been removed from disk.

        Each file is stat'd at most once (cached per file_id).
        """
        if not results:
            return

        project_root = self.config._root

        # Cache: file_id -> (is_stale, stale_reason)
        staleness_cache: dict[int, tuple[bool, str | None]] = {}

        # Collect unique file_ids first to minimise lookups
        file_ids_needed: set[int] = set()
        for r in results:
            file_ids_needed.add(r.chunk.file_id)

        for file_id in file_ids_needed:
            file_record = self.store.get_file_record(file_id)
            if file_record is None:
                staleness_cache[file_id] = (True, "file record not found in index")
                continue

            abs_path = os.path.join(project_root, file_record.rel_path)
            try:
                disk_mtime = os.path.getmtime(abs_path)
                if disk_mtime != file_record.last_modified:
                    staleness_cache[file_id] = (True, "file modified since last index")
                else:
                    staleness_cache[file_id] = (False, None)
            except OSError:
                staleness_cache[file_id] = (True, "file no longer exists on disk")

        # Apply cached staleness to each result
        for r in results:
            is_stale, reason = staleness_cache[r.chunk.file_id]
            r.is_stale = is_stale
            r.stale_reason = reason

    def _record_events(
        self,
        results: list[QueryResult],
        query_text: str,
        session_id: str,
    ) -> None:
        """
        Record ``'retrieved'`` events for all returned chunks.

        Args:
            results:    The final result list.
            query_text: The original query string.
            session_id: Active session identifier.
        """
        if not self.analytics:
            return
        original_session = self.analytics._session_id
        self.analytics._session_id = session_id
        try:
            for result in results:
                if result.chunk.chunk_id is not None:
                    self.analytics.record(
                        result.chunk.chunk_id,
                        "retrieved",
                        query_text=query_text,
                    )
        finally:
            self.analytics._session_id = original_session
