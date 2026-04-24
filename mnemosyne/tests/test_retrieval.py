# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the ranking utilities (ranking.py) and the RetrievalEngine (retrieval.py).
"""

import sqlite3
import tempfile
import os
import unittest


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------


def _make_memory_store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    from mnemosyne.schema import init_db
    init_db(conn)
    from mnemosyne.store import Store
    return Store(conn), conn


def _default_config():
    from mnemosyne.config import Config
    return Config(root=tempfile.mkdtemp())


def _insert_file_and_chunks(store, file_path, chunks_content, language="python"):
    from mnemosyne.models import FileRecord, Chunk, estimate_tokens
    from mnemosyne.hasher import content_hash

    rec = FileRecord(
        file_id=None,
        rel_path=file_path,
        content_hash=content_hash(" ".join(chunks_content)),
        size_bytes=sum(len(c) for c in chunks_content),
        language=language,
        last_modified=1700000000.0,
    )
    fid = store.upsert_file(rec)

    chunk_ids = []
    for i, content in enumerate(chunks_content):
        chunk = Chunk(
            chunk_id=None,
            file_id=fid,
            content_hash=content_hash(content + str(i)),
            chunk_type="function",
            line_start=i * 5 + 1,
            line_end=i * 5 + 4,
            token_count=estimate_tokens(content),
            content=content,
        )
        cid = store.insert_chunk(chunk)
        chunk_ids.append(cid)

    return fid, chunk_ids


# ---------------------------------------------------------------------------
# TestRanking -- rrf_fuse
# ---------------------------------------------------------------------------


class TestRanking(unittest.TestCase):

    def test_rrf_fuse_single_source(self):
        from mnemosyne.ranking import rrf_fuse
        score_lists = {"bm25": [(1, 0.9), (2, 0.5), (3, 0.1)]}
        weights = {"bm25": 1.0}
        results = rrf_fuse(score_lists, weights, k=60)
        ids = [r[0] for r in results]
        # Should be in order: 1, 2, 3 (highest bm25 score -> rank 1 -> highest rrf)
        self.assertEqual(ids, [1, 2, 3])

    def test_rrf_fuse_two_sources_merges(self):
        from mnemosyne.ranking import rrf_fuse
        score_lists = {
            "bm25":   [(1, 0.9), (2, 0.5)],
            "vector": [(2, 0.9), (1, 0.3)],
        }
        weights = {"bm25": 1.0, "vector": 1.0}
        results = rrf_fuse(score_lists, weights, k=60)
        ids = [r[0] for r in results]
        # Both chunk 1 and 2 should appear
        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_rrf_fuse_returns_sorted_descending(self):
        from mnemosyne.ranking import rrf_fuse
        score_lists = {
            "bm25":   [(i, float(10 - i)) for i in range(10)],
            "vector": [(i, float(i)) for i in range(10)],
        }
        weights = {"bm25": 1.0, "vector": 1.0}
        results = rrf_fuse(score_lists, weights, k=60)
        scores = [r[1] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rrf_fuse_missing_source_gets_penalty_rank(self):
        """A chunk absent from one source list gets penalty rank."""
        from mnemosyne.ranking import rrf_fuse
        score_lists = {
            "bm25":   [(1, 1.0)],
            "vector": [(2, 1.0)],
        }
        weights = {"bm25": 1.0, "vector": 1.0}
        results = rrf_fuse(score_lists, weights, k=60)
        self.assertEqual(len(results), 2)
        # Both should appear despite not overlapping
        ids = {r[0] for r in results}
        self.assertEqual(ids, {1, 2})

    def test_rrf_fuse_source_scores_dict_populated(self):
        from mnemosyne.ranking import rrf_fuse
        score_lists = {
            "bm25":   [(1, 0.8)],
            "vector": [(1, 0.6)],
        }
        weights = {"bm25": 0.5, "vector": 0.5}
        results = rrf_fuse(score_lists, weights, k=60)
        chunk_id, rrf_score, source_scores = results[0]
        self.assertIn("bm25", source_scores)
        self.assertIn("vector", source_scores)
        self.assertIn("rrf", source_scores)
        self.assertAlmostEqual(source_scores["bm25"], 0.8)
        self.assertAlmostEqual(source_scores["vector"], 0.6)

    def test_rrf_fuse_empty_lists(self):
        from mnemosyne.ranking import rrf_fuse
        results = rrf_fuse({}, {}, k=60)
        self.assertEqual(results, [])

    def test_rrf_fuse_single_item_single_source(self):
        from mnemosyne.ranking import rrf_fuse
        results = rrf_fuse({"bm25": [(42, 1.0)]}, {"bm25": 1.0}, k=60)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 42)

    def test_cost_model_score_proportional(self):
        from mnemosyne.ranking import cost_model_score
        score_50_tokens = cost_model_score(1, 1.0, 50)
        score_100_tokens = cost_model_score(1, 1.0, 100)
        self.assertGreater(score_50_tokens, score_100_tokens)

    def test_cost_model_score_minimum_token_count(self):
        from mnemosyne.ranking import cost_model_score
        import math
        # token_count=0 should not divide by zero; max(1, 0)=1
        score = cost_model_score(1, 1.0, 0)
        expected = 1.0 / (1.0 + math.log1p(1))
        self.assertAlmostEqual(score, expected, places=6)


# ---------------------------------------------------------------------------
# TestRetrieval -- end-to-end with a populated database
# ---------------------------------------------------------------------------


CORPUS_CHUNKS = [
    ("auth/middleware.py", [
        "def authenticate_user(token):\n    user = verify_jwt(token)\n    return user\n",
        "def require_auth(view_func):\n    def wrapper(request):\n        token = request.headers.get('Authorization')\n        return authenticate_user(token)\n    return wrapper\n",
    ]),
    ("db/connection.py", [
        "def get_database_connection(host, port, user, password):\n    conn = psycopg2.connect(host=host, port=port)\n    return conn\n",
        "class ConnectionPool:\n    def __init__(self, size=10):\n        self.pool = []\n        self.size = size\n",
    ]),
    ("payments/processor.py", [
        "def process_payment(card_number, amount, currency):\n    result = stripe.charge(card_number, amount)\n    return result\n",
        "def calculate_total(items):\n    return sum(item.price for item in items)\n",
    ]),
]


class TestRetrieval(unittest.TestCase):

    def setUp(self):
        self.store, self.conn = _make_memory_store()
        self.cfg = _default_config()
        # Lower min_df for small corpus
        self.cfg.embedding.tfidf_min_df = 1
        self.cfg.retrieval.max_results = 10
        self.cfg.retrieval.token_budget = 2000

        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        self.tfidf = TFIDFBackend(self.cfg, store=None)

        # Ingest corpus
        all_chunk_ids = []
        all_texts = []
        for file_path, chunks in CORPUS_CHUNKS:
            fid, cids = _insert_file_and_chunks(self.store, file_path, chunks)
            for cid, content in zip(cids, chunks):
                all_chunk_ids.append(cid)
                all_texts.append(content)

        # Build TF-IDF vocabulary over the whole corpus
        self.tfidf.build_vocabulary(all_texts)

        # Build and store sparse embeddings
        chunk_vectors = []
        for cid, text in zip(all_chunk_ids, all_texts):
            vec = self.tfidf.embed(text)
            if vec:
                self.store.insert_sparse_embedding(cid, vec)
                chunk_vectors.append((cid, vec))

        self.tfidf.build_inverted_index(chunk_vectors)

        from mnemosyne.retrieval import RetrievalEngine
        self.engine = RetrievalEngine(
            store=self.store,
            tfidf_backend=self.tfidf,
            config=self.cfg,
        )

    def test_query_returns_non_empty_results(self):
        results = self.engine.query("authenticate user token")
        self.assertGreater(len(results), 0)

    def test_query_results_are_query_result_objects(self):
        from mnemosyne.models import QueryResult
        results = self.engine.query("authenticate user token")
        for r in results:
            self.assertIsInstance(r, QueryResult)

    def test_auth_query_ranks_auth_chunk_first(self):
        """Querying for authentication should surface auth/middleware.py chunks."""
        results = self.engine.query("authenticate user token")
        top_file = results[0].file_path if results else None
        self.assertIsNotNone(top_file)
        self.assertIn("auth", top_file)

    def test_database_query_surfaces_db_chunk(self):
        """Querying for database connection should surface db/connection.py."""
        results = self.engine.query("database connection pool")
        paths = [r.file_path for r in results]
        self.assertTrue(
            any("db" in p or "connection" in p for p in paths),
            f"Expected db chunk in results, got paths: {paths}",
        )

    def test_payment_query_surfaces_payment_chunk(self):
        """Querying for payment processing should surface payments/processor.py."""
        results = self.engine.query("process payment card charge")
        paths = [r.file_path for r in results]
        self.assertTrue(
            any("payment" in p or "processor" in p for p in paths),
            f"Expected payment chunk in results, got paths: {paths}",
        )

    def test_query_results_have_scores(self):
        results = self.engine.query("authenticate user")
        for r in results:
            self.assertIsInstance(r.scores, dict)
            self.assertIn("rrf", r.scores)

    def test_query_empty_string_returns_empty(self):
        results = self.engine.query("")
        self.assertEqual(results, [])

    def test_query_token_budget_respected(self):
        """Total tokens of results should not exceed the configured budget."""
        from mnemosyne.models import estimate_tokens
        budget = self.cfg.retrieval.token_budget
        results = self.engine.query("authenticate user token", budget=budget)
        total = sum(estimate_tokens(r.chunk.content) for r in results)
        self.assertLessEqual(total, budget)

    def test_query_with_small_budget_returns_fewer_results(self):
        """A very small token budget should limit the number of results."""
        results_large = self.engine.query("function", budget=10000)
        results_small = self.engine.query("function", budget=10)
        self.assertLessEqual(len(results_small), len(results_large))


# ---------------------------------------------------------------------------
# TestStaleness -- staleness detection at query time
# ---------------------------------------------------------------------------


class TestStaleness(unittest.TestCase):
    """
    Verify that QueryResult.is_stale and .stale_reason are set correctly
    depending on whether the underlying file has been modified or deleted
    since it was indexed.
    """

    def _setup_engine_with_real_files(self, tmpdir, file_contents):
        """
        Create real files in *tmpdir*, ingest them into an in-memory store,
        and return (engine, store, file_paths_on_disk).

        *file_contents* is a list of (rel_path, content_str) tuples.
        """
        from mnemosyne.config import Config
        from mnemosyne.models import FileRecord, Chunk, estimate_tokens
        from mnemosyne.hasher import content_hash
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        from mnemosyne.retrieval import RetrievalEngine

        store, conn = _make_memory_store()
        cfg = Config(root=tmpdir)
        cfg.embedding.tfidf_min_df = 1
        cfg.retrieval.max_results = 10
        cfg.retrieval.token_budget = 5000

        all_chunk_ids = []
        all_texts = []
        abs_paths = {}

        for rel_path, content in file_contents:
            # Write real file to disk
            abs_path = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w") as f:
                f.write(content)

            disk_mtime = os.path.getmtime(abs_path)
            abs_paths[rel_path] = abs_path

            rec = FileRecord(
                file_id=None,
                rel_path=rel_path,
                content_hash=content_hash(content),
                size_bytes=len(content.encode()),
                language="python",
                last_modified=disk_mtime,
            )
            fid = store.upsert_file(rec)

            chunk = Chunk(
                chunk_id=None,
                file_id=fid,
                content_hash=content_hash(content),
                chunk_type="function",
                line_start=1,
                line_end=content.count("\n") + 1,
                token_count=estimate_tokens(content),
                content=content,
            )
            cid = store.insert_chunk(chunk)
            all_chunk_ids.append(cid)
            all_texts.append(content)

        tfidf = TFIDFBackend(cfg, store=None)
        tfidf.build_vocabulary(all_texts)
        chunk_vectors = []
        for cid, text in zip(all_chunk_ids, all_texts):
            vec = tfidf.embed(text)
            if vec:
                store.insert_sparse_embedding(cid, vec)
                chunk_vectors.append((cid, vec))
        tfidf.build_inverted_index(chunk_vectors)

        engine = RetrievalEngine(
            store=store,
            tfidf_backend=tfidf,
            config=cfg,
        )
        return engine, store, abs_paths

    def test_staleness_detection_fresh(self):
        """Files unchanged on disk should have is_stale=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, store, _ = self._setup_engine_with_real_files(tmpdir, [
                ("auth.py", "def authenticate_user(token):\n    return verify(token)\n"),
                ("db.py", "def get_connection(host, port):\n    return connect(host)\n"),
            ])
            results = engine.query("authenticate user token")
            self.assertGreater(len(results), 0)
            for r in results:
                self.assertFalse(r.is_stale, f"Expected fresh but got stale: {r.file_path}")
                self.assertIsNone(r.stale_reason)

    def test_staleness_detection_stale(self):
        """A file modified on disk after indexing should have is_stale=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, store, abs_paths = self._setup_engine_with_real_files(tmpdir, [
                ("auth.py", "def authenticate_user(token):\n    return verify(token)\n"),
                ("db.py", "def get_connection(host, port):\n    return connect(host)\n"),
            ])
            # Modify auth.py on disk -- force a distinctly different mtime.
            # Filesystem mtime resolution can be coarse (1s on some FS), so
            # we explicitly set a future timestamp to guarantee detection.
            auth_path = abs_paths["auth.py"]
            with open(auth_path, "a") as f:
                f.write("# modified\n")
            future_time = os.path.getmtime(auth_path) + 10
            os.utime(auth_path, (future_time, future_time))

            results = engine.query("authenticate user token")
            self.assertGreater(len(results), 0)

            # Find results from auth.py and verify they are stale
            auth_results = [r for r in results if "auth" in r.file_path]
            self.assertGreater(len(auth_results), 0, "Expected auth.py in results")
            for r in auth_results:
                self.assertTrue(r.is_stale, f"Expected stale for {r.file_path}")
                self.assertEqual(r.stale_reason, "file modified since last index")

            # db.py should still be fresh
            db_results = [r for r in results if "db" in r.file_path]
            for r in db_results:
                self.assertFalse(r.is_stale, f"Expected fresh for {r.file_path}")

    def test_staleness_detection_deleted(self):
        """A file removed from disk should have is_stale=True with deletion reason."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, store, abs_paths = self._setup_engine_with_real_files(tmpdir, [
                ("auth.py", "def authenticate_user(token):\n    return verify(token)\n"),
                ("db.py", "def get_connection(host, port):\n    return connect(host)\n"),
            ])
            # Delete auth.py from disk
            os.remove(abs_paths["auth.py"])

            results = engine.query("authenticate user token")
            self.assertGreater(len(results), 0)

            # Find results from auth.py and verify stale with deletion reason
            auth_results = [r for r in results if "auth" in r.file_path]
            self.assertGreater(len(auth_results), 0, "Expected auth.py in results")
            for r in auth_results:
                self.assertTrue(r.is_stale, f"Expected stale for {r.file_path}")
                self.assertEqual(r.stale_reason, "file no longer exists on disk")


# ---------------------------------------------------------------------------
# TestInjectionHeuristics -- Milestone 1.3 precision tuning
# ---------------------------------------------------------------------------


class TestInjectionHeuristics(unittest.TestCase):
    """
    Tests for tightened injection heuristics in _filename_boost,
    _file_level_filter, and _import_graph_boost (Milestone 1.3).
    """

    def _make_engine(self, file_specs):
        """
        Build a minimal RetrievalEngine with the given files.

        *file_specs* is a list of (rel_path, [chunk_contents]) tuples.
        Returns (engine, store, config, {rel_path: (file_id, [chunk_ids])}).
        """
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        from mnemosyne.retrieval import RetrievalEngine

        store, conn = _make_memory_store()
        cfg = _default_config()
        cfg.embedding.tfidf_min_df = 1
        cfg.retrieval.max_results = 20
        cfg.retrieval.token_budget = 5000

        all_cids = []
        all_texts = []
        file_map = {}

        for rel_path, chunks in file_specs:
            fid, cids = _insert_file_and_chunks(store, rel_path, chunks)
            file_map[rel_path] = (fid, cids)
            for cid, content in zip(cids, chunks):
                all_cids.append(cid)
                all_texts.append(content)

        tfidf = TFIDFBackend(cfg, store=None)
        tfidf.build_vocabulary(all_texts)
        chunk_vectors = []
        for cid, text in zip(all_cids, all_texts):
            vec = tfidf.embed(text)
            if vec:
                store.insert_sparse_embedding(cid, vec)
                chunk_vectors.append((cid, vec))
        tfidf.build_inverted_index(chunk_vectors)

        engine = RetrievalEngine(
            store=store, tfidf_backend=tfidf, config=cfg,
        )
        return engine, store, cfg, file_map

    # ------------------------------------------------------------------
    # Test 1: filename boost rejects short prefix
    # ------------------------------------------------------------------

    def test_filename_boost_rejects_short_prefix(self):
        """
        Query term 'test' (4 chars) should NOT boost a file named
        'testing_utils.py' -- the term is too short (< 5 chars) to qualify
        for stem-prefix matching.
        """
        engine, store, cfg, file_map = self._make_engine([
            ("testing_utils.py", [
                "def helper():\n    return 'utility'\n",
            ]),
            ("auth.py", [
                "def authenticate(token):\n    return verify(token)\n",
            ]),
        ])

        # Build a synthetic fused result that includes both files
        fused = []
        for rel_path, (fid, cids) in file_map.items():
            for cid in cids:
                fused.append((cid, 0.5, {"bm25": 0.5, "rrf": 0.5}))

        # "test" is only 4 characters -- should not boost testing_utils.py
        result = engine._filename_boost(fused, "test the function")

        # No chunk should have been boosted (scores should remain 0.5)
        for cid, score, _ in result:
            self.assertAlmostEqual(
                score, 0.5,
                msg="Short prefix 'test' should not boost testing_utils.py",
            )

    # ------------------------------------------------------------------
    # Test 2: filename boost does not inject new files
    # ------------------------------------------------------------------

    def test_filename_boost_no_new_file_injection(self):
        """
        A file that was NOT in the fused results should NOT be injected
        by _filename_boost, even if its name matches a query term.
        """
        engine, store, cfg, file_map = self._make_engine([
            ("scorer.py", [
                "def compute_score(data):\n    return sum(data)\n",
            ]),
            ("auth.py", [
                "def authenticate(token):\n    return verify(token)\n",
            ]),
        ])

        # Build fused results containing ONLY auth.py chunks
        auth_fid, auth_cids = file_map["auth.py"]
        fused = [(cid, 0.5, {"bm25": 0.5, "rrf": 0.5}) for cid in auth_cids]

        # Query mentions "scorer" which matches scorer.py, but scorer.py
        # is not in the fused results -- it should NOT be injected.
        result = engine._filename_boost(fused, "scorer function details")

        result_chunk_ids = {cid for cid, _, _ in result}
        scorer_fid, scorer_cids = file_map["scorer.py"]
        for cid in scorer_cids:
            self.assertNotIn(
                cid, result_chunk_ids,
                "scorer.py chunks should not be injected -- file was not in fused results",
            )

    # ------------------------------------------------------------------
    # Test 3: file-level filter adaptive sizing
    # ------------------------------------------------------------------

    def test_file_level_filter_adaptive(self):
        """
        Two-pass file filter: top-N by aggregate score PLUS chunk-qualified
        files from top-50 individual chunks.

        With few fused entries (< 50), all files are chunk-qualified so
        the surviving set equals the number of unique files.  With many
        entries, only the aggregate top-N plus chunk-qualified files survive.
        """
        store, conn = _make_memory_store()
        cfg = _default_config()
        cfg.embedding.tfidf_min_df = 1
        cfg.retrieval.max_results = 50
        cfg.retrieval.token_budget = 50000
        cfg.retrieval.max_files = 0

        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        from mnemosyne.retrieval import RetrievalEngine

        # Create 30 files with 1 chunk each
        all_cids = []
        all_texts = []
        for i in range(30):
            content = f"def func_{i}():\n    return {i}\n"
            fid, cids = _insert_file_and_chunks(
                store, f"mod_{i:02d}.py", [content],
            )
            all_cids.extend(cids)
            all_texts.append(content)

        tfidf = TFIDFBackend(cfg, store=None)
        tfidf.build_vocabulary(all_texts)
        chunk_vectors = []
        for cid, text in zip(all_cids, all_texts):
            vec = tfidf.embed(text)
            if vec:
                store.insert_sparse_embedding(cid, vec)
                chunk_vectors.append((cid, vec))
        tfidf.build_inverted_index(chunk_vectors)

        engine = RetrievalEngine(
            store=store, tfidf_backend=tfidf, config=cfg,
        )

        # 12 fused entries < 50, so all chunk-qualify.  All 12 files survive.
        fused_12 = [(cid, 1.0 / (1 + cid), {"bm25": 0.5, "rrf": 0.5}) for cid in all_cids[:12]]
        result_12 = engine._file_level_filter(fused_12)
        file_ids_12 = {store.get_chunk(cid).file_id for cid, _, _ in result_12 if store.get_chunk(cid)}
        self.assertEqual(len(file_ids_12), 12, f"12 fused (all chunk-qualify) -> expected 12, got {len(file_ids_12)}")

        # 30 fused entries < 50, so all chunk-qualify.  All 30 files survive.
        fused_30 = [(cid, 1.0 / (1 + cid), {"bm25": 0.5, "rrf": 0.5}) for cid in all_cids[:30]]
        result_30 = engine._file_level_filter(fused_30)
        file_ids_30 = {store.get_chunk(cid).file_id for cid, _, _ in result_30 if store.get_chunk(cid)}
        self.assertEqual(len(file_ids_30), 30, f"30 fused (all chunk-qualify) -> expected 30, got {len(file_ids_30)}")

        # Verify soft penalty: chunks from files NOT in top-N get 0.7x scores
        top_n = min(max(4, 30 // 3), 10)  # = 10
        # The top-10 files by max score keep full scores, rest get penalized
        penalized_chunks = [(cid, rrf, sc) for cid, rrf, sc in result_30
                           if store.get_chunk(cid) and store.get_chunk(cid).file_id not in
                           {store.get_chunk(c).file_id for c, _, _ in sorted(fused_30, key=lambda x: x[1], reverse=True)[:top_n]}]
        for cid, rrf, sc in penalized_chunks[:3]:
            orig = next(r for c, r, _ in fused_30 if c == cid)
            self.assertAlmostEqual(rrf, orig * 0.7, places=5)

    # ------------------------------------------------------------------
    # Test 4: import graph single reference not injected
    # ------------------------------------------------------------------

    def test_import_graph_single_reference_not_injected(self):
        """
        A file referenced only once from retrieved results should NOT be
        injected by _import_graph_boost (minimum threshold is 2).
        """
        engine, store, cfg, file_map = self._make_engine([
            ("main.py", [
                "import utils\ndef main():\n    utils.helper()\n",
            ]),
            ("utils.py", [
                "def helper():\n    return 42\n",
            ]),
            ("config.py", [
                "DB_HOST = 'localhost'\nDB_PORT = 5432\n",
            ]),
        ])

        # Ensure import_inject_threshold is 2 (default)
        cfg.retrieval.import_inject_threshold = 2

        # Build fused results with ONLY main.py (which imports utils.py once)
        main_fid, main_cids = file_map["main.py"]
        fused = [(cid, 0.8, {"bm25": 0.8, "rrf": 0.8}) for cid in main_cids]

        result = engine._import_graph_boost(fused)

        # utils.py is referenced only once -- should NOT be injected
        result_chunk_ids = {cid for cid, _, _ in result}
        utils_fid, utils_cids = file_map["utils.py"]
        for cid in utils_cids:
            self.assertNotIn(
                cid, result_chunk_ids,
                "utils.py (referenced once) should not be injected -- below threshold",
            )


# ---------------------------------------------------------------------------
# TestIsTestDemotion -- fuse-time is_test score demotion + env flag gating
# ---------------------------------------------------------------------------


class TestIsTestDemotion(unittest.TestCase):
    """
    Tests for Wave 0 fuse-time is_test demotion (MNEMOSYNE_TEST_DEMOTION).

    The demotion applies a score multiplier to RRF results whose source file
    is a test file.  It is gated behind two env flags read once per engine
    construction: MNEMOSYNE_TEST_DEMOTION (on/off) and
    MNEMOSYNE_TEST_DEMOTION_FACTOR (float in (0.0, 1.0]).
    """

    # Environment keys we manipulate; always restore in tearDown.
    _ENV_TOGGLE = "MNEMOSYNE_TEST_DEMOTION"
    _ENV_FACTOR = "MNEMOSYNE_TEST_DEMOTION_FACTOR"

    def setUp(self):
        self._saved_env = {
            k: os.environ.get(k)
            for k in (self._ENV_TOGGLE, self._ENV_FACTOR)
        }

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set_env(self, toggle=None, factor=None):
        if toggle is None:
            os.environ.pop(self._ENV_TOGGLE, None)
        else:
            os.environ[self._ENV_TOGGLE] = toggle
        if factor is None:
            os.environ.pop(self._ENV_FACTOR, None)
        else:
            os.environ[self._ENV_FACTOR] = factor

    def _build_engine_with_prod_and_test(self):
        """Seed an engine with one prod chunk and one test chunk.

        Both chunks have identical content so raw BM25 + TF-IDF scores match;
        any rank difference comes from the fuse-time demotion.

        Returns (engine, prod_chunk_id, test_chunk_id).
        """
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        from mnemosyne.retrieval import RetrievalEngine

        store, _ = _make_memory_store()
        cfg = _default_config()
        cfg.embedding.tfidf_min_df = 1
        cfg.retrieval.max_results = 10
        cfg.retrieval.token_budget = 5000

        # Identical content for both -- forces raw signals to tie.
        identical = "def authenticate(token):\n    return verify(token)\n"

        prod_fid, prod_cids = _insert_file_and_chunks(
            store, "src/auth.py", [identical],
        )
        test_fid, test_cids = _insert_file_and_chunks(
            store, "tests/test_auth.py", [identical],
        )

        tfidf = TFIDFBackend(cfg, store=None)
        tfidf.build_vocabulary([identical, identical])
        chunk_vectors = []
        for cid in prod_cids + test_cids:
            vec = tfidf.embed(identical)
            if vec:
                store.insert_sparse_embedding(cid, vec)
                chunk_vectors.append((cid, vec))
        tfidf.build_inverted_index(chunk_vectors)

        engine = RetrievalEngine(
            store=store, tfidf_backend=tfidf, config=cfg,
        )
        return engine, prod_cids[0], test_cids[0]

    # ------------------------------------------------------------------
    # Test b: demotion applied at fuse time with identical raw signals
    # ------------------------------------------------------------------

    def test_is_test_demotion_applied_at_fuse_time(self):
        """
        With identical BM25 + TF-IDF signals, the prod chunk must outrank
        the test chunk after fuse-time demotion.  The score delta must
        match the configured 0.7 default factor.
        """
        self._set_env(toggle="on", factor=None)  # default 0.7
        engine, prod_cid, test_cid = self._build_engine_with_prod_and_test()

        # Build synthetic fused list with IDENTICAL raw scores.
        fused = [
            (prod_cid, 1.0, {"bm25": 1.0, "vector": 1.0, "rrf": 1.0}),
            (test_cid, 1.0, {"bm25": 1.0, "vector": 1.0, "rrf": 1.0}),
        ]

        out = engine._apply_test_demotion(fused)

        # Map chunk_id -> (score, sources) for assertions.
        by_cid = {cid: (score, sources) for cid, score, sources in out}
        self.assertIn(prod_cid, by_cid)
        self.assertIn(test_cid, by_cid)

        prod_score, prod_sources = by_cid[prod_cid]
        test_score, test_sources = by_cid[test_cid]

        # Prod chunk must rank first (higher score).
        self.assertEqual(out[0][0], prod_cid)
        self.assertEqual(out[1][0], test_cid)

        # Prod stays at 1.0 and has no test_demotion marker.
        self.assertAlmostEqual(prod_score, 1.0, places=6)
        self.assertNotIn("test_demotion", prod_sources)

        # Test chunk is demoted to 1.0 * 0.7 = 0.7.
        self.assertAlmostEqual(test_score, 0.7, places=6)
        self.assertAlmostEqual(test_sources["rrf"], 0.7, places=6)
        self.assertAlmostEqual(test_sources["test_demotion"], 0.7, places=6)

    # ------------------------------------------------------------------
    # Test c: env=off preserves raw ordering
    # ------------------------------------------------------------------

    def test_is_test_demotion_env_off_preserves_order(self):
        """
        With MNEMOSYNE_TEST_DEMOTION=off, the demotion path is a no-op.
        Tests must be allowed to tie or outrank prod when raw signals say so.
        """
        self._set_env(toggle="off", factor=None)
        engine, prod_cid, test_cid = self._build_engine_with_prod_and_test()

        # Raw signals favour the test chunk so the tie-break would have
        # otherwise flipped to prod under demotion.
        fused = [
            (test_cid, 0.9, {"bm25": 0.9, "rrf": 0.9}),
            (prod_cid, 0.8, {"bm25": 0.8, "rrf": 0.8}),
        ]

        out = engine._apply_test_demotion(fused)

        # Order is preserved and scores are untouched.
        self.assertEqual([cid for cid, _, _ in out], [test_cid, prod_cid])
        self.assertAlmostEqual(out[0][1], 0.9, places=6)
        self.assertAlmostEqual(out[1][1], 0.8, places=6)

        # No test_demotion marker leaks into source scores when disabled.
        for _, _, sources in out:
            self.assertNotIn("test_demotion", sources)

    # ------------------------------------------------------------------
    # Test d: env factor override
    # ------------------------------------------------------------------

    def test_is_test_demotion_factor_env_override(self):
        """
        Setting MNEMOSYNE_TEST_DEMOTION_FACTOR=0.5 must be honoured when the
        engine is constructed: the multiplier applied to test chunks is 0.5.
        """
        self._set_env(toggle="on", factor="0.5")
        engine, prod_cid, test_cid = self._build_engine_with_prod_and_test()

        fused = [
            (prod_cid, 1.0, {"bm25": 1.0, "rrf": 1.0}),
            (test_cid, 1.0, {"bm25": 1.0, "rrf": 1.0}),
        ]

        out = engine._apply_test_demotion(fused)
        by_cid = {cid: (score, sources) for cid, score, sources in out}

        self.assertEqual(out[0][0], prod_cid)
        self.assertAlmostEqual(by_cid[prod_cid][0], 1.0, places=6)
        self.assertAlmostEqual(by_cid[test_cid][0], 0.5, places=6)
        self.assertAlmostEqual(
            by_cid[test_cid][1]["test_demotion"], 0.5, places=6,
        )

    # ------------------------------------------------------------------
    # Extra: invalid factor falls back to default (defensive)
    # ------------------------------------------------------------------

    def test_is_test_demotion_invalid_factor_falls_back(self):
        """
        A bad MNEMOSYNE_TEST_DEMOTION_FACTOR value (non-float, out of range)
        must fall back to the 0.7 default rather than silently disabling
        the feature or propagating an invalid multiplier.
        """
        self._set_env(toggle="on", factor="nope")  # non-numeric
        engine, prod_cid, test_cid = self._build_engine_with_prod_and_test()
        self.assertTrue(engine._test_demotion_enabled)
        self.assertAlmostEqual(engine._test_demotion_factor, 0.7, places=6)

        # Out-of-range (0.0 or >1.0) also falls back to 0.7.
        self._set_env(toggle="on", factor="0.0")
        engine2, _, _ = self._build_engine_with_prod_and_test()
        self.assertAlmostEqual(engine2._test_demotion_factor, 0.7, places=6)

        self._set_env(toggle="on", factor="1.5")
        engine3, _, _ = self._build_engine_with_prod_and_test()
        self.assertAlmostEqual(engine3._test_demotion_factor, 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
