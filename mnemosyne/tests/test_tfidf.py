# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the TF-IDF embedding backend (embeddings/tfidf_backend.py).

Covers: vocabulary building, embed, similarity, inverted index search,
and ranking correctness.
"""

import unittest


def _default_config():
    import tempfile
    from mnemosyne.config import Config
    cfg = Config(root=tempfile.mkdtemp())
    # Lower min_df so the small test corpus passes the filter
    cfg.embedding.tfidf_min_df = 1
    cfg.embedding.tfidf_max_features = 1000
    return cfg


CORPUS = [
    "def authenticate_user(token):\n    return verify_token(token)\n",
    "def login_handler(request):\n    user = get_user_by_email(request.email)\n    return user\n",
    "class AuthMiddleware:\n    def process_request(self, request):\n        token = request.headers.get('Authorization')\n        return authenticate(token)\n",
    "def database_connection(host, port):\n    conn = connect(host, port)\n    return conn\n",
    "class DatabasePool:\n    def __init__(self, size):\n        self.pool = []\n",
    "def calculate_total(items):\n    return sum(item.price for item in items)\n",
    "def format_currency(amount, currency='USD'):\n    return f'{currency} {amount:.2f}'\n",
    "class PaymentProcessor:\n    def charge(self, card, amount):\n        return process_payment(card, amount)\n",
]


class TestTFIDF(unittest.TestCase):

    def setUp(self):
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        self.cfg = _default_config()
        self.backend = TFIDFBackend(self.cfg, store=None)
        self.backend.build_vocabulary(CORPUS)

    def test_vocabulary_non_empty_after_build(self):
        self.assertGreater(len(self.backend.vocabulary), 0)

    def test_idf_non_empty_after_build(self):
        self.assertGreater(len(self.backend.idf), 0)

    def test_total_docs_matches_corpus_size(self):
        self.assertEqual(self.backend.total_docs, len(CORPUS))

    def test_embed_returns_non_empty_dict(self):
        vec = self.backend.embed(CORPUS[0])
        self.assertIsInstance(vec, dict)
        self.assertGreater(len(vec), 0)

    def test_embed_values_are_floats(self):
        vec = self.backend.embed(CORPUS[0])
        for term, weight in vec.items():
            self.assertIsInstance(term, str)
            self.assertIsInstance(weight, float)

    def test_embed_weights_are_positive(self):
        vec = self.backend.embed(CORPUS[0])
        for weight in vec.values():
            self.assertGreater(weight, 0.0)

    def test_embed_only_vocabulary_terms(self):
        vec = self.backend.embed(CORPUS[0])
        for term in vec:
            self.assertIn(
                term, self.backend.idf,
                f"Term '{term}' in embed output is not in vocabulary IDF map",
            )

    def test_embed_empty_text_returns_empty(self):
        vec = self.backend.embed("")
        self.assertEqual(vec, {})

    def test_similarity_identical_vectors(self):
        vec = self.backend.embed(CORPUS[0])
        sim = self.backend.similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_similarity_orthogonal_vectors(self):
        """Completely disjoint term sets should yield similarity = 0."""
        sim = self.backend.similarity({"term_a": 1.0}, {"term_b": 1.0})
        self.assertAlmostEqual(sim, 0.0)

    def test_similarity_empty_vector(self):
        vec = self.backend.embed(CORPUS[0])
        self.assertAlmostEqual(self.backend.similarity({}, vec), 0.0)

    def test_search_returns_relevant_results(self):
        """A query about authentication should rank auth-related chunks first."""
        vectors = [(i, self.backend.embed(text)) for i, text in enumerate(CORPUS)]
        self.backend.build_inverted_index(vectors)

        results = self.backend.search("authenticate user token", top_k=5)
        self.assertGreater(len(results), 0)
        # Top result should be one of the auth-related chunks (indices 0, 1, 2)
        top_id, top_score = results[0]
        self.assertIn(
            top_id, {0, 1, 2},
            f"Expected auth-related chunk in top result, got chunk_id={top_id}",
        )

    def test_search_ranking_order_makes_sense(self):
        """Results are sorted by descending score."""
        vectors = [(i, self.backend.embed(text)) for i, text in enumerate(CORPUS)]
        self.backend.build_inverted_index(vectors)
        results = self.backend.search("payment processor charge", top_k=len(CORPUS))
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_search_empty_query_returns_empty(self):
        vectors = [(i, self.backend.embed(text)) for i, text in enumerate(CORPUS)]
        self.backend.build_inverted_index(vectors)
        results = self.backend.search("")
        self.assertEqual(results, [])

    def test_build_vocabulary_min_df_filter(self):
        """Terms appearing only once are filtered when min_df=2."""
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        cfg = _default_config()
        cfg.embedding.tfidf_min_df = 2
        backend = TFIDFBackend(cfg, store=None)
        # Give it a corpus where some terms appear only once
        corpus = [
            "common_term unique_term_alpha",
            "common_term unique_term_beta",
            "common_term unique_term_gamma",
        ]
        backend.build_vocabulary(corpus)
        # common_term appears in all 3 docs — should be in vocabulary
        self.assertIn("common_term", backend.vocabulary)
        # unique terms appear once each — should be filtered out
        self.assertNotIn("unique_term_alpha", backend.vocabulary)

    def test_camelcase_tokenization_expands_parts(self):
        """getUserById should expand to 'get', 'user', 'by', 'id' tokens."""
        tokens = self.backend.tokenize("getUserById")
        # Should contain the lowercased full token and sub-parts
        lower_tokens = [t.lower() for t in tokens]
        self.assertIn("getuserbyid", lower_tokens)

    def test_tokenize_returns_list_of_strings(self):
        tokens = self.backend.tokenize("def some_function(arg):")
        self.assertIsInstance(tokens, list)
        for tok in tokens:
            self.assertIsInstance(tok, str)

    def test_vocabulary_persistence_via_store(self):
        """build_vocabulary calls store.save_vocabulary and loading restores it."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(conn)
        from mnemosyne.store import Store
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

        store = Store(conn)
        cfg = _default_config()
        backend = TFIDFBackend(cfg, store=store)
        backend.build_vocabulary(CORPUS)

        # A fresh backend with the same store should load the vocabulary
        backend2 = TFIDFBackend(cfg, store=store)
        self.assertGreater(len(backend2.vocabulary), 0)
        self.assertGreater(len(backend2.idf), 0)


    def test_tokenizer_hash_is_deterministic(self):
        """_compute_tokenizer_hash returns the same value on repeated calls."""
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        backend = TFIDFBackend(_default_config(), store=None)
        h1 = backend._compute_tokenizer_hash()
        h2 = backend._compute_tokenizer_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)  # SHA-256 hex

    def test_tokenizer_hash_stored_on_build(self):
        """build_vocabulary persists the hash via store.set_index_metadata."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(conn)
        from mnemosyne.store import Store
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

        store = Store(conn)
        backend = TFIDFBackend(_default_config(), store=store)
        backend.build_vocabulary(CORPUS)

        stored = store.get_index_metadata("tokenizer_hash")
        self.assertIsNotNone(stored)
        self.assertEqual(stored, backend._compute_tokenizer_hash())

    def test_tokenizer_hash_matches_on_reload(self):
        """A fresh backend loading the same vocab sees matching hash, not stale."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(conn)
        from mnemosyne.store import Store
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

        store = Store(conn)
        b1 = TFIDFBackend(_default_config(), store=store)
        b1.build_vocabulary(CORPUS)

        # Second backend loads from same store — should NOT be stale
        b2 = TFIDFBackend(_default_config(), store=store)
        self.assertFalse(b2._vocabulary_stale)
        self.assertGreater(len(b2.vocabulary), 0)

    def test_stale_vocabulary_blocks_search(self):
        """When stopwords change, the backend returns [] from search()."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        from mnemosyne.schema import init_db
        init_db(conn)
        from mnemosyne.store import Store
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend, _STOPWORDS

        store = Store(conn)
        backend = TFIDFBackend(_default_config(), store=store)
        backend.build_vocabulary(CORPUS)

        vectors = [(i, backend.embed(t)) for i, t in enumerate(CORPUS)]
        backend.build_inverted_index(vectors)

        # Sanity: search works before mutation
        results_before = backend.search("authenticate user token", top_k=5)
        self.assertGreater(len(results_before), 0)

        # Now tamper with the stored hash to simulate a tokenizer change
        store.set_index_metadata("tokenizer_hash", "tampered_hash_value")

        # Reload into a fresh backend
        backend2 = TFIDFBackend(_default_config(), store=store)
        backend2.build_inverted_index(vectors)
        self.assertTrue(backend2._vocabulary_stale)

        results_after = backend2.search("authenticate user token", top_k=5)
        self.assertEqual(results_after, [])


if __name__ == "__main__":
    unittest.main()
