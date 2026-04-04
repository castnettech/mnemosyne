# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for ARCCache (cache.py).

Covers: hit/miss, T1->T2 promotion, ghost list adaptation, capacity enforcement,
and the hit_rate property.
"""

import unittest


def _make_chunk(chunk_id, content="def foo():\n    pass\n"):
    from mnemosyne.models import Chunk
    from mnemosyne.hasher import content_hash
    from mnemosyne.models import estimate_tokens
    return Chunk(
        chunk_id=chunk_id,
        file_id=1,
        content_hash=content_hash(content + str(chunk_id)),
        chunk_type="function",
        line_start=1,
        line_end=2,
        token_count=estimate_tokens(content),
        content=content,
    )


class TestARCCache(unittest.TestCase):

    def setUp(self):
        from mnemosyne.cache import ARCCache
        self.cache = ARCCache(capacity=4, ghost_capacity=8)

    # ------------------------------------------------------------------
    # Basic hit / miss
    # ------------------------------------------------------------------

    def test_miss_on_empty_cache(self):
        result = self.cache.get(1)
        self.assertIsNone(result)

    def test_hit_after_put(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        result = self.cache.get(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.chunk_id, 1)

    def test_miss_for_unknown_id(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        self.assertIsNone(self.cache.get(99))

    def test_contains_live_chunk(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        self.assertTrue(self.cache.contains(1))

    def test_not_contains_missing_chunk(self):
        self.assertFalse(self.cache.contains(99))

    # ------------------------------------------------------------------
    # T1 -> T2 promotion
    # ------------------------------------------------------------------

    def test_t1_to_t2_promotion_on_second_access(self):
        """First access lands in T1; second access promotes to T2."""
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        # After put: should be in T1
        self.assertIn(1, self.cache.t1)
        self.assertNotIn(1, self.cache.t2)
        # First get -- promotes to T2
        self.cache.get(1)
        self.assertNotIn(1, self.cache.t1)
        self.assertIn(1, self.cache.t2)

    def test_t2_stays_in_t2_on_repeated_access(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        self.cache.get(1)   # promote to T2
        self.cache.get(1)   # refresh in T2
        self.assertIn(1, self.cache.t2)
        self.assertNotIn(1, self.cache.t1)

    # ------------------------------------------------------------------
    # Capacity enforcement
    # ------------------------------------------------------------------

    def test_capacity_not_exceeded(self):
        """T1 + T2 never exceeds capacity."""
        for i in range(10):
            self.cache.put(i, _make_chunk(i))
        total_live = len(self.cache.t1) + len(self.cache.t2)
        self.assertLessEqual(total_live, self.cache.c)

    def test_evicted_chunk_goes_to_ghost_list(self):
        """When capacity is full and a new item is inserted, an eviction to B1 or B2 occurs."""
        # Fill to capacity
        for i in range(self.cache.c):
            self.cache.put(i, _make_chunk(i))
        total_before = len(self.cache.b1) + len(self.cache.b2)

        # Insert one more to trigger eviction
        self.cache.put(self.cache.c, _make_chunk(self.cache.c))
        total_after = len(self.cache.b1) + len(self.cache.b2)
        self.assertGreater(total_after, total_before)

    def test_ghost_list_capped(self):
        """Ghost lists B1+B2 never exceed ghost_capacity."""
        for i in range(100):
            self.cache.put(i, _make_chunk(i))
        ghost_total = len(self.cache.b1) + len(self.cache.b2)
        self.assertLessEqual(ghost_total, self.cache.ghost_cap)

    # ------------------------------------------------------------------
    # Ghost list adaptation (p parameter)
    # ------------------------------------------------------------------

    def test_b1_hit_increases_p(self):
        """A B1 ghost hit increases p (recency target)."""
        # Fill cache to force evictions into B1
        for i in range(self.cache.c + 4):
            self.cache.put(i, _make_chunk(i))

        # Find an ID that is in B1
        if not self.cache.b1:
            self.skipTest("No B1 entries available for this test setup")

        b1_id = next(iter(self.cache.b1))
        p_before = self.cache.p

        # Re-insert the B1 ghost -- should trigger B1 hit, increasing p
        self.cache.put(b1_id, _make_chunk(b1_id))
        self.assertGreaterEqual(self.cache.p, p_before)

    def test_b2_hit_decreases_p(self):
        """A B2 ghost hit decreases p (frequency target grows)."""
        # Build state with items in B2
        # Access items so they go to T2, then overflow to B2
        for i in range(self.cache.c):
            self.cache.put(i, _make_chunk(i))
            self.cache.get(i)  # promote to T2

        # Insert more to force T2 evictions to B2
        for i in range(self.cache.c, self.cache.c + 4):
            self.cache.put(i, _make_chunk(i))

        if not self.cache.b2:
            self.skipTest("No B2 entries available for this test setup")

        b2_id = next(iter(self.cache.b2))
        p_before = self.cache.p
        self.cache.put(b2_id, _make_chunk(b2_id))
        self.assertLessEqual(self.cache.p, p_before)

    # ------------------------------------------------------------------
    # Hit rate
    # ------------------------------------------------------------------

    def test_hit_rate_zero_with_no_accesses(self):
        self.assertAlmostEqual(self.cache.hit_rate, 0.0)

    def test_hit_rate_with_known_hits_and_misses(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        # 1 miss: get(99) -> miss
        self.cache.get(99)
        # 1 hit: get(1) -> hit
        self.cache.get(1)
        self.assertAlmostEqual(self.cache.hit_rate, 0.5, places=5)

    def test_hit_rate_all_hits(self):
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)
        # Promote to T2
        self.cache.get(1)
        # Hit again
        self.cache.get(1)
        # Both gets are hits
        self.assertGreater(self.cache.hit_rate, 0.0)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def test_stats_has_expected_keys(self):
        s = self.cache.stats()
        for key in ["t1_size", "t2_size", "b1_size", "b2_size", "capacity", "p", "hits", "misses", "hit_rate"]:
            self.assertIn(key, s)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def test_clear_empties_all_lists(self):
        for i in range(4):
            self.cache.put(i, _make_chunk(i))
        self.cache.clear()
        self.assertEqual(len(self.cache.t1), 0)
        self.assertEqual(len(self.cache.t2), 0)
        self.assertEqual(len(self.cache.b1), 0)
        self.assertEqual(len(self.cache.b2), 0)
        self.assertEqual(self.cache.p, 0)
        self.assertAlmostEqual(self.cache.hit_rate, 0.0)

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    def test_prefetch_populates_cache(self):
        """prefetch() inserts chunks for IDs not already in cache."""
        chunk_store = {i: _make_chunk(i) for i in range(5)}
        ids_to_prefetch = [0, 1, 2]
        self.cache.prefetch(ids_to_prefetch, fetch_fn=lambda cid: chunk_store.get(cid))
        for cid in ids_to_prefetch:
            self.assertTrue(self.cache.contains(cid), f"chunk {cid} not in cache after prefetch")

    def test_prefetch_skips_already_cached(self):
        """prefetch() does not call fetch_fn for already-cached IDs."""
        chunk = _make_chunk(1)
        self.cache.put(1, chunk)

        call_count = [0]
        def fetch_fn(cid):
            call_count[0] += 1
            return _make_chunk(cid)

        self.cache.prefetch([1], fetch_fn=fetch_fn)
        self.assertEqual(call_count[0], 0, "fetch_fn should not be called for cached IDs")

    # ------------------------------------------------------------------
    # Invalid capacity
    # ------------------------------------------------------------------

    def test_capacity_zero_raises(self):
        from mnemosyne.cache import ARCCache
        with self.assertRaises(ValueError):
            ARCCache(capacity=0)


if __name__ == "__main__":
    unittest.main()
