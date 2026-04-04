# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hot/warm/cold tier manager for Mnemosyne.

Wraps the ARC cache (hot tier) and the persistent Store (cold tier) behind a
single ``get`` interface.  A "warm" tier is defined as chunks present in the
database but not currently in the ARC cache -- they are returned after being
promoted into the cache on access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemosyne.cache import ARCCache

if TYPE_CHECKING:
    from mnemosyne.models import Chunk
    from mnemosyne.store import Store


class TierManager:
    """
    Three-tier chunk access manager.

    Tiers:

    * **hot**  -- live in the in-memory :class:`~mnemosyne.cache.ARCCache`.
    * **warm** -- on disk in the :class:`~mnemosyne.store.Store`, but recently
                 accessed (present in a ghost list or likely to be fetched
                 again soon).  Fetching a warm chunk promotes it to hot.
    * **cold** -- on disk, not recently accessed.  Fetching a cold chunk
                 promotes it to hot.

    In practice "warm" vs "cold" is determined by ghost list membership: if
    the chunk's id is in B1 or B2 of the ARC cache, it is warm; otherwise
    cold.  Both warm and cold reads result in a cache insert.

    Args:
        cache: The :class:`~mnemosyne.cache.ARCCache` instance.
        store: The persistent :class:`~mnemosyne.store.Store` instance.
    """

    def __init__(self, cache: ARCCache, store: "Store") -> None:
        self.cache = cache
        self.store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, chunk_id: int) -> "Chunk | None":
        """
        Retrieve *chunk_id*, checking the hot cache first.

        On a cache miss the chunk is loaded from the store and inserted into
        the cache before returning.

        Returns:
            The :class:`~mnemosyne.models.Chunk`, or ``None`` if not found
            anywhere.
        """
        cached = self.cache.get(chunk_id)
        if cached is not None:
            return cached

        chunk = self.store.get_chunk(chunk_id)
        if chunk is not None:
            self.cache.put(chunk_id, chunk)
        return chunk

    def get_tier(self, chunk_id: int) -> str:
        """
        Return the tier label for *chunk_id*.

        Returns:
            ``'hot'`` if in T1 or T2, ``'warm'`` if in a ghost list (B1/B2),
            ``'cold'`` otherwise (meaning the chunk may still exist in the
            store but has no recent cache history).
        """
        if chunk_id in self.cache.t1 or chunk_id in self.cache.t2:
            return "hot"
        if chunk_id in self.cache.b1 or chunk_id in self.cache.b2:
            return "warm"
        return "cold"

    def prefetch(self, chunk_ids: list[int]) -> None:
        """
        Pre-warm the cache with the given chunk IDs.

        Skips IDs that are already in the hot cache.  Fetches each remaining
        chunk from the store and inserts it.

        Args:
            chunk_ids: List of chunk IDs to pre-fetch.
        """
        for chunk_id in chunk_ids:
            if not self.cache.contains(chunk_id):
                chunk = self.store.get_chunk(chunk_id)
                if chunk is not None:
                    self.cache.put(chunk_id, chunk)

    def stats(self) -> dict:
        """
        Return tier statistics.

        Returns a dict with::

            {
                "hot_count":   int,   # items in T1 + T2
                "warm_count":  int,   # ghost entries in B1 + B2
                "cold_count":  int,   # total chunks - hot - warm
                "total_chunks": int,  # from store
                "cache": dict,        # full ARCCache.stats() dict
            }
        """
        cache_stats = self.cache.stats()
        hot_count = cache_stats["t1_size"] + cache_stats["t2_size"]
        warm_count = cache_stats["b1_size"] + cache_stats["b2_size"]

        try:
            total_chunks = self.store.count_chunks()
        except Exception:
            total_chunks = 0

        cold_count = max(0, total_chunks - hot_count)

        return {
            "hot_count": hot_count,
            "warm_count": warm_count,
            "cold_count": cold_count,
            "total_chunks": total_chunks,
            "cache": cache_stats,
        }
