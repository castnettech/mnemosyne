# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ARC (Adaptive Replacement Cache) for Mnemosyne.

Implements the full ARC algorithm with ghost lists (B1, B2) and adaptive
target-size parameter *p* that balances recency (T1) vs. frequency (T2).

Reference: Nimrod Megiddo and Dharmendra S. Modha, "ARC: A Self-Tuning,
Low Overhead Replacement Cache", USENIX FAST 2003.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.models import Chunk


class ARCCache:
    """
    Adaptive Replacement Cache with ghost lists.

    The cache is partitioned into four ordered dictionaries:

    * **T1** — recently-inserted items seen exactly once.
    * **T2** — frequently-accessed items (promoted from T1 or re-hit in T2).
    * **B1** — ghost keys evicted from T1 (no data stored).
    * **B2** — ghost keys evicted from T2 (no data stored).

    The adaptive parameter *p* (target T1 size) grows when a B1 hit occurs
    (recency is underrepresented) and shrinks when a B2 hit occurs (frequency
    is underrepresented).

    Args:
        capacity:       Maximum number of live chunks held in T1 + T2.
        ghost_capacity: Maximum number of ghost keys held in B1 + B2 combined.
    """

    def __init__(self, capacity: int = 500, ghost_capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.c: int = capacity
        self.p: int = 0
        self.ghost_cap: int = max(ghost_capacity, 1)

        self.t1: OrderedDict[int, Chunk] = OrderedDict()
        self.t2: OrderedDict[int, Chunk] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, chunk_id: int) -> "Chunk | None":
        """
        Look up *chunk_id* in the cache.

        On a hit the entry is promoted/refreshed according to ARC rules:

        * T1 hit → move to MRU end of T2 (first access seen twice = frequent).
        * T2 hit → move to MRU end of T2 (refresh LRU clock).

        Returns:
            The :class:`~mnemosyne.models.Chunk` if present, else ``None``.
        """
        if chunk_id in self.t1:
            # Promote from T1 to T2
            chunk = self.t1.pop(chunk_id)
            self.t2[chunk_id] = chunk
            self.t2.move_to_end(chunk_id)
            self._hits += 1
            return chunk

        if chunk_id in self.t2:
            # Refresh position in T2
            self.t2.move_to_end(chunk_id)
            self._hits += 1
            return self.t2[chunk_id]

        self._misses += 1
        return None

    def put(self, chunk_id: int, chunk: "Chunk") -> None:
        """
        Insert or update *chunk_id* in the cache.

        ARC insertion rules:

        * **B1 ghost hit** — increase *p* (recency should grow), insert into T2.
        * **B2 ghost hit** — decrease *p* (frequency should grow), insert into T2.
        * **Already in T1 or T2** — update value in-place, refresh T2 position.
        * **New entry** — insert into T1.

        Triggers :meth:`_evict` whenever ``len(T1) + len(T2) >= capacity``.
        """
        # Already live — just refresh
        if chunk_id in self.t1:
            self.t1[chunk_id] = chunk
            return
        if chunk_id in self.t2:
            self.t2[chunk_id] = chunk
            self.t2.move_to_end(chunk_id)
            return

        if chunk_id in self.b1:
            # Ghost hit in B1: recency was underrepresented → grow T1 target
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.c, self.p + delta)
            del self.b1[chunk_id]
            # Evict if necessary before inserting into T2
            if len(self.t1) + len(self.t2) >= self.c:
                self._evict()
            self.t2[chunk_id] = chunk
            self.t2.move_to_end(chunk_id)
            return

        if chunk_id in self.b2:
            # Ghost hit in B2: frequency was underrepresented → shrink T1 target
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            del self.b2[chunk_id]
            # Evict if necessary before inserting into T2
            if len(self.t1) + len(self.t2) >= self.c:
                self._evict()
            self.t2[chunk_id] = chunk
            self.t2.move_to_end(chunk_id)
            return

        # Brand new entry → T1
        if len(self.t1) + len(self.t2) >= self.c:
            self._evict()
        self.t1[chunk_id] = chunk
        self.t1.move_to_end(chunk_id)

    # ------------------------------------------------------------------
    # Internal eviction
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """
        Evict one item from T1 or T2 based on adaptive parameter *p*.

        * If ``len(T1) > p``: evict LRU of T1, promote its key to B1.
        * Else: evict LRU of T2, promote its key to B2.

        Ghost lists are capped at ``ghost_cap`` (combined).
        """
        if self.t1 and (len(self.t1) > self.p or not self.t2):
            # Evict LRU from T1
            evicted_id, _ = next(iter(self.t1.items()))
            del self.t1[evicted_id]
            self.b1[evicted_id] = None
            self.b1.move_to_end(evicted_id)
        elif self.t2:
            # Evict LRU from T2
            evicted_id, _ = next(iter(self.t2.items()))
            del self.t2[evicted_id]
            self.b2[evicted_id] = None
            self.b2.move_to_end(evicted_id)
        elif self.t1:
            # T2 is empty; must evict from T1
            evicted_id, _ = next(iter(self.t1.items()))
            del self.t1[evicted_id]
            self.b1[evicted_id] = None
            self.b1.move_to_end(evicted_id)

        # Cap ghost lists to avoid unbounded growth
        self._trim_ghost_lists()

    def _trim_ghost_lists(self) -> None:
        """Remove oldest ghost entries until combined B1+B2 <= ghost_cap."""
        while len(self.b1) + len(self.b2) > self.ghost_cap:
            # Evict oldest ghost from the larger ghost list
            if len(self.b1) >= len(self.b2) and self.b1:
                self.b1.popitem(last=False)
            elif self.b2:
                self.b2.popitem(last=False)
            else:
                break

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def contains(self, chunk_id: int) -> bool:
        """Return True if *chunk_id* is in T1 or T2 (live, not ghost)."""
        return chunk_id in self.t1 or chunk_id in self.t2

    def prefetch(self, chunk_ids: list[int], fetch_fn: Callable[[int], "Chunk | None"]) -> None:
        """
        Warm the cache with predicted chunks.

        For each *chunk_id* not already in the cache, calls *fetch_fn* to
        retrieve the ``Chunk`` and inserts it via :meth:`put`.

        Args:
            chunk_ids: List of chunk IDs to pre-warm.
            fetch_fn:  Callable that takes a chunk_id and returns a
                       ``Chunk`` or ``None``.
        """
        for chunk_id in chunk_ids:
            if not self.contains(chunk_id):
                chunk = fetch_fn(chunk_id)
                if chunk is not None:
                    self.put(chunk_id, chunk)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a float in [0.0, 1.0].  Returns 0.0 when no accesses."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def stats(self) -> dict:
        """
        Return a snapshot of cache internals.

        Returns:
            Dict with keys: ``t1_size``, ``t2_size``, ``b1_size``,
            ``b2_size``, ``capacity``, ``p``, ``hits``, ``misses``,
            ``hit_rate``.
        """
        return {
            "t1_size": len(self.t1),
            "t2_size": len(self.t2),
            "b1_size": len(self.b1),
            "b2_size": len(self.b2),
            "capacity": self.c,
            "p": self.p,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def clear(self) -> None:
        """Evict all live and ghost entries and reset statistics."""
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.p = 0
        self._hits = 0
        self._misses = 0
