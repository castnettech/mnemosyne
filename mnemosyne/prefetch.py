# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Predictive pre-fetching for Mnemosyne.

Learns query-to-chunk patterns from historical retrievals and uses them to
pre-warm the cache before the retrieval engine runs.

Pattern identity is based on a normalized, sorted, deduplicated term set
hashed to a 16-character hex fingerprint.  Patterns that have been seen at
least ``min_hits`` times are eligible for pre-fetching.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.store import Store


class Prefetcher:
    """
    Query-pattern-based chunk pre-fetcher.

    Maintains a ``task_patterns`` table (via the store) that maps query
    signatures to the chunk IDs most frequently selected for that pattern.
    On each new query the signature is looked up; if the pattern has enough
    hit history the associated chunks are returned for cache warming.

    Args:
        store:     The persistent :class:`~mnemosyne.store.Store` instance.
        min_hits:  Minimum number of times a pattern must have been observed
                   before its pre-fetch candidates are returned.  This
                   prevents cold-start noise from polluting the cache.
    """

    def __init__(self, store: "Store", min_hits: int = 3) -> None:
        self.store = store
        self.min_hits = min_hits

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query_signature(self, query_text: str) -> str:
        """
        Compute a stable 16-char hex fingerprint for *query_text*.

        Normalisation steps:

        1. Lowercase the text.
        2. Extract all alphanumeric tokens of length >= 3 (filters
           noise words, punctuation, single-char operators).
        3. Deduplicate and sort the token set.
        4. MD5-hash the joined token string.

        Args:
            query_text: Raw query string from the user.

        Returns:
            16-character lowercase hex string.
        """
        terms = sorted(set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query_text.lower())))
        fingerprint = hashlib.md5(" ".join(terms).encode()).hexdigest()[:16]
        return fingerprint

    def get_prefetch_ids(self, query_text: str) -> set[int]:
        """
        Return chunk IDs to pre-fetch for the given query.

        Looks up the query's signature in ``task_patterns``.  Returns the
        associated chunk IDs only when the pattern has been seen at least
        ``min_hits`` times.

        Args:
            query_text: The query string to look up.

        Returns:
            Set of chunk IDs to pre-fetch (may be empty).
        """
        sig = self.query_signature(query_text)
        pattern = self.store.get_pattern(sig)
        if pattern is not None:
            chunk_ids, hit_count = pattern
            if hit_count >= self.min_hits:
                return set(chunk_ids)
        return set()

    def record_pattern(self, query_text: str, selected_ids: list[int]) -> None:
        """
        Record which chunks were selected for this query type.

        Merges *selected_ids* with any existing chunk-id list for the
        pattern and increments the hit counter.

        Args:
            query_text:   The query string that produced these results.
            selected_ids: Chunk IDs that were included in the final result set.
        """
        if not selected_ids:
            return

        sig = self.query_signature(query_text)
        existing = self.store.get_pattern(sig)

        if existing is not None:
            existing_ids, hit_count = existing
            merged = list(set(existing_ids) | set(selected_ids))
            self.store.upsert_pattern(sig, merged, hit_count + 1)
        else:
            self.store.upsert_pattern(sig, list(selected_ids), 1)
