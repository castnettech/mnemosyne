# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Usage analytics and decay-weighted scoring for Mnemosyne.

Tracks ``UsageEvent`` records and derives exponentially-decayed frequency
scores that feed back into the retrieval ranking pipeline.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mnemosyne.models import UsageEvent

if TYPE_CHECKING:
    from mnemosyne.store import Store


class Analytics:
    """
    Session-aware usage tracker with exponential time-decay scoring.

    Decay formula (half-life model)::

        score(event) = 2 ^ (-age_days / halflife)

    For each chunk the per-event contributions of ``'selected'`` and
    ``'used'`` event types are summed.  ``'retrieved'`` and ``'discarded'``
    events are stored but do not contribute to the usage score.

    Args:
        store:  The persistent :class:`~mnemosyne.store.Store` instance.
        config: Mnemosyne :class:`~mnemosyne.config.Config` instance.
                Reads ``config.analytics.decay_halflife_days``.
    """

    def __init__(self, store: "Store", config) -> None:
        self.store = store
        self.halflife: float = float(config.analytics.decay_halflife_days)
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, session_id: str | None = None) -> str:
        """
        Start or resume a usage-tracking session.

        Args:
            session_id: Explicit session identifier.  A new 8-hex-char UUID
                        fragment is generated when this is ``None``.

        Returns:
            The active session ID string.
        """
        self._session_id = session_id or self._generate_session_id()
        return self._session_id

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record(
        self,
        chunk_id: int,
        event_type: str,
        query_text: str | None = None,
    ) -> None:
        """
        Record a usage event for *chunk_id*.

        Args:
            chunk_id:    The chunk that was interacted with.
            event_type:  One of ``'retrieved'``, ``'selected'``, ``'used'``,
                         ``'discarded'``.
            query_text:  The raw query string (may be None).
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        event = UsageEvent(
            event_id=None,
            chunk_id=chunk_id,
            query_text=query_text,
            session_id=self._session_id,
            event_type=event_type,
            timestamp=now_iso,
        )
        self.store.save_usage_event(event)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_usage_scores(self) -> dict[int, float]:
        """
        Compute exponentially-decayed usage scores for all chunks.

        Only ``'selected'`` and ``'used'`` events contribute to the score.
        The contribution of each event decays with time::

            contribution = 2 ^ (-age_days / halflife)

        Returns:
            Mapping of ``chunk_id -> score``.  Chunks with no qualifying
            events are absent from the dict.
        """
        now_utc = datetime.now(timezone.utc)
        # Fetch all 'selected' and 'used' events from the store
        events = self.store.get_usage_events(event_types=["selected", "used"])

        scores: dict[int, float] = {}
        for event in events:
            if not event.timestamp:
                continue
            try:
                # Parse ISO-8601 timestamps (handle both Z and +00:00 suffixes)
                ts_str = event.timestamp.replace("Z", "+00:00")
                event_time = datetime.fromisoformat(ts_str)
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                continue

            age_days = (now_utc - event_time).total_seconds() / 86400.0
            contribution = math.pow(2.0, -age_days / max(1e-9, self.halflife))
            scores[event.chunk_id] = scores.get(event.chunk_id, 0.0) + contribution

        return scores

    # ------------------------------------------------------------------
    # Co-occurrence analysis
    # ------------------------------------------------------------------

    def get_co_occurrence(self, chunk_ids: list[int]) -> dict[int, int]:
        """
        Find chunks frequently co-retrieved with the given chunks.

        Looks up sessions in which any of the provided *chunk_ids* were
        retrieved, then counts how often other chunks appeared in those same
        sessions.

        Args:
            chunk_ids: Reference set of chunk IDs.

        Returns:
            Mapping of ``co_chunk_id -> session_co_occurrence_count`` for
            all chunks that appeared alongside the input set (excluding the
            input IDs themselves).
        """
        if not chunk_ids:
            return {}

        # Get sessions that involved any of our reference chunks
        reference_sessions: set[str] = set()
        for cid in chunk_ids:
            events = self.store.get_usage_events_for_chunk(cid)
            for event in events:
                if event.session_id:
                    reference_sessions.add(event.session_id)

        if not reference_sessions:
            return {}

        # Count co-occurrences within those sessions
        co_counts: dict[int, int] = {}
        reference_set = set(chunk_ids)
        for session_id in reference_sessions:
            session_events = self.store.get_usage_events_for_session(session_id)
            for event in session_events:
                if event.chunk_id not in reference_set:
                    co_counts[event.chunk_id] = co_counts.get(event.chunk_id, 0) + 1

        return co_counts

    # ------------------------------------------------------------------
    # Precision / feedback analytics
    # ------------------------------------------------------------------

    def compute_precision_at_k(self, session_id: str | None = None) -> dict:
        """
        Compute precision from feedback events.

        Precision is defined as ``used / (used + discarded)``.  When both
        counts are zero the precision is reported as ``0.0``.

        Args:
            session_id: If provided, only events for this session are
                        considered.  ``None`` aggregates across all sessions.

        Returns:
            Dict with keys ``precision``, ``total_retrieved``,
            ``total_used``, ``total_discarded``, ``total_selected``.
        """
        if session_id is not None:
            events = self.store.get_usage_events_for_session(session_id)
        else:
            events = self.store.get_usage_events()

        counts: dict[str, int] = {
            "retrieved": 0,
            "used": 0,
            "discarded": 0,
            "selected": 0,
        }
        for event in events:
            if event.event_type in counts:
                counts[event.event_type] += 1

        denominator = counts["used"] + counts["discarded"]
        precision = counts["used"] / denominator if denominator > 0 else 0.0

        return {
            "precision": precision,
            "total_retrieved": counts["retrieved"],
            "total_used": counts["used"],
            "total_discarded": counts["discarded"],
            "total_selected": counts["selected"],
        }

    def get_top_used_chunks(self, limit: int = 5) -> list[dict]:
        """
        Return the *limit* most-used chunks ranked by ``'used'`` event count.

        Each entry is a dict with ``chunk_id``, ``use_count``, ``file_path``,
        ``symbol_name``, ``line_start``, and ``line_end``.
        """
        events = self.store.get_usage_events(event_types=["used"])

        # Tally per chunk_id
        chunk_counts: dict[int, int] = {}
        for event in events:
            chunk_counts[event.chunk_id] = chunk_counts.get(event.chunk_id, 0) + 1

        # Sort descending by count
        ranked = sorted(chunk_counts.items(), key=lambda x: -x[1])[:limit]

        results: list[dict] = []
        for chunk_id, count in ranked:
            chunk = self.store.get_chunk(chunk_id)
            file_path = ""
            symbol_name = None
            line_start = 0
            line_end = 0
            if chunk is not None:
                line_start = chunk.line_start
                line_end = chunk.line_end
                symbol_name = chunk.symbol_name
                file_rec = self.store.get_file_by_id(chunk.file_id)
                if file_rec is not None:
                    file_path = file_rec.rel_path
            results.append({
                "chunk_id": chunk_id,
                "use_count": count,
                "file_path": file_path,
                "symbol_name": symbol_name,
                "line_start": line_start,
                "line_end": line_end,
            })
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_session_id(self) -> str:
        return str(uuid.uuid4())[:8]
