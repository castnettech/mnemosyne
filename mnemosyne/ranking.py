# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Ranking utilities for the Mnemosyne retrieval pipeline.

Provides:
- ``rrf_fuse``         -- Reciprocal Rank Fusion across multiple scored lists
- ``cost_model_score`` -- Value density: relevance per token
- ``budget_cut``       -- Greedy token-budget selection with optional compression
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.models import QueryResult


def rrf_fuse(
    score_lists: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
    k: int = 60,
) -> list[tuple[int, float, dict]]:
    """
    Reciprocal Rank Fusion across multiple ranked lists.

    For each source list, chunks are sorted by score descending and assigned
    1-based ranks.  A chunk absent from a list receives a penalty rank of
    ``len(list) + 1``.  The combined RRF score is::

        rrf_score(id) = sum(weight[src] / (k + rank[src](id)))

    Args:
        score_lists: Mapping of source name -> list of ``(chunk_id, score)``
                     pairs (any order; will be sorted internally).
        weights:     Per-source weighting factors.  Missing sources default
                     to weight 1.0.
        k:           RRF smoothing constant (default 60).

    Returns:
        List of ``(chunk_id, rrf_score, source_scores_dict)`` tuples sorted
        by ``rrf_score`` descending.
    """
    # Build rank maps for each source
    rank_maps: dict[str, dict[int, int]] = {}
    list_lengths: dict[str, int] = {}

    for source, pairs in score_lists.items():
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        list_lengths[source] = len(sorted_pairs)
        rank_maps[source] = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sorted_pairs)}

    # Collect all unique chunk ids
    all_ids: set[int] = set()
    for pairs in score_lists.values():
        for chunk_id, _ in pairs:
            all_ids.add(chunk_id)

    # Build raw score lookup for reporting
    raw_scores: dict[str, dict[int, float]] = {}
    for source, pairs in score_lists.items():
        raw_scores[source] = {chunk_id: score for chunk_id, score in pairs}

    results: list[tuple[int, float, dict]] = []
    for chunk_id in all_ids:
        rrf_score = 0.0
        source_scores: dict[str, float] = {}
        for source in score_lists:
            w = weights.get(source, 1.0)
            rank = rank_maps[source].get(chunk_id, list_lengths[source] + 1)
            contribution = w / (k + rank)
            rrf_score += contribution
            source_scores[source] = raw_scores[source].get(chunk_id, 0.0)
        source_scores["rrf"] = rrf_score
        results.append((chunk_id, rrf_score, source_scores))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def cost_model_score(
    chunk_id: int,
    rrf_score: float,
    token_count: int,
    boilerplate_ratio: float = 0.0,
    is_structured_code: bool = False,
) -> float:
    """
    Compute value density as relevance per token (dampened), with
    boilerplate penalty and structured-code boost.

    Uses a logarithmic denominator so that chunk size influences the
    ranking gently.  Chunks with high boilerplate content are penalised.
    Chunks that the language-aware chunker identified as named functions
    or classes get a 1.4x boost, since they carry concrete code semantics
    that generic/HTML chunks lack.

    Args:
        chunk_id:            Chunk identifier (unused; for call-site symmetry).
        rrf_score:           The fused RRF relevance score.
        token_count:         Approximate token count of the chunk.
        boilerplate_ratio:   Fraction of lines detected as boilerplate [0, 1].
        is_structured_code:  True if chunk has a symbol_name or is typed as
                             function/class (not generic/block).

    Returns:
        Density score.
    """
    import math
    relevance = rrf_score * (1.0 - 0.5 * boilerplate_ratio)
    if is_structured_code:
        relevance *= 2.0
    return relevance / (1.0 + math.log1p(max(1, token_count)))


def budget_cut(
    candidates: list[tuple],
    budget: int,
    compressor=None,
    store=None,
) -> list:
    """
    Greedy selection of chunks within a token budget.

    Candidates are first sorted by value density (``cost_model_score``
    descending).  Chunks are added greedily while the remaining budget
    permits.  If a chunk alone exceeds the remaining budget and a
    *compressor* is provided, a compressed version is tried.

    Args:
        candidates:  List of ``(chunk_id, rrf_score, source_scores)`` tuples.
                     Requires *store* to look up actual ``Chunk`` objects.
        budget:      Maximum total tokens to return.
        compressor:  Optional compressor with a ``compress(text) -> str``
                     method.
        store:       Store instance with a ``get_chunk(chunk_id) -> Chunk``
                     method.  Required when candidates is non-empty.

    Returns:
        List of ``Chunk`` objects selected within the budget, preserving
        value-density order.  Returns empty list when store is None or
        no candidates fit.
    """
    if not candidates or store is None:
        return []

    # Fetch chunks -- preserve input order from _cost_model_rank which
    # already applied boilerplate penalties and structured-code boosts.
    # Assign positional density (higher = earlier in pre-ranked list).
    enriched: list[tuple[float, int, object, dict]] = []
    for i, (chunk_id, rrf_score, source_scores) in enumerate(candidates):
        chunk = store.get_chunk(chunk_id)
        if chunk is None:
            continue
        # Use inverse position as density so the pre-ranked order is preserved
        density = 1.0 / (1 + i)
        enriched.append((density, chunk_id, chunk, source_scores))

    selected = []
    remaining = budget
    from mnemosyne.models import estimate_tokens

    for density, chunk_id, chunk, source_scores in enriched:
        if remaining <= 0:
            break

        effective_content = chunk.compressed or chunk.content
        tokens = estimate_tokens(effective_content)

        if tokens <= remaining:
            selected.append((chunk, source_scores))
            remaining -= tokens
        elif compressor is not None:
            # Try compressing the chunk to fit within remaining budget
            try:
                # Use strict mode for symbol chunks (named functions/classes)
                # to skip TF-IDF thinning on semantically dense code.
                use_strict = bool(getattr(chunk, "symbol_name", None))
                compressed_text = compressor.compress(chunk, strict=use_strict)
                compressed_tokens = estimate_tokens(compressed_text)
                if compressed_tokens <= remaining:
                    # Return a modified chunk with compressed content applied
                    from dataclasses import replace
                    compressed_chunk = replace(
                        chunk,
                        compressed=compressed_text,
                        compression_ratio=(
                            len(compressed_text) / max(1, len(chunk.content))
                        ),
                    )
                    selected.append((compressed_chunk, source_scores))
                    remaining -= compressed_tokens
            except Exception:
                pass  # Compression failed; skip this chunk

    return selected
