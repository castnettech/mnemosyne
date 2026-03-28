# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Output formatters for Mnemosyne query results.

Provides plain-text and JSON serialisation of ``QueryResult`` lists,
suitable for both human consumption (CLI) and machine consumption (agent
pipelines).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mnemosyne.models import estimate_tokens

if TYPE_CHECKING:
    from mnemosyne.models import QueryResult


class Formatter:
    """
    Static collection of output-format methods.

    All methods are ``@staticmethod`` so the class can be used without
    instantiation (``Formatter.format_plain(...)``).
    """

    @staticmethod
    def format_plain(
        results: list["QueryResult"],
        query: str,
        budget: int,
        session_id: str | None,
    ) -> str:
        """
        Format *results* as human-readable plain text.

        Layout::

            ## Context for: "<query>"
            ## N chunks, X,XXX tokens (budget: Y,YYY)[, session: sid]

            ### file: path/to/file.py (lines S-E) [SymbolName] [score: 0.NNN]
            <chunk content or delta diff>

            ...

        For delta chunks the diff text is emitted instead of the full content.
        Compressed content is preferred over raw when available.

        Args:
            results:    Ordered list of :class:`~mnemosyne.models.QueryResult`.
            query:      The original query string.
            budget:     The token budget that was applied.
            session_id: Active session ID, or ``None``.

        Returns:
            A UTF-8 plain-text string.
        """
        tokens_used = sum(
            estimate_tokens(r.delta_text if r.is_delta else (r.chunk.compressed or r.chunk.content))
            for r in results
        )
        header_parts = [f"## {len(results)} chunks, {tokens_used:,} tokens (budget: {budget:,})"]
        if session_id:
            header_parts.append(f"session: {session_id}")

        lines: list[str] = [
            f'## Context for: "{query}"',
            ", ".join(header_parts),
            "",
        ]

        for r in results:
            content = (
                r.delta_text
                if r.is_delta
                else (r.chunk.compressed or r.chunk.content)
            )
            rrf = r.scores.get("rrf", 0.0)

            # Build chunk header
            loc = f"lines {r.chunk.line_start}-{r.chunk.line_end}"
            symbol = f" [{r.chunk.symbol_name}]" if r.chunk.symbol_name else ""
            delta_marker = " [DELTA]" if r.is_delta else ""
            stale_marker = " [STALE]" if r.is_stale else ""
            header = (
                f"### file: {r.file_path} ({loc}){symbol}{delta_marker}{stale_marker}"
                f" [score: {rrf:.3f}]"
            )
            lines.append(header)
            lines.append(content or "")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_json(
        results: list["QueryResult"],
        query: str,
        budget: int,
        session_id: str | None,
    ) -> str:
        """
        Format *results* as a JSON object.

        Schema::

            {
              "query":       str,
              "session_id":  str | null,
              "budget":      int,
              "tokens_used": int,
              "results": [
                {
                  "chunk_id":  int,
                  "file":      str,
                  "lines":     [start, end],
                  "symbol":    str | null,
                  "type":      str,
                  "scores":    {source: float, ...},
                  "content":   str,
                  "is_delta":  bool,
                  "tokens":    int
                },
                ...
              ]
            }

        Args:
            results:    Ordered list of :class:`~mnemosyne.models.QueryResult`.
            query:      The original query string.
            budget:     The token budget that was applied.
            session_id: Active session ID, or ``None``.

        Returns:
            A pretty-printed JSON string (2-space indent).
        """
        tokens_used = sum(
            estimate_tokens(r.delta_text if r.is_delta else (r.chunk.compressed or r.chunk.content))
            for r in results
        )

        result_items = []
        for r in results:
            content = (
                r.delta_text
                if r.is_delta
                else (r.chunk.compressed or r.chunk.content)
            )
            result_items.append({
                "chunk_id": r.chunk.chunk_id,
                "file": r.file_path,
                "lines": [r.chunk.line_start, r.chunk.line_end],
                "symbol": r.chunk.symbol_name,
                "type": r.chunk.chunk_type,
                "scores": r.scores,
                "content": content or "",
                "is_delta": r.is_delta,
                "is_stale": r.is_stale,
                "stale_reason": r.stale_reason,
                "tokens": estimate_tokens(r.chunk.compressed or r.chunk.content),
            })

        output = {
            "query": query,
            "session_id": session_id,
            "budget": budget,
            "tokens_used": tokens_used,
            "results": result_items,
        }

        return json.dumps(output, indent=2, ensure_ascii=False)
