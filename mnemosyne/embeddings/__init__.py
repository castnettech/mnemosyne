# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Embedding backend registry for Mnemosyne.

Selects and returns the best available embedding backend based on configuration.
The only built-in backend is :class:`~mnemosyne.embeddings.tfidf_backend.TFIDFBackend`,
which requires no external dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemosyne.embeddings.tfidf_backend import TFIDFBackend

if TYPE_CHECKING:
    from mnemosyne.config import Config


def get_backend(config: "Config", store=None) -> TFIDFBackend:
    """
    Return the TF-IDF embedding backend.

    Args:
        config: Mnemosyne :class:`~mnemosyne.config.Config` instance.
        store:  Optional persistent store for vocabulary persistence.

    Returns:
        A configured TFIDFBackend instance.
    """
    return TFIDFBackend(config, store)


__all__ = ["TFIDFBackend", "get_backend"]
