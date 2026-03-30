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


def get_dense_backend(config: "Config", store=None, model_dir: str | None = None):
    """Return the dense embedding backend if configured and available.

    Args:
        config:    Mnemosyne :class:`~mnemosyne.config.Config` instance.
        store:     Optional persistent store for embedding CRUD.
        model_dir: Directory to cache model files.

    Returns:
        A :class:`~mnemosyne.embeddings.dense_backend.DenseBackend` instance,
        or ``None`` if the backend is not configured or onnxruntime is not
        installed.
    """
    dense_model = getattr(config.embedding, "dense_model", None)
    if not dense_model:
        return None
    try:
        from mnemosyne.embeddings.dense_backend import DenseBackend
        return DenseBackend(config, store, model_dir=model_dir)
    except ImportError:
        return None


__all__ = ["TFIDFBackend", "get_backend", "get_dense_backend"]
