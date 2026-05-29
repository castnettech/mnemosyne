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


def get_semantic_backend(
    config: "Config",
    *,
    cache_dir: str | None = None,
    local_path: str | None = None,
    base_url: str | None = None,
):
    """Return the BGE-small semantic embedding backend, or ``None``.

    Returns ``None`` when the semantic backend is not enabled in config
    (``embedding.semantic_model`` falsy) so the caller can fall back to the
    lighter lanes.  The instance itself degrades gracefully (``embed`` returns
    ``None`` when onnxruntime/numpy or the model artifact is missing), so it is
    safe to construct even when those are absent.

    Args:
        config:     Mnemosyne :class:`~mnemosyne.config.Config` instance.
        cache_dir:  Override the shared model cache directory.
        local_path: Offline directory of pre-placed model files (skips network).
        base_url:   Override the artifact base URL.

    Returns:
        A :class:`~mnemosyne.embeddings.semantic_backend.SemanticBackend`, or
        ``None`` if not enabled.
    """
    if not getattr(config.embedding, "semantic_model", None):
        return None
    from mnemosyne.embeddings.semantic_backend import SemanticBackend

    return SemanticBackend(
        config, cache_dir=cache_dir, local_path=local_path, base_url=base_url
    )


__all__ = [
    "TFIDFBackend",
    "get_backend",
    "get_dense_backend",
    "get_semantic_backend",
]
