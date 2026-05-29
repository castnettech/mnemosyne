# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Semantic sentence-embedding backend (BGE-small-en-v1.5, quantized ONNX).

This backend produces 384-dim semantic embeddings that discriminate meaning,
not just surface tokens, so a query like "what animal do I like" can retrieve a
stored "I like dogs" even with no lexical overlap.  It runs the quantized ONNX
model 100% locally on CPU via onnxruntime; there is no network call at query
time.  It is a strict superset of the lighter hashed and TF-IDF lanes and is
opt-in (it needs the ``[dense]`` extra and a one-time model download).

Why a separate module from ``dense_backend``:
    ``dense_backend.DenseBackend`` is wired to a specific MiniLM artifact, its
    own download layout, and float32 storage.  This backend targets a different
    model with different sourcing (project-owned, pinned, SHA-verified),
    query-vs-passage prompting, a shared cross-project cache, an offline
    pre-place path, and int8 storage.  Keeping it separate avoids disturbing
    the existing, proven doc path.

Model artifact (sourcing contract)
-----------------------------------
The model is NOT bundled in this repository and is NOT downloaded from a
personal account.  A maintainer converts + quantizes the official BAAI weights
ONCE (see ``tools/convert_bge_small_onnx.py``), hosts the result at a
PROJECT-OWNED location, and pins it here:

    * MODEL_REPO / MODEL_BASE_URL -- where the artifact lives.  A placeholder
      default ships; set the real value on publish (or override via config /
      the ``MNEMOSYNE_SEMANTIC_MODEL_REPO`` env var / ``base_url`` argument).
    * MODEL_REVISION -- pin to a SPECIFIC immutable commit hash on publish.
    * MODEL_SHA256 / TOKENIZER_SHA256 -- the exact hashes the conversion tool
      printed.  When set, downloads are verified and REJECTED + deleted on
      mismatch.  When None, a LOUD warning is emitted (unpinned / unverified).
    * MODEL_FILENAME / TOKENIZER_FILENAME -- the quantized onnx + tokenizer.

Security model
--------------
- Local-only inference (CPUExecutionProvider, single-threaded, no telemetry).
- One-time download over stdlib HTTPS (urllib) -- no huggingface_hub, no extra
  runtime dependency.  Downloaded files are SHA-256 verified (when pinned) and
  chmod 0600.  On a hash mismatch the partial file is removed and the load
  fails closed.
- Offline supported: point ``local_path`` at a directory holding pre-placed
  ``model_quantized.onnx`` + ``tokenizer.json`` and the network is never
  touched.  An already-cached file is likewise reused without a download.
- Graceful degradation: if onnxruntime or numpy is missing, the model files
  are unavailable, or the text is empty, every embed call returns ``None`` and
  the caller simply skips this lane.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from mnemosyne.embeddings.dense_backend import (
    _WordPieceTokenizer,
    _download_file,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mnemosyne.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model constants -- the maintainer sets these on artifact publish.
# ---------------------------------------------------------------------------

#: Project-owned artifact location.  This is a DOCUMENTED PLACEHOLDER -- the
#: maintainer replaces it (or overrides it at runtime) with the real host they
#: publish the converted artifact to.  It MUST NOT point at a personal account.
#: Override precedence: ``base_url`` arg > config ``semantic_model_repo`` >
#: ``MNEMOSYNE_SEMANTIC_MODEL_REPO`` env var > this default.
MODEL_REPO: str = "https://models.example-project.invalid/bge-small-en-v1.5"

#: Alias kept for callers/readers who think in "base URL" terms.
MODEL_BASE_URL: str = MODEL_REPO

#: Pin to a SPECIFIC immutable revision/commit when the artifact is published.
#: A revision path segment is only appended when this is set.
MODEL_REVISION: str | None = None  # set on artifact publish

#: The quantized ONNX model + the HuggingFace tokenizer.json filenames.
MODEL_FILENAME: str = "model_quantized.onnx"
TOKENIZER_FILENAME: str = "tokenizer.json"

#: Expected SHA-256 of each artifact.  SET THESE on publish to the values the
#: conversion tool prints.  When None the download is UNVERIFIED -- a loud
#: warning fires; do not ship a public release with these unset.
MODEL_SHA256: str | None = None      # set on artifact publish
TOKENIZER_SHA256: str | None = None  # set on artifact publish

#: Output dimensionality and max sequence length for bge-small-en-v1.5.
MODEL_DIM: int = 384
MAX_SEQ_LEN: int = 512

#: bge retrieval models expect this instruction prefixed to the QUERY ONLY.
#: Stored passages are embedded with no prefix.  This is the official BAAI
#: retrieval instruction for the v1.5 small/base/large English models.
QUERY_INSTRUCTION: str = (
    "Represent this sentence for searching relevant passages:"
)

#: Environment variable that overrides MODEL_REPO at runtime.
ENV_MODEL_REPO: str = "MNEMOSYNE_SEMANTIC_MODEL_REPO"
#: Environment variable that overrides the cache directory at runtime.
ENV_CACHE_DIR: str = "MNEMOSYNE_MODEL_CACHE"


# ---------------------------------------------------------------------------
# Cache directory resolution (XDG-aware, shared across projects)
# ---------------------------------------------------------------------------


def default_cache_dir() -> str:
    """Return the shared model cache dir, honoring XDG_CACHE_HOME.

    Defaults to ``~/.cache/mnemosyne/models`` (or ``$XDG_CACHE_HOME/mnemosyne/
    models``).  A single shared cache means the (~10 MB) artifact is downloaded
    once per machine and reused by every project on it.
    """
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return override
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return str(base / "mnemosyne" / "models")


def _resolve_base_url(config: "Config | None", base_url: str | None) -> str:
    """Resolve the artifact base URL by precedence (arg > config > env > default)."""
    if base_url:
        return base_url
    if config is not None:
        cfg_repo = getattr(
            getattr(config, "embedding", None), "semantic_model_repo", None
        )
        if cfg_repo:
            return cfg_repo
    env_repo = os.environ.get(ENV_MODEL_REPO)
    if env_repo:
        return env_repo
    return MODEL_REPO


def _artifact_url(base_url: str, filename: str) -> str:
    """Join base URL + optional pinned revision + filename into a download URL."""
    parts = [base_url.rstrip("/")]
    if MODEL_REVISION:
        parts.append(MODEL_REVISION)
    parts.append(filename)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Semantic backend
# ---------------------------------------------------------------------------


class SemanticBackend:
    """BGE-small-en-v1.5 semantic embedding backend (quantized ONNX, CPU-only).

    Args:
        config: Mnemosyne :class:`~mnemosyne.config.Config` instance (optional;
            used only to read ``embedding.semantic_model_repo`` /
            ``embedding.semantic_local_path`` overrides).
        cache_dir: Directory to cache downloaded model files.  Defaults to the
            shared XDG cache (:func:`default_cache_dir`).
        local_path: Offline directory holding pre-placed ``model_quantized.onnx``
            + ``tokenizer.json``.  When set (or readable from config), the
            network is NEVER contacted.
        base_url: Override the artifact base URL directly (highest precedence).
    """

    def __init__(
        self,
        config: "Config | None" = None,
        *,
        cache_dir: str | None = None,
        local_path: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir or default_cache_dir()
        # local_path precedence: explicit arg > config.embedding.semantic_local_path
        cfg_local = None
        if config is not None:
            cfg_local = getattr(
                getattr(config, "embedding", None), "semantic_local_path", None
            )
        self._local_path = local_path or cfg_local
        self._base_url = _resolve_base_url(config, base_url)
        self._session = None  # lazy onnxruntime.InferenceSession
        self._tokenizer: _WordPieceTokenizer | None = None  # lazy

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> tuple[str, str]:
        """Return (model_path, tokenizer_path) for the active source dir."""
        source_dir = self._local_path or self._cache_dir
        return (
            os.path.join(source_dir, MODEL_FILENAME),
            os.path.join(source_dir, TOKENIZER_FILENAME),
        )

    def _ensure_model(self) -> None:
        """Load the ONNX session, downloading + verifying the artifact if needed.

        Raises ``ImportError`` if onnxruntime is absent (caller treats as a
        graceful skip).  Raises other exceptions on a hard failure (missing
        offline files, SHA mismatch, corrupt model); :meth:`embed` catches them.
        """
        if self._session is not None:
            return

        import onnxruntime  # lazy import -- graceful if not installed

        model_path, tokenizer_path = self._resolve_paths()

        if self._local_path:
            # Offline mode: files MUST already be present.  Never touch network.
            if not os.path.isfile(model_path) or not os.path.isfile(
                tokenizer_path
            ):
                raise FileNotFoundError(
                    "semantic model local_path is set but "
                    f"{MODEL_FILENAME} / {TOKENIZER_FILENAME} were not found "
                    f"in {self._local_path!r}"
                )
        else:
            self._maybe_warn_unverified()
            if not os.path.isfile(model_path):
                url = _artifact_url(self._base_url, MODEL_FILENAME)
                logger.info("Downloading semantic model from %s", url)
                _download_file(url, model_path, MODEL_SHA256)
            if not os.path.isfile(tokenizer_path):
                url = _artifact_url(self._base_url, TOKENIZER_FILENAME)
                logger.info("Downloading tokenizer from %s", url)
                _download_file(url, tokenizer_path, TOKENIZER_SHA256)

        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3  # suppress warnings
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        self._session = onnxruntime.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = _WordPieceTokenizer(tokenizer_path)

    @staticmethod
    def _maybe_warn_unverified() -> None:
        """Emit a loud warning when SHA-256 pins are unset before a download."""
        missing = []
        if MODEL_SHA256 is None:
            missing.append("MODEL_SHA256")
        if TOKENIZER_SHA256 is None:
            missing.append("TOKENIZER_SHA256")
        if missing:
            logger.warning(
                "SECURITY: semantic model is UNPINNED/UNVERIFIED -- %s are not "
                "set, so the downloaded artifact's integrity is NOT checked. "
                "Set the SHA-256 pins (and a pinned MODEL_REVISION) before any "
                "production use, or pre-place a trusted artifact via local_path.",
                " and ".join(missing),
            )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str, *, is_query: bool = False) -> list[float] | None:
        """Embed ``text`` into a 384-dim L2-normalized vector, or ``None``.

        Args:
            text: The text to embed.
            is_query: When True, the bge query instruction is prefixed (use this
                for SEARCH queries).  Stored passages MUST use ``False`` so they
                are not prefixed -- mixing the two degrades retrieval.

        Returns ``None`` when onnxruntime/numpy is unavailable, the model cannot
        be loaded, or ``text`` is empty -- the caller then skips this lane.
        """
        if not text or not text.strip():
            return None

        try:
            self._ensure_model()
        except ImportError:
            return None
        except Exception:
            logger.warning("Failed to load semantic model", exc_info=True)
            return None

        try:
            import numpy as np
        except ImportError:
            return None

        assert self._tokenizer is not None
        assert self._session is not None

        prompt = (
            f"{QUERY_INSTRUCTION} {text.strip()}" if is_query else text.strip()
        )
        input_ids, attention_mask = self._tokenizer.tokenize(
            prompt, max_length=MAX_SEQ_LEN
        )

        ids_arr = np.array([input_ids], dtype=np.int64)
        mask_arr = np.array([attention_mask], dtype=np.int64)
        type_arr = np.zeros_like(ids_arr)

        feeds = {
            "input_ids": ids_arr,
            "attention_mask": mask_arr,
            "token_type_ids": type_arr,
        }
        input_names = {inp.name for inp in self._session.get_inputs()}
        feeds = {k: v for k, v in feeds.items() if k in input_names}

        outputs = self._session.run(None, feeds)

        # outputs[0]: (1, seq_len, hidden_dim).  bge uses the [CLS] token as the
        # sentence embedding, but mean-pooling over non-pad tokens is robust
        # across export variants and matches the existing dense path; both are
        # L2-normalized so cosine ordering is what matters.
        token_embeddings = outputs[0][0]  # (seq_len, hidden_dim)
        mask = np.array(attention_mask, dtype=np.float32)

        masked = token_embeddings * mask[:, np.newaxis]
        summed = masked.sum(axis=0)
        count = mask.sum()
        if count == 0:
            return None
        mean_vec = summed / count

        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        return mean_vec.astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float] | None:
        """Embed ``text`` as a SEARCH QUERY (bge instruction prefixed)."""
        return self.embed(text, is_query=True)

    def embed_passage(self, text: str) -> list[float] | None:
        """Embed ``text`` as a STORED PASSAGE (no prefix)."""
        return self.embed(text, is_query=False)

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def embed_to_int8_bytes(
        self, text: str, *, is_query: bool = False
    ) -> bytes | None:
        """Embed and pack as ``MODEL_DIM`` int8 bytes for compact BLOB storage.

        The float vector is L2-normalized, so scaling by 127 and clamping to
        ``[-127, 127]`` keeps quantization error small.  Pairs with the
        ``"int8"`` encoding of :func:`mnemosyne.embeddings.vector_search`.
        """
        vec = self.embed(text, is_query=is_query)
        if vec is None:
            return None
        q = [max(-127, min(127, int(round(x * 127)))) for x in vec]
        return struct.pack(f"{len(q)}b", *q)

    def embed_to_float32_bytes(
        self, text: str, *, is_query: bool = False
    ) -> bytes | None:
        """Embed and pack as float32 little-endian bytes (full precision)."""
        vec = self.embed(text, is_query=is_query)
        if vec is None:
            return None
        return struct.pack(f"<{len(vec)}f", *vec)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if onnxruntime (and numpy) can be imported."""
        try:
            import numpy  # noqa: F401
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False


__all__ = [
    "SemanticBackend",
    "MODEL_REPO",
    "MODEL_BASE_URL",
    "MODEL_REVISION",
    "MODEL_FILENAME",
    "TOKENIZER_FILENAME",
    "MODEL_SHA256",
    "TOKENIZER_SHA256",
    "MODEL_DIM",
    "MAX_SEQ_LEN",
    "QUERY_INSTRUCTION",
    "default_cache_dir",
]
