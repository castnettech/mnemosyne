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
The model is NOT bundled in this repository.  It is the vetted, pre-built int8
ONNX export of BAAI/bge-small-en-v1.5 maintained by the reputable HF-staff
Xenova account -- a single quantized file, no local conversion step.  It is
pinned to an immutable commit and SHA-verified here:

    * MODEL_REPO / MODEL_BASE_URL -- the artifact repository base URL.  Defaults
      to ``https://huggingface.co/Xenova/bge-small-en-v1.5``.  Override via
      config ``semantic_model_repo`` / the ``MNEMOSYNE_SEMANTIC_MODEL_REPO`` env
      var / the ``base_url`` argument (a mirror must serve the same
      ``resolve/<rev>/<path>`` layout).
    * MODEL_REVISION -- the immutable commit pin.  The download URL is built as
      ``<repo>/resolve/<rev>/<path>``.
    * MODEL_SHA256 / TOKENIZER_SHA256 -- the verified hashes of the int8 model
      and tokenizer.json; each download is REJECTED + deleted on mismatch.  Both
      are pinned, so the download verifies the model AND the tokenizer.  If
      either is ever left None, a LOUD per-artifact warning names exactly which
      file is unverified until its hash is set.
    * MODEL_FILENAME / TOKENIZER_FILENAME -- the int8 onnx (under ``onnx/``) and
      the tokenizer; cached locally by basename.

Security model
--------------
- Local-only inference (CPUExecutionProvider, single-threaded, no telemetry).
- One-time download over stdlib HTTPS (urllib) -- no huggingface_hub, no extra
  runtime dependency.  Downloaded files are SHA-256 verified (when pinned) and
  chmod 0600.  On a hash mismatch the partial file is removed and the load
  fails closed.
- Offline supported: point ``local_path`` at a directory holding pre-placed
  ``model_int8.onnx`` + ``tokenizer.json`` and the network is never touched.
  An already-cached file is likewise reused without a download.
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

#: Base URL of the pre-built artifact repository.  Defaults to the vetted,
#: HF-staff-maintained Xenova int8 ONNX export of BAAI/bge-small-en-v1.5 -- a
#: single-file quantized model, no conversion step, no personal account.
#: Override precedence: ``base_url`` arg > config ``semantic_model_repo`` >
#: ``MNEMOSYNE_SEMANTIC_MODEL_REPO`` env var > this default.  A custom mirror
#: must serve the same ``resolve/<rev>/<path>`` layout (HuggingFace style).
MODEL_REPO: str = "https://huggingface.co/Xenova/bge-small-en-v1.5"

#: Alias kept for callers/readers who think in "base URL" terms.
MODEL_BASE_URL: str = MODEL_REPO

#: Immutable commit pin on the artifact repo.  A revision is ALWAYS used to
#: build the download URL when set; this makes every artifact (model and
#: ``tokenizer.json``) content-addressable, on top of the explicit SHA-256 pins
#: below that we verify each downloaded file against.
MODEL_REVISION: str | None = "ea104dacec62c0de699686887e3f920caeb4f3e3"

#: Remote paths of the int8 ONNX model + the HuggingFace tokenizer.json,
#: RELATIVE to ``resolve/<rev>/``.  The model lives under an ``onnx/`` prefix in
#: the Xenova repo; the local cache file is flattened to its basename (see
#: :func:`_local_filename`) so offline ``local_path`` dirs stay flat.
MODEL_FILENAME: str = "onnx/model_int8.onnx"
TOKENIZER_FILENAME: str = "tokenizer.json"

#: Expected SHA-256 of each artifact.  BOTH are SHA-pinned (verified ->
#: each download is REJECTED + deleted on mismatch), so the int8 model AND the
#: tokenizer.json are checked against a known-good hash.  These are the hashes
#: of the files served by the pinned ``MODEL_REVISION`` commit; rotate them
#: together if the revision is ever bumped.  Leaving either None disables
#: verification for that file and fires a loud per-artifact warning.
MODEL_SHA256: str | None = (
    "bf64d05457cb391fa88d045faf5927a15ea36d96228ddf23ea970087afdc1197"
)
TOKENIZER_SHA256: str | None = (
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
)

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
    """Build the download URL for ``filename`` from the artifact repo.

    Uses the HuggingFace ``resolve/<rev>/<path>`` layout when a revision is
    pinned (the default), e.g.::

        https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/<rev>/onnx/model_int8.onnx

    When ``MODEL_REVISION`` is None (a custom flat mirror), the filename is
    joined directly onto the base URL.  ``filename`` keeps its remote prefix
    (e.g. ``onnx/...``); only the LOCAL cache path is flattened.
    """
    parts = [base_url.rstrip("/")]
    if MODEL_REVISION:
        parts.extend(("resolve", MODEL_REVISION))
    parts.append(filename.lstrip("/"))
    return "/".join(parts)


def _local_filename(remote_filename: str) -> str:
    """Map a remote artifact path to its FLAT local cache filename.

    The remote model path carries an ``onnx/`` prefix; the local cache (and
    offline ``local_path`` dirs) store the file by basename so no subdirectory
    is required.  The loader reads exactly the name written here.
    """
    return os.path.basename(remote_filename)


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
        """Return (model_path, tokenizer_path) for the active source dir.

        Local files are stored FLAT (by basename), so the remote ``onnx/``
        prefix on ``MODEL_FILENAME`` is stripped here -- the offline
        ``local_path`` dir holds ``model_int8.onnx`` + ``tokenizer.json``
        directly, no ``onnx/`` subdirectory.
        """
        source_dir = self._local_path or self._cache_dir
        return (
            os.path.join(source_dir, _local_filename(MODEL_FILENAME)),
            os.path.join(source_dir, _local_filename(TOKENIZER_FILENAME)),
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
                    f"{_local_filename(MODEL_FILENAME)} / "
                    f"{_local_filename(TOKENIZER_FILENAME)} were not found "
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
        """Warn -- per artifact -- when a SHA-256 pin is unset before a download.

        The message names exactly which file(s) lack a hash so the warning is
        accurate:

        * If MODEL_SHA256 is unset, the MODEL itself is unverified -- the serious
          case (arbitrary downloaded code/weights run locally).  The message
          also notes if the tokenizer is unverified alongside it.
        * If only TOKENIZER_SHA256 is unset, the model IS verified and only
          ``tokenizer.json`` is unpinned -- a narrow gap; the message says so
          explicitly so it does not imply the model is unverified.
        * If both are pinned (the shipped default), nothing is emitted.
        """
        model_unpinned = MODEL_SHA256 is None
        tokenizer_unpinned = TOKENIZER_SHA256 is None

        if model_unpinned:
            # The model is the high-risk artifact; lead with it.
            also = (
                " (tokenizer.json is also unpinned)"
                if tokenizer_unpinned
                else ""
            )
            logger.warning(
                "SECURITY: the semantic MODEL is UNPINNED/UNVERIFIED -- "
                "MODEL_SHA256 is not set, so the downloaded model artifact's "
                "integrity is NOT checked.%s Set the SHA-256 pin (and a pinned "
                "MODEL_REVISION) before any production use, or pre-place a "
                "trusted artifact via local_path.",
                also,
            )
        elif tokenizer_unpinned:
            # Narrow gap: model verified, only the tokenizer lacks a hash.
            logger.warning(
                "SECURITY: tokenizer.json is not SHA-pinned -- "
                "TOKENIZER_SHA256 is not set, so the downloaded tokenizer's "
                "integrity is NOT checked. The semantic MODEL IS verified "
                "against MODEL_SHA256; set TOKENIZER_SHA256 to close this gap, "
                "or pre-place a trusted tokenizer via local_path."
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

        # outputs[0] is last_hidden_state: (1, seq_len, hidden_dim).  bge-small-
        # en-v1.5 is a CLS-pooling model (model card: "pooling: cls, normalize:
        # true") -- the sentence embedding is token 0 (the [CLS] vector), NOT a
        # mean over tokens.  Mean-pooling here is WRONG for bge and degrades
        # retrieval; take the CLS row, then L2-normalize.
        last_hidden_state = outputs[0]  # (1, seq_len, hidden_dim)
        cls_vec = last_hidden_state[:, 0, :][0]  # token 0 -> (hidden_dim,)

        norm = np.linalg.norm(cls_vec)
        if norm > 0:
            cls_vec = cls_vec / norm

        return cls_vec.astype(np.float32).tolist()

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
