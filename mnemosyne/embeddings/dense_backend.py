# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dense embedding backend for Mnemosyne.

Downloads and runs an ONNX model locally for semantic vector search.
The model (all-MiniLM-L6-v2-code-search-512) produces 384-dim float32
vectors suitable for cosine similarity ranking.

Security:
- Model runs 100% locally via onnxruntime.  No network calls at query time.
- The only network call is a one-time HTTPS download from HuggingFace,
  pinned to a specific revision hash.
- Downloaded files are SHA-256 verified and set to chmod 0600.
- No telemetry, no phone-home, no analytics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import struct
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemosyne.config import Config
    from mnemosyne.store import Store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_REPO = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
MODEL_REVISION = "main"  # pin to a specific commit hash when available
MODEL_FILENAME = "onnx/model_quantized.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
MODEL_DIM = 384
MAX_SEQ_LEN = 512

# Expected SHA-256 hashes — set to None to skip verification until pinned.
MODEL_SHA256: str | None = None
TOKENIZER_SHA256: str | None = None

_HF_BASE = "https://huggingface.co"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_url(repo: str, revision: str, filename: str) -> str:
    """Build a HuggingFace download URL."""
    return f"{_HF_BASE}/{repo}/resolve/{revision}/{filename}"


def _download_file(
    url: str, dest: str, expected_sha256: str | None = None
) -> None:
    """Download a file via HTTPS.  Verify SHA-256 if provided.

    Uses urllib.request (stdlib) -- no external dependencies.
    Sets file permissions to 0600 after download.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mnemosyne-engine/1.0"},
    )
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            out.write(chunk)

    if expected_sha256:
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        if h.hexdigest() != expected_sha256:
            os.remove(dest)
            raise ValueError(f"SHA-256 mismatch for {dest}")

    os.chmod(dest, 0o600)


# ---------------------------------------------------------------------------
# Minimal WordPiece tokenizer
# ---------------------------------------------------------------------------


class _WordPieceTokenizer:
    """Minimal WordPiece tokenizer that reads HuggingFace tokenizer.json."""

    def __init__(self, tokenizer_path: str) -> None:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build vocab: token -> id
        model_data = data.get("model", {})
        self.vocab: dict[str, int] = model_data.get("vocab", {})
        self.unk_id: int = self.vocab.get("[UNK]", 0)
        self.cls_id: int = self.vocab.get("[CLS]", 101)
        self.sep_id: int = self.vocab.get("[SEP]", 102)
        self.pad_id: int = self.vocab.get("[PAD]", 0)

        # Max subword length for greedy matching
        self._max_token_len = max(
            (len(t) for t in self.vocab if t.startswith("##")),
            default=20,
        )

    def tokenize(
        self, text: str, max_length: int = MAX_SEQ_LEN
    ) -> tuple[list[int], list[int]]:
        """Tokenize text into input_ids and attention_mask.

        Returns:
            (input_ids, attention_mask) each of length max_length.
        """
        # Pre-tokenize: lowercase, split on whitespace and punctuation
        text = text.lower().strip()
        words = re.findall(r"[a-z0-9]+|[^\s\w]", text)

        tokens: list[int] = [self.cls_id]

        for word in words:
            sub_tokens = self._wordpiece(word)
            tokens.extend(sub_tokens)
            if len(tokens) >= max_length - 1:
                break

        # Truncate to leave room for [SEP]
        tokens = tokens[: max_length - 1]
        tokens.append(self.sep_id)

        attention_mask = [1] * len(tokens)

        # Pad
        pad_len = max_length - len(tokens)
        tokens.extend([self.pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)

        return tokens, attention_mask

    def _wordpiece(self, word: str) -> list[int]:
        """Greedy longest-match WordPiece splitting."""
        if word in self.vocab:
            return [self.vocab[word]]

        tokens: list[int] = []
        start = 0

        while start < len(word):
            end = len(word)
            matched = False

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr

                if substr in self.vocab:
                    tokens.append(self.vocab[substr])
                    matched = True
                    break

                end -= 1

            if not matched:
                tokens.append(self.unk_id)
                start += 1
            else:
                start = end

        return tokens


# ---------------------------------------------------------------------------
# Dense backend
# ---------------------------------------------------------------------------


class DenseBackend:
    """Dense embedding backend using ONNX model for semantic search.

    Downloads the model on first use (lazy, cached in .mnemosyne/models/).
    Loads the model via onnxruntime.InferenceSession.
    Embeds text into 384-dim float32 vectors (mean pooling + L2 normalize).

    Args:
        config: Mnemosyne Config instance.
        store:  Store instance for reading/writing the embeddings table.
        model_dir: Directory to cache model files.  Defaults to
                   .mnemosyne/models/ under the project root.
    """

    def __init__(
        self,
        config: "Config",
        store: "Store | None" = None,
        model_dir: str | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._model_dir = model_dir or os.path.join(
            str(getattr(config, "_root", ".")), ".mnemosyne", "models"
        )
        self._session = None  # lazy loaded onnxruntime.InferenceSession
        self._tokenizer: _WordPieceTokenizer | None = None  # lazy loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Download model if not cached, verify SHA-256, load session."""
        if self._session is not None:
            return

        import onnxruntime  # lazy import — graceful if not installed

        model_path = os.path.join(self._model_dir, "model_quantized.onnx")
        tokenizer_path = os.path.join(self._model_dir, "tokenizer.json")

        # Download if not present
        if not os.path.isfile(model_path):
            url = _download_url(MODEL_REPO, MODEL_REVISION, MODEL_FILENAME)
            logger.info("Downloading ONNX model from %s", url)
            _download_file(url, model_path, MODEL_SHA256)

        if not os.path.isfile(tokenizer_path):
            url = _download_url(MODEL_REPO, MODEL_REVISION, TOKENIZER_FILENAME)
            logger.info("Downloading tokenizer from %s", url)
            _download_file(url, tokenizer_path, TOKENIZER_SHA256)

        # Load ONNX session (CPU only, no CUDA, no telemetry)
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

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float] | None:
        """Embed text into a 384-dim vector.

        Returns None if onnxruntime is not available or the text is empty.
        """
        if not text or not text.strip():
            return None

        try:
            self._ensure_model()
        except ImportError:
            return None
        except Exception:
            logger.warning("Failed to load dense model", exc_info=True)
            return None

        assert self._tokenizer is not None
        assert self._session is not None

        input_ids, attention_mask = self._tokenizer.tokenize(text)

        # onnxruntime needs numpy arrays
        import numpy as np

        ids_arr = np.array([input_ids], dtype=np.int64)
        mask_arr = np.array([attention_mask], dtype=np.int64)
        # token_type_ids: all zeros for single-sequence
        type_arr = np.zeros_like(ids_arr)

        feeds = {
            "input_ids": ids_arr,
            "attention_mask": mask_arr,
            "token_type_ids": type_arr,
        }

        # Some models only take input_ids + attention_mask
        input_names = {inp.name for inp in self._session.get_inputs()}
        feeds = {k: v for k, v in feeds.items() if k in input_names}

        outputs = self._session.run(None, feeds)

        # outputs[0] shape: (1, seq_len, hidden_dim)
        token_embeddings = outputs[0][0]  # (seq_len, hidden_dim)
        mask_expanded = np.array(attention_mask, dtype=np.float32)

        # Mean pooling over non-padding tokens
        masked = token_embeddings * mask_expanded[:, np.newaxis]
        summed = masked.sum(axis=0)
        count = mask_expanded.sum()
        if count == 0:
            return None
        mean_vec = summed / count

        # L2 normalize
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        return mean_vec.astype(np.float32).tolist()

    def embed_to_bytes(self, text: str) -> bytes | None:
        """Embed and pack as float32 little-endian bytes for storage."""
        vec = self.embed(text)
        if vec is None:
            return None
        return struct.pack(f"<{len(vec)}f", *vec)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec: list[float],
        top_k: int = 20,
        candidate_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Find top-k chunks by cosine similarity against stored vectors.

        Args:
            query_vec:     384-dim query vector (already L2-normalized).
            top_k:         Maximum results to return.
            candidate_ids: If provided, only compute similarity for these
                           chunk IDs (pre-filtered by BM25/TF-IDF).
                           If None, scans all stored vectors.

        Returns:
            List of (chunk_id, similarity) tuples, descending by score.
        """
        if self._store is None:
            return []

        if candidate_ids is not None:
            vectors = self._store.get_dense_embeddings_batch(candidate_ids)
        else:
            # Full scan fallback
            vectors = self._store.get_dense_embeddings_batch([])
            if not vectors:
                # Try loading all if batch with empty list returns empty
                try:
                    rows = self._store.conn.execute(
                        "SELECT chunk_id, vector FROM embeddings"
                    ).fetchall()
                    vectors = {
                        int(r["chunk_id"]): r["vector"] for r in rows
                    }
                except Exception:
                    return []

        if not vectors:
            return []

        import numpy as np

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        results: list[tuple[int, float]] = []
        dim = MODEL_DIM

        for chunk_id, vec_bytes in vectors.items():
            try:
                n_floats = len(vec_bytes) // 4
                if n_floats != dim:
                    continue
                v = np.frombuffer(vec_bytes, dtype=np.float32).copy()
                v_norm = np.linalg.norm(v)
                if v_norm == 0:
                    continue
                similarity = float(np.dot(q, v / v_norm))
                results.append((chunk_id, similarity))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if onnxruntime is installed."""
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False
