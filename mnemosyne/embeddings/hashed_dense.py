# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Lightweight hashed-TFIDF dense embedder for Mnemosyne doc chunks.

This mirrors the ``hashed_tfidf_v1`` embedder that the
``mnemosyne_capture.embed_worker`` ships for turn embeddings.  Reason for
a local copy:

  - mnemosyne (this repo, AGPL) must not take a hard runtime dependency
    on mnemosyne-muses (the proprietary capture repo).  The algorithm is
    stable and tiny, so we vendor it.
  - Shipping now beats shipping perfect.  A proper BGE / MiniLM dense
    backend is tracked separately in ``dense_backend.py`` and requires
    onnxruntime, a download, and tokenizer wiring.  The hashed backend
    activates today with zero new dependencies.

Model identity
--------------
    model_id       = "hashed_tfidf_v1"
    model_version  = 1
    dim            = 128
    quantization   = "int8"

Algorithm
---------
Weinberger-style hashing trick: each token hashes into one of ``DIM``
buckets with a stable sign.  Augmented TF + log dampening keeps long
chunks from overwhelming short ones.  Output is L2-normalised then
quantized to int8 (``[-127, 127]``).  Unknown / empty input returns a
zero vector (``DIM`` zero bytes) -- documented contract for the
retriever's cosine path, which treats zero-norm vectors as "skip".
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter

# ---------------------------------------------------------------------------
# Public constants -- kept in sync with mnemosyne_capture.embed_worker
# ---------------------------------------------------------------------------

MODEL_ID: str = "hashed_tfidf_v1"
MODEL_VERSION: int = 1
DIM: int = 128
QUANTIZATION: str = "int8"

# Matches mnemosyne_capture.embed_worker._TOKEN_RE -- keep tokens aligned
# so the muses turn lane and the mnemosyne doc lane agree on surface form.
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")

# Minimal stopword set -- identical to embed_worker so the two lanes'
# hashed vectors live in the same space.
_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on",
    "with", "is", "was", "were", "be", "are", "this", "that", "these",
    "those", "it", "its", "by", "as", "from", "at", "if",
})


def _tokens(text: str) -> list[str]:
    """Lowercased tokens with camelCase + snake_case expansion."""
    raw = _TOKEN_RE.findall(text or "")
    out: list[str] = []
    for tok in raw:
        low = tok.lower()
        if low in _STOP:
            continue
        out.append(low)
        snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", tok).lower()
        if "_" in snake:
            for part in snake.split("_"):
                if len(part) >= 2 and part not in _STOP:
                    out.append(part)
    return out


def _hash_bucket(token: str) -> tuple[int, int]:
    """Map *token* to ``(bucket, sign)`` using the hashing trick."""
    h = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(h[:4], "big") % DIM
    sign = 1 if (h[4] & 1) == 0 else -1
    return bucket, sign


def embed_floats(text: str) -> list[float]:
    """Return the float vector *before* int8 quantization.

    Used by the retrieval lane when a direct cosine against stored int8
    vectors is cheaper than round-tripping through bytes.
    """
    vec = [0.0] * DIM
    tokens = _tokens(text)
    if not tokens:
        return vec

    tf = Counter(tokens)
    max_tf = max(tf.values()) or 1
    for term, count in tf.items():
        augmented = 0.5 + 0.5 * (count / max_tf)
        weight = augmented * math.log1p(count)
        bucket, sign = _hash_bucket(term)
        vec[bucket] += sign * weight

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_bytes(text: str) -> bytes:
    """Return an int8-quantized ``DIM``-byte vector for storage."""
    floats = embed_floats(text)
    q = [max(-127, min(127, int(round(x * 127)))) for x in floats]
    return struct.pack(f"{DIM}b", *q)


def decode_int8(vector_bytes: bytes) -> list[float]:
    """Unpack stored int8 bytes back into a normalised float list.

    The output norm is approximately 1.0 (quantization noise aside)
    because the source vector was L2-normalised before quantization.
    """
    if not vector_bytes:
        return [0.0] * DIM
    n = len(vector_bytes)
    if n != DIM:
        # Respect whatever was stored; caller may pick a different dim
        # if a newer model_version is in play.
        fmt = f"{n}b"
    else:
        fmt = f"{DIM}b"
    ints = struct.unpack(fmt, vector_bytes)
    return [x / 127.0 for x in ints]


def cosine(a: list[float], b: list[float]) -> float:
    """Standard cosine similarity; returns 0.0 on any zero-norm side."""
    if not a or not b:
        return 0.0
    la = len(a)
    lb = len(b)
    n = la if la == lb else min(la, lb)
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        ai = a[i]
        bi = b[i]
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


__all__ = [
    "DIM",
    "MODEL_ID",
    "MODEL_VERSION",
    "QUANTIZATION",
    "embed_floats",
    "embed_bytes",
    "decode_int8",
    "cosine",
]
