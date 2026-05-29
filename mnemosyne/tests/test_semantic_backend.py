# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the BGE-small semantic embedding backend.

The ONNX-dependent tests (real 384-dim vectors, SHA-verified download against a
fixture model, offline local_path load) are gated on onnxruntime + numpy being
importable, since this engine ships them only as the optional ``[dense]`` extra.
The fallback-to-None paths and the pure-logic tests run unconditionally.
"""

from __future__ import annotations

import struct

import pytest

from mnemosyne.embeddings import semantic_backend as sb

# ---------------------------------------------------------------------------
# Optional-dependency gating
# ---------------------------------------------------------------------------

try:  # numpy + onnxruntime are the [dense] extra; absent in the base test env.
    import numpy as _np  # noqa: F401
    import onnxruntime as _ort  # noqa: F401

    _HAVE_ORT = True
except ImportError:
    _HAVE_ORT = False

try:
    import onnx as _onnx  # noqa: F401

    _HAVE_ONNX = True
except ImportError:
    _HAVE_ONNX = False

requires_ort = pytest.mark.skipif(
    not _HAVE_ORT, reason="onnxruntime/numpy not installed ([dense] extra)"
)
requires_onnx = pytest.mark.skipif(
    not _HAVE_ONNX, reason="onnx not installed (needed to build the fixture)"
)


# ---------------------------------------------------------------------------
# Minimal config double (the backend only reads embedding.* overrides)
# ---------------------------------------------------------------------------


class _Embedding:
    def __init__(self, **kw):
        self.semantic_model = kw.get("semantic_model")
        self.semantic_model_repo = kw.get("semantic_model_repo")
        self.semantic_local_path = kw.get("semantic_local_path")


class _Config:
    def __init__(self, **kw):
        self.embedding = _Embedding(**kw)


# ---------------------------------------------------------------------------
# Pure-logic tests (no heavy deps) -- always run
# ---------------------------------------------------------------------------


class TestCacheAndUrlResolution:
    def test_default_cache_dir_uses_xdg(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdgcache")
        monkeypatch.delenv(sb.ENV_CACHE_DIR, raising=False)
        assert sb.default_cache_dir() == "/tmp/xdgcache/mnemosyne/models"

    def test_default_cache_dir_falls_back_to_home(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.delenv(sb.ENV_CACHE_DIR, raising=False)
        got = sb.default_cache_dir()
        assert got.endswith("/.cache/mnemosyne/models")

    def test_cache_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(sb.ENV_CACHE_DIR, "/srv/models")
        assert sb.default_cache_dir() == "/srv/models"

    def test_base_url_precedence_arg_over_config_over_env(self, monkeypatch):
        monkeypatch.setenv(sb.ENV_MODEL_REPO, "https://env.invalid/m")
        cfg = _Config(semantic_model_repo="https://config.invalid/m")
        # arg beats everything
        assert (
            sb._resolve_base_url(cfg, "https://arg.invalid/m")
            == "https://arg.invalid/m"
        )
        # config beats env
        assert sb._resolve_base_url(cfg, None) == "https://config.invalid/m"
        # env beats default
        assert (
            sb._resolve_base_url(_Config(), None) == "https://env.invalid/m"
        )

    def test_base_url_default_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv(sb.ENV_MODEL_REPO, raising=False)
        assert sb._resolve_base_url(_Config(), None) == sb.MODEL_REPO

    def test_artifact_url_without_revision(self, monkeypatch):
        monkeypatch.setattr(sb, "MODEL_REVISION", None)
        url = sb._artifact_url("https://host.invalid/dir/", "model_quantized.onnx")
        assert url == "https://host.invalid/dir/model_quantized.onnx"

    def test_artifact_url_with_pinned_revision(self, monkeypatch):
        monkeypatch.setattr(sb, "MODEL_REVISION", "abc123")
        url = sb._artifact_url("https://host.invalid/dir", "tokenizer.json")
        assert url == "https://host.invalid/dir/abc123/tokenizer.json"

    def test_query_instruction_is_official_bge_prefix(self):
        assert sb.QUERY_INSTRUCTION == (
            "Represent this sentence for searching relevant passages:"
        )


class TestUnverifiedWarning:
    def test_warns_when_shas_unset(self, monkeypatch, caplog):
        monkeypatch.setattr(sb, "MODEL_SHA256", None)
        monkeypatch.setattr(sb, "TOKENIZER_SHA256", None)
        with caplog.at_level("WARNING"):
            sb.SemanticBackend._maybe_warn_unverified()
        assert any(
            "UNPINNED/UNVERIFIED" in rec.message for rec in caplog.records
        )

    def test_silent_when_shas_set(self, monkeypatch, caplog):
        monkeypatch.setattr(sb, "MODEL_SHA256", "a" * 64)
        monkeypatch.setattr(sb, "TOKENIZER_SHA256", "b" * 64)
        with caplog.at_level("WARNING"):
            sb.SemanticBackend._maybe_warn_unverified()
        assert not any(
            "UNPINNED/UNVERIFIED" in rec.message for rec in caplog.records
        )


class TestGracefulFallback:
    def test_empty_text_returns_none(self):
        backend = sb.SemanticBackend(_Config())
        assert backend.embed("") is None
        assert backend.embed("   ") is None
        assert backend.embed_query("") is None
        assert backend.embed_passage("\n\t") is None

    def test_onnxruntime_absent_returns_none(self, monkeypatch):
        """Simulate onnxruntime import failure -> embed returns None.

        Runs in every environment: when onnxruntime is genuinely absent the
        import already fails; when it is present this forces the ImportError
        branch.  Either way embed must degrade to None on non-empty text.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("simulated missing onnxruntime")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        backend = sb.SemanticBackend(_Config())
        assert backend.embed("hello world", is_query=True) is None
        assert backend.embed("a stored passage") is None

    def test_local_path_missing_files_returns_none(self, tmp_path, monkeypatch):
        """local_path set but files absent -> _ensure_model raises -> None.

        Skipped without onnxruntime, since _ensure_model imports it before the
        file check; the file-missing branch is only reachable once ORT loads.
        """
        if not _HAVE_ORT:
            pytest.skip("onnxruntime not installed")
        backend = sb.SemanticBackend(_Config(), local_path=str(tmp_path))
        assert backend.embed("query", is_query=True) is None

    def test_is_available_reflects_environment(self):
        assert sb.SemanticBackend.is_available() == _HAVE_ORT


class TestConfigWiring:
    def test_get_semantic_backend_disabled_returns_none(self):
        from mnemosyne.embeddings import get_semantic_backend

        assert get_semantic_backend(_Config(semantic_model=None)) is None

    def test_get_semantic_backend_enabled_returns_instance(self):
        from mnemosyne.embeddings import get_semantic_backend

        backend = get_semantic_backend(
            _Config(semantic_model="bge-small-en-v1.5")
        )
        assert isinstance(backend, sb.SemanticBackend)

    def test_local_path_read_from_config(self, tmp_path):
        backend = sb.SemanticBackend(
            _Config(semantic_local_path=str(tmp_path))
        )
        assert backend._local_path == str(tmp_path)


# ---------------------------------------------------------------------------
# Fixture ONNX builder -- only when onnx is importable
# ---------------------------------------------------------------------------


def _build_fixture_model(path: str, dim: int = sb.MODEL_DIM) -> None:
    """Write a trivial ONNX model that outputs a (1, seq, dim) tensor.

    The model ignores token *values*: it casts input_ids to float and projects
    via a fixed identity-ish matmul to ``dim`` channels, giving a deterministic
    per-token embedding.  Enough to exercise the tokenize -> run -> mean-pool ->
    normalize path and assert the output dim + unit norm.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # input_ids: (batch, seq) int64 ; attention_mask: (batch, seq) int64
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["b", "s"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["b", "s"]
    )
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["b", "s", dim]
    )

    cast = helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT)
    # unsqueeze to (b, s, 1) then matmul by (1, dim) weight -> (b, s, dim)
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    unsqueeze = helper.make_node("Unsqueeze", ["ids_f", "axes"], ["ids_u"])
    w = np.linspace(0.01, 0.5, dim, dtype=np.float32).reshape(1, dim)
    weight = numpy_helper.from_array(w, name="proj")
    matmul = helper.make_node("MatMul", ["ids_u", "proj"], ["last_hidden_state"])

    graph = helper.make_graph(
        [cast, unsqueeze, matmul],
        "fixture",
        [input_ids, attention_mask],
        [output],
        initializer=[axes, weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


def _write_fixture_tokenizer(path: str) -> None:
    """Write a minimal HuggingFace-style tokenizer.json the WordPiece reader can parse."""
    import json

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for i, tok in enumerate(
        ["dog", "dogs", "animal", "like", "cat", "favorite", "i"], start=4
    ):
        vocab[tok] = i
    data = {"model": {"vocab": vocab}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@requires_ort
@requires_onnx
class TestRealEmbeddingWithFixture:
    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        _build_fixture_model(str(d / sb.MODEL_FILENAME))
        _write_fixture_tokenizer(str(d / sb.TOKENIZER_FILENAME))
        return str(d)

    def test_passage_is_384_dim_and_normalized(self, model_dir):
        import numpy as np

        backend = sb.SemanticBackend(_Config(), local_path=model_dir)
        vec = backend.embed_passage("i like dogs")
        assert vec is not None
        assert len(vec) == sb.MODEL_DIM
        assert abs(float(np.linalg.norm(np.array(vec))) - 1.0) < 1e-4

    def test_query_is_384_dim_and_normalized(self, model_dir):
        import numpy as np

        backend = sb.SemanticBackend(_Config(), local_path=model_dir)
        vec = backend.embed_query("what animal do i like")
        assert vec is not None
        assert len(vec) == sb.MODEL_DIM
        assert abs(float(np.linalg.norm(np.array(vec))) - 1.0) < 1e-4

    def test_query_prefix_changes_the_embedding(self, model_dir, monkeypatch):
        """The query path must prepend the instruction; passage must not.

        We capture the tokenizer input to prove the prefix is applied on query
        embeds only.
        """
        backend = sb.SemanticBackend(_Config(), local_path=model_dir)
        backend._ensure_model()
        seen: list[str] = []
        real_tokenize = backend._tokenizer.tokenize

        def spy(text, max_length=sb.MAX_SEQ_LEN):
            seen.append(text)
            return real_tokenize(text, max_length=max_length)

        monkeypatch.setattr(backend._tokenizer, "tokenize", spy)
        backend.embed_passage("i like dogs")
        backend.embed_query("i like dogs")
        assert seen[0] == "i like dogs"  # passage: no prefix
        assert seen[1].startswith(sb.QUERY_INSTRUCTION)  # query: prefixed
        assert seen[1].endswith("i like dogs")

    def test_int8_byte_packing_roundtrips_dim(self, model_dir):
        backend = sb.SemanticBackend(_Config(), local_path=model_dir)
        blob = backend.embed_to_int8_bytes("i like dogs")
        assert blob is not None
        assert len(blob) == sb.MODEL_DIM  # 1 byte per dim
        ints = struct.unpack(f"{sb.MODEL_DIM}b", blob)
        assert all(-127 <= x <= 127 for x in ints)

    def test_offline_local_path_makes_no_network_call(
        self, model_dir, monkeypatch
    ):
        """With local_path set, urllib must never be invoked."""
        import urllib.request

        def boom(*a, **k):  # pragma: no cover - asserts it is NOT called
            raise AssertionError("network call attempted in offline mode")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        backend = sb.SemanticBackend(_Config(), local_path=model_dir)
        vec = backend.embed_passage("i like dogs")
        assert vec is not None and len(vec) == sb.MODEL_DIM

    def test_cached_files_skip_download(self, model_dir, monkeypatch):
        """Files already in the cache dir -> no download attempted."""
        import urllib.request

        def boom(*a, **k):  # pragma: no cover
            raise AssertionError("download attempted though cache is populated")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        # Use cache_dir (not local_path) but pre-populate it; download is skipped
        # because os.path.isfile() is true for both files.
        backend = sb.SemanticBackend(_Config(), cache_dir=model_dir)
        assert backend.embed_passage("i like dogs") is not None


@requires_ort
@requires_onnx
class TestShaVerifiedDownload:
    """SHA-256 mismatch on a (mocked) download -> rejected + file removed."""

    def test_sha_mismatch_rejects_and_removes_file(
        self, tmp_path, monkeypatch
    ):
        import os
        import urllib.request

        cache = tmp_path / "cache"
        cache.mkdir()

        # Build a real fixture model + tokenizer to serve as the "download".
        src = tmp_path / "src"
        src.mkdir()
        _build_fixture_model(str(src / sb.MODEL_FILENAME))
        _write_fixture_tokenizer(str(src / sb.TOKENIZER_FILENAME))
        model_bytes = (src / sb.MODEL_FILENAME).read_bytes()

        class _FakeResp:
            def __init__(self, payload):
                self._payload = payload
                self._read = False

            def read(self, n=-1):
                if self._read:
                    return b""
                self._read = True
                return self._payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, *a, **k):
            return _FakeResp(model_bytes)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        # Pin a WRONG sha so verification fails.
        monkeypatch.setattr(sb, "MODEL_SHA256", "0" * 64)
        monkeypatch.setattr(sb, "TOKENIZER_SHA256", None)

        backend = sb.SemanticBackend(_Config(), cache_dir=str(cache))
        # embed swallows the ValueError and returns None; the file must be gone.
        assert backend.embed_passage("i like dogs") is None
        assert not os.path.isfile(str(cache / sb.MODEL_FILENAME))
