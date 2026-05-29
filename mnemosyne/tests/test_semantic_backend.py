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
        # A flat custom mirror (no pinned revision): filename joins directly.
        monkeypatch.setattr(sb, "MODEL_REVISION", None)
        url = sb._artifact_url("https://host.invalid/dir/", "onnx/model_int8.onnx")
        assert url == "https://host.invalid/dir/onnx/model_int8.onnx"

    def test_artifact_url_with_pinned_revision_uses_hf_resolve(self, monkeypatch):
        # The default path: HuggingFace-style resolve/<rev>/<path>, prefix kept.
        monkeypatch.setattr(sb, "MODEL_REVISION", "abc123")
        url = sb._artifact_url(
            "https://huggingface.co/Xenova/bge-small-en-v1.5",
            "onnx/model_int8.onnx",
        )
        assert url == (
            "https://huggingface.co/Xenova/bge-small-en-v1.5"
            "/resolve/abc123/onnx/model_int8.onnx"
        )

    def test_artifact_url_builds_for_pinned_defaults(self):
        # The real shipped constants must produce a well-formed resolve URL.
        url = sb._artifact_url(sb.MODEL_REPO, sb.MODEL_FILENAME)
        assert url == (
            f"{sb.MODEL_REPO}/resolve/{sb.MODEL_REVISION}/{sb.MODEL_FILENAME}"
        )
        assert "/resolve/" in url and url.endswith("onnx/model_int8.onnx")

    def test_local_filename_flattens_remote_prefix(self):
        assert sb._local_filename("onnx/model_int8.onnx") == "model_int8.onnx"
        assert sb._local_filename("tokenizer.json") == "tokenizer.json"

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

    def test_shipped_defaults_pin_model_warn_only_tokenizer(self, caplog):
        """As shipped: the model IS sha-pinned; only the tokenizer warns."""
        # The big-risk file is verified, the immutable-commit tokenizer is not.
        assert sb.MODEL_SHA256 is not None and len(sb.MODEL_SHA256) == 64
        assert sb.TOKENIZER_SHA256 is None
        with caplog.at_level("WARNING"):
            sb.SemanticBackend._maybe_warn_unverified()
        msgs = [rec.message for rec in caplog.records]
        warned = [m for m in msgs if "UNPINNED/UNVERIFIED" in m]
        assert warned, "tokenizer should warn while unpinned"
        # The warning names the tokenizer, not the model.
        assert all("TOKENIZER_SHA256" in m for m in warned)
        assert all("MODEL_SHA256" not in m for m in warned)


class TestShippedModelPins:
    """Lock the vetted Xenova artifact pins so a silent repoint is caught."""

    def test_model_repo_is_xenova_prebuilt(self):
        assert sb.MODEL_REPO == "https://huggingface.co/Xenova/bge-small-en-v1.5"
        assert sb.MODEL_BASE_URL == sb.MODEL_REPO

    def test_model_revision_is_immutable_commit_pin(self):
        assert (
            sb.MODEL_REVISION == "ea104dacec62c0de699686887e3f920caeb4f3e3"
        )

    def test_model_filename_is_int8_under_onnx_prefix(self):
        assert sb.MODEL_FILENAME == "onnx/model_int8.onnx"
        assert sb.TOKENIZER_FILENAME == "tokenizer.json"

    def test_model_sha256_is_the_verified_hash(self):
        assert sb.MODEL_SHA256 == (
            "bf64d05457cb391fa88d045faf5927a15ea36d96228ddf23ea970087afdc1197"
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
    per-token embedding.  Enough to exercise the tokenize -> run -> CLS-pool ->
    normalize path and assert the output dim + unit norm.  (Every row here is a
    scalar multiple of one weight vector, so it cannot distinguish CLS from mean
    pooling -- :func:`_build_pooling_fixture_model` is used for that.)
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


#: Per-id embedding table used by the pooling fixture.  Rows are deliberately
#: NOT colinear so token 0 (CLS) points in a different direction than the mean
#: of all tokens -- which is what lets the test tell CLS-pooling from mean.
_POOL_VOCAB_SIZE = 16


def _pooling_embedding_table(dim: int) -> "object":
    """A (vocab, dim) table whose rows differ in DIRECTION, not just scale."""
    import numpy as np

    rng = np.random.default_rng(1234)
    table = rng.standard_normal((_POOL_VOCAB_SIZE, dim)).astype(np.float32)
    # Make the CLS row (id 2) point somewhere clearly distinct so CLS != mean.
    table[2] = 0.0
    table[2, 0] = 1.0  # CLS embedding = unit e0
    return table


def _build_pooling_fixture_model(path: str, dim: int = sb.MODEL_DIM) -> None:
    """ONNX model that Gathers a distinct per-id row -> last_hidden_state.

    Unlike :func:`_build_fixture_model` (colinear rows), each token id maps to
    its OWN embedding row via Gather, so the CLS row and the token-mean point in
    different directions.  This is what makes a CLS-vs-mean assertion meaningful.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT64, ["b", "s"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["b", "s"]
    )
    output = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["b", "s", dim]
    )

    table = _pooling_embedding_table(dim)
    emb = numpy_helper.from_array(table, name="emb")
    # Gather(emb, input_ids) -> (b, s, dim)
    gather = helper.make_node("Gather", ["emb", "input_ids"], ["last_hidden_state"], axis=0)

    graph = helper.make_graph(
        [gather],
        "pooling_fixture",
        [input_ids, attention_mask],
        [output],
        initializer=[emb],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 9
    onnx.save(model, path)


@requires_ort
@requires_onnx
class TestRealEmbeddingWithFixture:
    @pytest.fixture
    def model_dir(self, tmp_path):
        # Files are placed FLAT (basename), matching how the loader resolves an
        # offline/cache dir -- the remote ``onnx/`` prefix is stripped locally.
        d = tmp_path / "models"
        d.mkdir()
        _build_fixture_model(str(d / sb._local_filename(sb.MODEL_FILENAME)))
        _write_fixture_tokenizer(
            str(d / sb._local_filename(sb.TOKENIZER_FILENAME))
        )
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
class TestClsPoolingNotMean:
    """bge-small-en-v1.5 is CLS-pooled: the embedding is token 0, NOT the mean.

    Uses the Gather fixture (distinct, non-colinear per-id rows) so CLS and mean
    point in different directions; asserts the produced vector equals the
    normalized CLS row and does NOT equal the normalized token-mean.
    """

    @pytest.fixture
    def pooling_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        _build_pooling_fixture_model(
            str(d / sb._local_filename(sb.MODEL_FILENAME))
        )
        _write_fixture_tokenizer(
            str(d / sb._local_filename(sb.TOKENIZER_FILENAME))
        )
        return str(d)

    def test_embedding_is_cls_token_not_mean(self, pooling_dir):
        import numpy as np

        text = "i like dogs"  # several distinct tokens -> CLS row != mean row
        backend = sb.SemanticBackend(_Config(), local_path=pooling_dir)
        backend._ensure_model()  # builds the same tokenizer the embed path uses

        # Reproduce the exact token sequence the backend feeds the model.
        input_ids, attention_mask = backend._tokenizer.tokenize(
            text, max_length=sb.MAX_SEQ_LEN
        )
        table = _pooling_embedding_table(sb.MODEL_DIM)
        rows = table[np.array(input_ids)]  # (seq, dim) -- model output rows

        # Expected CLS pooling: token 0 (the [CLS] row), L2-normalized.
        cls_row = rows[0]
        expected_cls = cls_row / np.linalg.norm(cls_row)

        # The WRONG (mean) result for the same input, masked + L2-normalized.
        mask = np.array(attention_mask, dtype=np.float32)
        mean_row = (rows * mask[:, None]).sum(axis=0) / mask.sum()
        expected_mean = mean_row / np.linalg.norm(mean_row)

        # Sanity: the two pooling strategies genuinely differ for this fixture,
        # otherwise the test would pass vacuously.
        assert not np.allclose(expected_cls, expected_mean, atol=1e-3)

        got = backend.embed_passage(text)
        assert got is not None
        got = np.array(got, dtype=np.float32)

        # The backend must match CLS pooling and NOT mean pooling.
        assert np.allclose(got, expected_cls, atol=1e-5)
        assert not np.allclose(got, expected_mean, atol=1e-3)

    def test_cls_row_is_token_zero_exactly(self, pooling_dir):
        """A single-token passage isolates token 0 -> output == normalized CLS."""
        import numpy as np

        backend = sb.SemanticBackend(_Config(), local_path=pooling_dir)
        backend._ensure_model()
        table = _pooling_embedding_table(sb.MODEL_DIM)
        # The CLS row of this fixture is the unit vector e0.
        cls_norm = table[2] / np.linalg.norm(table[2])

        got = np.array(backend.embed_passage("dog"), dtype=np.float32)
        # Token 0 is always [CLS] (id 2); CLS pooling returns its normalized row.
        assert np.allclose(got, cls_norm, atol=1e-5)
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-5


@requires_ort
class TestClsPoolingNoOnnx:
    """CLS-pooling assertion that needs only onnxruntime+numpy (no ``onnx``).

    The ``onnx`` package is the optional fixture *builder*; it is absent in the
    base ``[dense]`` env, so the fixture-backed pooling tests above skip there.
    This test injects a fake session whose ``run`` returns a crafted
    ``last_hidden_state`` with non-colinear rows, so it proves the embed path
    takes token 0 (CLS), NOT the mean, wherever onnxruntime is importable.
    """

    def test_embed_uses_cls_token_zero_not_mean(self, monkeypatch):
        import numpy as np

        # A (1, seq, dim) last_hidden_state with rows pointing different ways.
        seq, dim = 4, sb.MODEL_DIM
        rows = np.zeros((seq, dim), dtype=np.float32)
        rows[0, 0] = 1.0   # CLS row -> e0
        rows[1, 1] = 2.0   # other tokens -> other axes (so mean != e0)
        rows[2, 2] = 3.0
        rows[3, 3] = 4.0
        last_hidden_state = rows[np.newaxis, :, :]  # (1, seq, dim)

        class _FakeInput:
            def __init__(self, name):
                self.name = name

        class _FakeSession:
            def get_inputs(self):
                return [_FakeInput("input_ids"), _FakeInput("attention_mask")]

            def run(self, _outs, _feeds):
                return [last_hidden_state]

        class _FakeTok:
            def tokenize(self, text, max_length=sb.MAX_SEQ_LEN):
                # CLS at index 0; full attention over all 4 positions.
                return [2, 5, 6, 7], [1, 1, 1, 1]

        backend = sb.SemanticBackend(_Config())

        def fake_ensure():
            backend._session = _FakeSession()
            backend._tokenizer = _FakeTok()

        monkeypatch.setattr(backend, "_ensure_model", fake_ensure)

        got = np.array(backend.embed_passage("anything"), dtype=np.float32)

        # CLS pooling: normalized token-0 row == e0.
        expected_cls = np.zeros(dim, dtype=np.float32)
        expected_cls[0] = 1.0
        # Mean pooling (the WRONG result) would blend all four axes.
        masked_mean = rows.mean(axis=0)
        expected_mean = masked_mean / np.linalg.norm(masked_mean)

        assert not np.allclose(expected_cls, expected_mean, atol=1e-3)
        assert np.allclose(got, expected_cls, atol=1e-6)
        assert not np.allclose(got, expected_mean, atol=1e-3)
        assert abs(float(np.linalg.norm(got)) - 1.0) < 1e-6


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
        model_name = sb._local_filename(sb.MODEL_FILENAME)
        _build_fixture_model(str(src / model_name))
        _write_fixture_tokenizer(
            str(src / sb._local_filename(sb.TOKENIZER_FILENAME))
        )
        model_bytes = (src / model_name).read_bytes()

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
        assert not os.path.isfile(str(cache / model_name))
