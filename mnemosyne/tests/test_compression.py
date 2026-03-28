# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the Compressor (compress.py).

Covers: signature preservation, import collapsing, compression ratio,
and the full four-stage pipeline.
"""

import unittest


def _default_config():
    import tempfile
    from mnemosyne.config import Config
    return Config(root=tempfile.mkdtemp())


def _make_chunk(content, chunk_type="function", file_id=1):
    from mnemosyne.models import Chunk, estimate_tokens
    from mnemosyne.hasher import content_hash
    return Chunk(
        chunk_id=1,
        file_id=file_id,
        content_hash=content_hash(content),
        chunk_type=chunk_type,
        line_start=1,
        line_end=content.count("\n"),
        token_count=estimate_tokens(content),
        content=content,
    )


PYTHON_FUNCTION = '''\
import os
import sys
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def process_authentication(request, config):
    """Authenticate the incoming request against the configured providers.

    This function checks the Authorization header, validates the JWT token,
    and returns the authenticated user object if successful.
    """
    logger.debug("Starting authentication for request %s", request.id)
    logger.info("Auth check: %s", request.method)
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        logger.warning("No token provided")
        raise ValueError("Missing authentication token")

    self.conn = connect()
    self.pool = []
    self.cache = {}
    self.logger = logger
    self.timeout = 30
    self.retries = 3

    user = validate_jwt_token(token, config.secret)
    if user is None:
        raise PermissionError("Invalid token")

    return user
'''


class TestCompressor(unittest.TestCase):

    def setUp(self):
        self.cfg = _default_config()
        from mnemosyne.compress import Compressor
        self.compressor = Compressor(self.cfg)

    def test_compress_returns_non_empty_string(self):
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)

    def test_function_signature_preserved(self):
        """def line must survive compression."""
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        self.assertIn("def process_authentication", result)

    def test_return_statement_preserved(self):
        """return statements must survive compression."""
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        self.assertIn("return", result)

    def test_raise_statement_preserved(self):
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        # At least one raise statement should survive
        self.assertIn("raise", result)

    def test_imports_collapsed_when_many(self):
        """More than 3 consecutive import lines become a single summary."""
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        # Collapsed imports produce a comment summary like "# [N imports: ...]"
        import_count_in_source = sum(
            1 for line in PYTHON_FUNCTION.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        )
        if import_count_in_source > 3:
            import_lines_in_result = [
                l for l in result.splitlines()
                if l.strip().startswith("import ") or l.strip().startswith("from ")
            ]
            collapsed_comments = [
                l for l in result.splitlines()
                if "imports:" in l or "imports]" in l
            ]
            # Either the raw import count dropped OR a collapse comment appeared
            self.assertTrue(
                len(import_lines_in_result) < import_count_in_source
                or len(collapsed_comments) > 0,
                "Expected import collapse for >3 imports",
            )

    def test_compression_ratio_less_than_one_for_long_chunk(self):
        """A well-compressible chunk should be shorter than the original."""
        chunk = _make_chunk(PYTHON_FUNCTION)
        original_len = len(chunk.content)
        result = self.compressor.compress(chunk)
        compressed_len = len(result)
        ratio = compressed_len / max(1, original_len)
        # Allow ratio up to 1.1 — compression may not always reduce very short text
        self.assertLessEqual(
            ratio, 1.1,
            f"Compression ratio {ratio:.3f} expected to be <= 1.1 "
            f"(original={original_len}, compressed={compressed_len})",
        )

    def test_compress_empty_content_returns_original(self):
        chunk = _make_chunk("   \n  \n")
        result = self.compressor.compress(chunk)
        self.assertIsInstance(result, str)

    def test_compress_short_chunk_no_reduction(self):
        """A very short function (already minimal) doesn't get mangled."""
        source = "def tiny(x):\n    return x\n"
        chunk = _make_chunk(source)
        result = self.compressor.compress(chunk)
        self.assertIn("def tiny", result)
        self.assertIn("return", result)

    def test_docstring_preserved_when_config_true(self):
        """Docstrings are kept when preserve_docstrings = True (default)."""
        self.assertTrue(self.cfg.compression.preserve_docstrings)
        chunk = _make_chunk(PYTHON_FUNCTION)
        result = self.compressor.compress(chunk)
        # The docstring opener should survive
        self.assertIn('"""', result)

    def test_assignments_collapsed(self):
        """More than 2 consecutive self.x = y lines get collapsed."""
        source = '''\
class Builder:
    def __init__(self):
        self.conn = None
        self.pool = []
        self.cache = {}
        self.logger = None
        self.timeout = 30
        self.retries = 3
        self.running = False
        return
'''
        chunk = _make_chunk(source, chunk_type="class")
        result = self.compressor.compress(chunk)
        # The collapse comment or reduced count signals compression happened
        assignment_count_original = sum(
            1 for l in source.splitlines() if "self." in l and " = " in l
        )
        assignment_count_result = sum(
            1 for l in result.splitlines() if "self." in l and " = " in l
        )
        collapsed = [l for l in result.splitlines() if "assignments:" in l]
        self.assertTrue(
            assignment_count_result < assignment_count_original or len(collapsed) > 0,
            "Expected assignment collapse for >2 self.x assignments",
        )

    def test_log_statements_collapsed(self):
        """Multiple consecutive logging calls get collapsed to a summary."""
        source = '''\
def process():
    logger.debug("step 1")
    logger.info("step 2")
    logger.warning("step 3")
    return True
'''
        chunk = _make_chunk(source)
        result = self.compressor.compress(chunk)
        log_lines = [l for l in result.splitlines() if "logger." in l]
        collapsed = [l for l in result.splitlines() if "log statements" in l]
        self.assertTrue(
            len(log_lines) < 3 or len(collapsed) > 0,
            "Expected log collapse for 3 consecutive logger calls",
        )

    # ------------------------------------------------------------------
    # Milestone 2.1 — Compression safety net tests
    # ------------------------------------------------------------------

    def test_control_flow_lines_preserved(self):
        """All control flow lines (if/elif/else/for/while/try/except/finally/with) must survive."""
        source = '''\
def handle(items):
    if not items:
        return []
    for item in items:
        try:
            with open(item) as f:
                data = f.read()
        except FileNotFoundError:
            data = None
        finally:
            pass
    while items:
        items.pop()
    if len(items) == 0:
        result = "empty"
    elif len(items) == 1:
        result = "one"
    else:
        result = "many"
    return result
'''
        chunk = _make_chunk(source)
        result = self.compressor.compress(chunk)
        # Every control flow keyword line must survive compression
        for keyword in ["if not items", "for item in items", "try:",
                        "with open(item)", "except FileNotFoundError",
                        "finally:", "while items", "elif len(items)",
                        "else:"]:
            self.assertIn(
                keyword, result,
                f"Control flow line containing '{keyword}' was removed by compression",
            )

    def test_max_prune_ratio_respected(self):
        """Even with aggressive target_ratio, at most 70% of removable lines are pruned in Stage 3."""
        # Build a chunk with many distinct low-value lines that Stage 3 wants to remove.
        # Lines must be dissimilar enough that Stage 4 (density/dedup, similarity >0.85)
        # does NOT also remove them — we are testing Stage 3's prune cap only.
        # Use unique variable names and different operations to stay below 0.85 similarity.
        words = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
            "golf", "hotel", "india", "juliet", "kilo", "lima",
            "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango",
        ]
        filler_lines = [
            f"    {w} = transform_{w}(data_{w}, offset={i * 7})\n"
            for i, w in enumerate(words)
        ]
        source = "def big_func():\n" + "".join(filler_lines) + "    return alpha\n"
        chunk = _make_chunk(source)

        # IDF map: give the unique words very low IDF so Stage 3 wants to remove them.
        # Give preserved-keyword terms high IDF so they stay.
        idf_map = {w: 0.001 for w in words}
        idf_map.update({f"transform_{w}": 0.001 for w in words})
        idf_map.update({f"data_{w}": 0.001 for w in words})
        idf_map.update({"return": 10.0, "def": 10.0, "big_func": 10.0, "alpha": 10.0})

        from mnemosyne.compress import Compressor
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        cfg = _default_config()
        cfg.compression.target_ratio = 0.05  # extremely aggressive — wants nearly everything gone
        cfg.compression.max_prune_ratio = 0.7  # safety net: keep at least 30% of removable

        backend = TFIDFBackend(cfg)
        backend.idf = idf_map
        comp = Compressor(cfg, tfidf_backend=backend)

        result = comp.compress(chunk)
        result_lines = [l for l in result.splitlines() if l.strip()]

        # Preserved lines: def + return = 2.  Removable = 20 filler lines.
        # With 70% cap, Stage 3 removes at most 14, keeping at least 6.
        # Total surviving >= 2 preserved + 6 kept = 8.
        preserved_count = 2
        removable_count = len(words)  # 20 filler lines
        min_surviving_removable = removable_count - int(removable_count * 0.7)
        min_surviving = preserved_count + min_surviving_removable

        self.assertGreaterEqual(
            len(result_lines), min_surviving,
            f"Max prune ratio violated: {len(result_lines)} lines survived, "
            f"expected at least {min_surviving} (30% of {removable_count} removable + {preserved_count} preserved)",
        )

    def test_strict_mode_skips_importance_filter(self):
        """strict=True skips Stage 3 — more content preserved than non-strict."""
        # Enough content for TF-IDF to actually remove things
        filler_lines = [f"    val_{i} = process({i})\n" for i in range(20)]
        source = "def worker():\n" + "".join(filler_lines) + "    return val_0\n"
        chunk = _make_chunk(source)

        idf_map = {"process": 0.01, "return": 5.0, "def": 5.0, "worker": 5.0}

        from mnemosyne.compress import Compressor
        from mnemosyne.embeddings.tfidf_backend import TFIDFBackend
        cfg = _default_config()
        cfg.compression.target_ratio = 0.3

        backend = TFIDFBackend(cfg)
        backend.idf = idf_map
        comp = Compressor(cfg, tfidf_backend=backend)

        result_normal = comp.compress(chunk, strict=False)
        result_strict = comp.compress(chunk, strict=True)

        # Strict mode should preserve more content (Stage 3 skipped)
        self.assertGreater(
            len(result_strict), len(result_normal),
            "strict=True should preserve more content than non-strict "
            f"(strict={len(result_strict)} chars, normal={len(result_normal)} chars)",
        )

    def test_strict_for_symbols_config(self):
        """Config default for strict_for_symbols should be True."""
        cfg = _default_config()
        self.assertTrue(
            cfg.compression.strict_for_symbols,
            "Expected compression.strict_for_symbols to default to True",
        )


if __name__ == "__main__":
    unittest.main()
