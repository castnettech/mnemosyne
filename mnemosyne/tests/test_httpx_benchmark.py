# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
M-GATE -- httpx benchmark regression gate.

Clones httpx 0.28.1 (cached), indexes it with Mnemosyne, runs 6 retrieval
questions, and asserts that key metrics do not regress below the established
baseline.

Run with:
    python3 -m pytest mnemosyne/tests/test_httpx_benchmark.py -m benchmark -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

from mnemosyne.tests.benchmark_suite import BenchmarkSuite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HTTPX_TAG = "0.28.1"
HTTPX_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), f"mnemosyne_benchmark_httpx_{HTTPX_TAG}"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def httpx_root():
    """Clone httpx 0.28.1 to a cached temp directory."""
    if os.path.isdir(HTTPX_CACHE_DIR) and os.path.isfile(
        os.path.join(HTTPX_CACHE_DIR, "pyproject.toml")
    ):
        return HTTPX_CACHE_DIR

    # Clone fresh
    if os.path.exists(HTTPX_CACHE_DIR):
        shutil.rmtree(HTTPX_CACHE_DIR)
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--branch", HTTPX_TAG,
            "https://github.com/encode/httpx.git", HTTPX_CACHE_DIR,
        ],
        check=True,
        capture_output=True,
    )
    return HTTPX_CACHE_DIR


@pytest.fixture(scope="session")
def httpx_benchmark_results(httpx_root):
    """Run all 6 httpx questions and return per-question results."""
    questions_file = os.path.join(
        os.path.dirname(__file__), "benchmark_questions", "httpx.json"
    )
    questions = BenchmarkSuite._load_questions(questions_file)

    # Setup mnemosyne on the httpx project
    suite = BenchmarkSuite([])
    engine, config = suite._setup_mnemosyne(httpx_root)

    # Run each question
    results = {}
    for q in questions:
        qr = suite._run_question(engine, q)
        results[q.id] = qr

    return results


# ---------------------------------------------------------------------------
# Regression gate tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestHttpxRegressionGate:
    """Baseline assertions -- no regression allowed."""

    def test_q1_relevant_at_5(self, httpx_benchmark_results):
        assert httpx_benchmark_results["Q1"].relevant_at_5 >= 5

    def test_q3_relevant_at_5(self, httpx_benchmark_results):
        assert httpx_benchmark_results["Q3"].relevant_at_5 >= 5

    def test_q4_relevant_at_5(self, httpx_benchmark_results):
        assert httpx_benchmark_results["Q4"].relevant_at_5 >= 5

    def test_overall_hit_at_3(self, httpx_benchmark_results):
        r = httpx_benchmark_results
        hit3 = sum(qr.hit_at_3 for qr in r.values()) / len(r)
        assert hit3 >= 0.666

    def test_overall_mrr_at_10(self, httpx_benchmark_results):
        r = httpx_benchmark_results
        mrr = sum(qr.mrr_at_10 for qr in r.values()) / len(r)
        assert mrr >= 0.597
