# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mnemosyne Benchmark — policylens (PrivacyPeep) edition.

Measures token reduction, retrieval precision, compression ratios, query
speed, and storage overhead for the Mnemosyne context engine against the
policylens project.

Usage:
    python -m mnemosyne.tests.benchmark --project-root /path/to/policylens
    python /path/to/benchmark.py --project-root /path/to/policylens [--budget 4000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path bootstrap — make sure the mnemosyne package is importable when this
# script is run directly (i.e. not as part of the package).
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

def get_policylens_questions() -> list[dict]:
    """
    Return the hardcoded policylens (PrivacyPeep) benchmark question set.

    Each question dict has keys: ``id``, ``question``, ``challenge``,
    ``ground_truth``.  This is the original 10-question set used for
    quick single-project benchmarking.

    Returns:
        List of question dicts.
    """
    return [
        {
            "id": "Q01",
            "question": "What does isNegated do and how does it detect negation?",
            "challenge": "Symbol lookup",
            "ground_truth": ["public/js/nlp.js"],
        },
        {
            "id": "Q02",
            "question": "How does the scoring pipeline work from findings to final grade?",
            "challenge": "Architecture",
            "ground_truth": ["public/js/scorer.js", "public/js/analyzer.js", "public/js/patterns.js"],
        },
        {
            "id": "Q03",
            "question": "How do patterns connect to the analyzer? What is the data flow from pattern definitions to findings?",
            "challenge": "Cross-file",
            "ground_truth": ["public/js/patterns.js", "public/js/analyzer.js", "public/js/utils.js"],
        },
        {
            "id": "Q04",
            "question": "What privacy frameworks are supported in the pattern detection like GDPR CCPA SOC2?",
            "challenge": "Domain scan",
            "ground_truth": ["public/js/patterns.js", "public/js/utils.js"],
        },
        {
            "id": "Q05",
            "question": "Where is negation detection handled and what are the known edge cases?",
            "challenge": "Bug investigation",
            "ground_truth": ["public/js/nlp.js", "public/js/analyzer.js"],
        },
        {
            "id": "Q06",
            "question": "How does policy comparison work?",
            "challenge": "Feature deep-dive",
            "ground_truth": ["public/js/comparator.js", "public/js/analyzer.js"],
        },
        {
            "id": "Q07",
            "question": "What dependencies does the project use and how is it served?",
            "challenge": "Config/setup",
            "ground_truth": ["package.json", "serve.sh", "public/index.html"],
        },
        {
            "id": "Q08",
            "question": "What test files exist and what do they cover?",
            "challenge": "Test coverage",
            "ground_truth": [
                "tests/engine.test.js",
                "tests/extractor.test.js",
                "tests/real-policy-test.js",
                "tests/top20-policy-test.js",
            ],
        },
        {
            "id": "Q09",
            "question": "How does a policy URL get analyzed end to end?",
            "challenge": "Full pipeline",
            "ground_truth": [
                "_score-runner.js",
                "public/js/extractor.js",
                "public/js/analyzer.js",
                "public/js/app.js",
            ],
        },
        {
            "id": "Q10",
            "question": "What categories does the privacy score include and what are their weights?",
            "challenge": "Constant lookup",
            "ground_truth": ["public/js/patterns.js", "public/js/scorer.js", "public/js/analyzer.js"],
        },
    ]


# Module-level constant — backward compatibility with code that references
# ``benchmark.BENCHMARK_QUESTIONS`` directly.
BENCHMARK_QUESTIONS = get_policylens_questions()

COMPRESSION_TARGETS = [
    "public/js/patterns.js",
    "public/js/app.js",
    "public/js/analyzer.js",
    "public/js/scorer.js",
    "public/js/nlp.js",
    "_score-runner.js",
]

# Directories to skip when walking the project tree
_SKIP_DIRS = {".mnemosyne", "node_modules", ".git", "mnemosyne", "__pycache__"}


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class MnemosyneBenchmark:
    """End-to-end benchmark harness for Mnemosyne against a target project."""

    def __init__(
        self,
        project_root: str,
        budget: int = 4000,
        questions: list[dict] | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self.budget = budget
        self.mnemosyne_dir = os.path.join(self.project_root, ".mnemosyne")
        self.questions = questions if questions is not None else BENCHMARK_QUESTIONS

        # Initialised in setup()
        self.conn = None
        self.store = None
        self.config = None
        self.tfidf = None
        self.engine = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> dict:
        """Initialize Mnemosyne and ingest the project. Returns ingest stats."""
        from mnemosyne.config import Config
        from mnemosyne.schema import open_store
        from mnemosyne.store import Store
        from mnemosyne.bloom import BloomFilter
        from mnemosyne.audit import AuditLog
        from mnemosyne.embeddings import get_backend
        from mnemosyne.ingest import Ingester

        # 1. Delete existing DB and bloom for a clean benchmark run
        db_path = os.path.join(self.mnemosyne_dir, "mnemosyne.db")
        bloom_path_init = os.path.join(self.mnemosyne_dir, "bloom.bin")
        for f in (db_path, bloom_path_init):
            if os.path.isfile(f):
                os.remove(f)
        os.makedirs(self.mnemosyne_dir, exist_ok=True)

        # 2. Build config — only add universally-correct ignore patterns.
        #    NO project-specific overrides. The engine must rank properly
        #    using its own signals (density, symbol matching, TF-IDF).
        self.config = Config(root=self.project_root)
        patterns = list(self.config.general.ignore_patterns)
        # Exclude the mnemosyne engine's own source when it lives inside
        # the project directory, and marketing assets (not source code).
        for ignore in ("mnemosyne", "marketing"):
            if ignore not in patterns:
                patterns.append(ignore)
        self.config.general.ignore_patterns = patterns

        # Override embedding min_df so unique-to-one-file terms are kept
        self.config.embedding.tfidf_min_df = 1

        # 3. Open connection and initialise schema
        self.conn = open_store(self.mnemosyne_dir)

        # 4. Build auxiliary components
        self.store = Store(self.conn)
        bloom = BloomFilter()

        bloom_path = os.path.join(self.mnemosyne_dir, "bloom.bin")
        if os.path.isfile(bloom_path):
            try:
                bloom = BloomFilter.load(bloom_path)
            except Exception:
                bloom = BloomFilter()

        self.tfidf = get_backend(self.config, self.store)

        audit = AuditLog(os.path.join(self.mnemosyne_dir, "audit.log"))

        # 5. Ingest (TF-IDF is the primary ingest backend)
        ingester = Ingester(
            project_root=self.project_root,
            config=self.config,
            store=self.store,
            bloom=bloom,
            tfidf_backend=self.tfidf,
            audit=audit,
        )
        stats = ingester.ingest(full=True)

        # Persist bloom filter
        try:
            bloom.save(bloom_path)
        except Exception:
            pass

        # 6. Wire up RetrievalEngine
        self._rebuild_engine()

        return stats

    # ------------------------------------------------------------------
    # Benchmark methods
    # ------------------------------------------------------------------

    def measure_token_reduction(self) -> list[dict]:
        """For each question: count raw tokens vs mnemosyne tokens."""
        from mnemosyne.models import estimate_tokens

        results = []
        for q in self.questions:
            raw_tokens = 0
            for f in q["ground_truth"]:
                path = os.path.join(self.project_root, f)
                if os.path.isfile(path):
                    with open(path, "r", errors="replace") as fh:
                        raw_tokens += estimate_tokens(fh.read())

            query_results = self.engine.query(q["question"], budget=self.budget)
            mn_tokens = sum(
                estimate_tokens(r.chunk.compressed or r.chunk.content)
                for r in query_results
            )

            reduction = (1 - mn_tokens / max(1, raw_tokens)) * 100 if raw_tokens > 0 else 0.0
            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "challenge": q["challenge"],
                    "raw_tokens": raw_tokens,
                    "mnemosyne_tokens": mn_tokens,
                    "reduction_pct": reduction,
                }
            )
        return results

    def measure_retrieval_precision(self) -> list[dict]:
        """For each question: check if retrieved files match ground truth."""
        results = []
        for q in self.questions:
            query_results = self.engine.query(q["question"], budget=self.budget)

            # Collect unique relative file paths from results
            retrieved_files: set[str] = set()
            for r in query_results:
                fp = r.file_path.replace("\\", "/")
                retrieved_files.add(fp)

            ground_truth = set(q["ground_truth"])
            hits = retrieved_files & ground_truth
            precision = len(hits) / len(retrieved_files) if retrieved_files else 0.0
            recall = len(hits) / len(ground_truth) if ground_truth else 1.0

            results.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "challenge": q["challenge"],
                    "ground_truth_count": len(ground_truth),
                    "retrieved_count": len(retrieved_files),
                    "hits": len(hits),
                    "precision": precision,
                    "recall": recall,
                    "retrieved_files": sorted(retrieved_files),
                    "ground_truth_files": sorted(ground_truth),
                }
            )
        return results

    def measure_compression(self) -> list[dict]:
        """Compress key files and report ratios."""
        from mnemosyne.compress import Compressor
        from mnemosyne.models import Chunk, estimate_tokens

        compressor = Compressor(self.config, self.tfidf)
        results = []

        for rel_path in COMPRESSION_TARGETS:
            abs_path = os.path.join(self.project_root, rel_path)
            if not os.path.isfile(abs_path):
                continue
            with open(abs_path, "r", errors="replace") as f:
                content = f.read()

            raw_tokens = estimate_tokens(content)

            chunk = Chunk(
                chunk_id=None,
                file_id=0,
                content_hash="",
                chunk_type="generic",
                line_start=1,
                line_end=content.count("\n") + 1,
                token_count=raw_tokens,
                content=content,
            )
            compressed = compressor.compress(chunk)
            comp_tokens = estimate_tokens(compressed)

            results.append(
                {
                    "file": rel_path,
                    "filename": os.path.basename(rel_path),
                    "raw_tokens": raw_tokens,
                    "compressed_tokens": comp_tokens,
                    "ratio_pct": (comp_tokens / max(1, raw_tokens)) * 100,
                }
            )
        return results

    def measure_speed(self) -> dict:
        """Measure ingest time, query time, and baseline sequential read time."""
        from mnemosyne.bloom import BloomFilter
        from mnemosyne.audit import AuditLog
        from mnemosyne.ingest import Ingester

        # Time a full re-ingest
        t0 = time.perf_counter()
        bloom = BloomFilter()
        audit = AuditLog(os.path.join(self.mnemosyne_dir, "audit.log"))
        ingester = Ingester(
            project_root=self.project_root,
            config=self.config,
            store=self.store,
            bloom=bloom,
            tfidf_backend=self.tfidf,
            audit=audit,
        )
        ingester.ingest(full=True)
        ingest_time = time.perf_counter() - t0

        # Rebuild retrieval engine so it sees the freshly re-ingested data
        self._rebuild_engine()

        # Query timing: run each question 3 times, take median
        query_times = []
        for q in self.questions:
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                self.engine.query(q["question"], budget=self.budget)
                times.append(time.perf_counter() - t0)
            times.sort()
            query_times.append(times[1])  # median of 3

        avg_query_ms = (sum(query_times) / len(query_times)) * 1000

        # Baseline: sequential read of all source files
        source_files = self._get_source_files()
        t0 = time.perf_counter()
        for f in source_files:
            try:
                with open(f, "r", errors="replace") as fh:
                    _ = fh.read()
            except Exception:
                pass
        seq_read_time = time.perf_counter() - t0

        return {
            "ingest_seconds": ingest_time,
            "avg_query_ms": avg_query_ms,
            "sequential_read_seconds": seq_read_time,
        }

    def measure_storage(self) -> dict:
        """Compare raw source size vs .mnemosyne/ size."""
        source_files = self._get_source_files()
        raw_size = sum(os.path.getsize(f) for f in source_files if os.path.isfile(f))

        db_size = 0
        if os.path.isdir(self.mnemosyne_dir):
            for fname in os.listdir(self.mnemosyne_dir):
                fp = os.path.join(self.mnemosyne_dir, fname)
                if os.path.isfile(fp):
                    db_size += os.path.getsize(fp)

        return {
            "raw_bytes": raw_size,
            "mnemosyne_bytes": db_size,
            "overhead_ratio": db_size / max(1, raw_size),
        }

    # ------------------------------------------------------------------
    # Aggregate runner
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Run all benchmarks and return a results dict."""
        print("  [1/5] Measuring token reduction...")
        token_reduction = self.measure_token_reduction()

        print("  [2/5] Measuring retrieval precision...")
        precision = self.measure_retrieval_precision()

        print("  [3/5] Measuring compression...")
        compression = self.measure_compression()

        print("  [4/5] Measuring speed...")
        speed = self.measure_speed()

        print("  [5/5] Measuring storage...")
        storage = self.measure_storage()

        return {
            "token_reduction": token_reduction,
            "precision": precision,
            "compression": compression,
            "speed": speed,
            "storage": storage,
        }

    # ------------------------------------------------------------------
    # Report formatter
    # ------------------------------------------------------------------

    def format_report(self, results: dict) -> str:
        """Format all benchmark results into an aligned, human-readable report."""
        lines: list[str] = []

        sep = "=" * 72

        lines.append(sep)
        lines.append("  MNEMOSYNE BENCHMARK REPORT — policylens (PrivacyPeep)")
        lines.append(f"  Project root : {self.project_root}")
        lines.append(f"  Token budget : {self.budget:,}")
        lines.append(sep)

        # ------------------------------------------------------------------
        # 1. Token Reduction
        # ------------------------------------------------------------------
        lines.append("")
        lines.append("TOKEN REDUCTION")
        lines.append("-" * 72)
        lines.append(
            f"  {'ID':<5}  {'Challenge':<20}  {'Raw':>7}  {'Mnem':>7}  {'Reduction':>10}"
        )
        lines.append(f"  {'-'*5}  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*10}")

        total_raw = 0
        total_mn = 0
        for r in results["token_reduction"]:
            total_raw += r["raw_tokens"]
            total_mn += r["mnemosyne_tokens"]
            lines.append(
                f"  {r['id']:<5}  {r['challenge']:<20}  {r['raw_tokens']:>7,}"
                f"  {r['mnemosyne_tokens']:>7,}  {r['reduction_pct']:>9.1f}%"
            )

        overall_reduction = (1 - total_mn / max(1, total_raw)) * 100
        lines.append(f"  {'-'*5}  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*10}")
        lines.append(
            f"  {'TOTAL':<5}  {'':<20}  {total_raw:>7,}  {total_mn:>7,}  {overall_reduction:>9.1f}%"
        )

        # ------------------------------------------------------------------
        # 2. Retrieval Precision
        # ------------------------------------------------------------------
        lines.append("")
        lines.append("RETRIEVAL PRECISION")
        lines.append("-" * 72)
        lines.append(
            f"  {'ID':<5}  {'Challenge':<20}  {'GT':>4}  {'Retr':>4}  {'Hits':>4}"
            f"  {'Prec':>6}  {'Recall':>6}"
        )
        lines.append(f"  {'-'*5}  {'-'*20}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*6}")

        total_hits = 0
        total_gt = 0
        total_retrieved = 0
        for r in results["precision"]:
            total_hits += r["hits"]
            total_gt += r["ground_truth_count"]
            total_retrieved += r["retrieved_count"]
            lines.append(
                f"  {r['id']:<5}  {r['challenge']:<20}  {r['ground_truth_count']:>4}"
                f"  {r['retrieved_count']:>4}  {r['hits']:>4}"
                f"  {r['precision']:>5.1%}  {r['recall']:>5.1%}"
            )

        avg_precision = (
            sum(r["precision"] for r in results["precision"]) / len(results["precision"])
            if results["precision"]
            else 0.0
        )
        avg_recall = (
            sum(r["recall"] for r in results["precision"]) / len(results["precision"])
            if results["precision"]
            else 0.0
        )
        lines.append(f"  {'-'*5}  {'-'*20}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*6}")
        lines.append(
            f"  {'AVG':<5}  {'':<20}  {'':<4}  {'':<4}  {'':<4}"
            f"  {avg_precision:>5.1%}  {avg_recall:>5.1%}"
        )

        # Per-question detail
        lines.append("")
        lines.append("  Retrieved vs Ground Truth (detail):")
        for r in results["precision"]:
            lines.append(f"    {r['id']} — {r['challenge']}")
            lines.append(f"      Ground truth : {', '.join(r['ground_truth_files'])}")
            lines.append(f"      Retrieved    : {', '.join(r['retrieved_files']) or '(none)'}")

        # ------------------------------------------------------------------
        # 3. Compression
        # ------------------------------------------------------------------
        lines.append("")
        lines.append("COMPRESSION RATIOS")
        lines.append("-" * 72)
        lines.append(
            f"  {'File':<30}  {'Raw':>7}  {'Compressed':>10}  {'Ratio':>7}"
        )
        lines.append(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*7}")

        for r in results["compression"]:
            lines.append(
                f"  {r['filename']:<30}  {r['raw_tokens']:>7,}  {r['compressed_tokens']:>10,}"
                f"  {r['ratio_pct']:>6.1f}%"
            )

        if results["compression"]:
            avg_ratio = sum(r["ratio_pct"] for r in results["compression"]) / len(
                results["compression"]
            )
            lines.append(f"  {'-'*30}  {'-'*7}  {'-'*10}  {'-'*7}")
            lines.append(f"  {'AVERAGE':<30}  {'':>7}  {'':>10}  {avg_ratio:>6.1f}%")

        # ------------------------------------------------------------------
        # 4. Speed
        # ------------------------------------------------------------------
        speed = results["speed"]
        lines.append("")
        lines.append("SPEED")
        lines.append("-" * 72)
        lines.append(f"  Ingest time (full re-ingest) : {speed['ingest_seconds']:.3f}s")
        lines.append(f"  Avg query time (median × 3)  : {speed['avg_query_ms']:.1f}ms")
        lines.append(f"  Baseline seq read (all files): {speed['sequential_read_seconds']:.3f}s")

        # ------------------------------------------------------------------
        # 5. Storage
        # ------------------------------------------------------------------
        storage = results["storage"]
        lines.append("")
        lines.append("STORAGE")
        lines.append("-" * 72)
        raw_kb = storage["raw_bytes"] / 1024
        db_kb = storage["mnemosyne_bytes"] / 1024
        lines.append(f"  Raw source files  : {raw_kb:>9.1f} KB  ({storage['raw_bytes']:,} bytes)")
        lines.append(f"  .mnemosyne/ total : {db_kb:>9.1f} KB  ({storage['mnemosyne_bytes']:,} bytes)")
        lines.append(f"  Overhead ratio    : {storage['overhead_ratio']:.2f}x")

        lines.append("")
        lines.append(sep)
        lines.append("  END OF REPORT")
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_source_files(self) -> list[str]:
        """Return all text source files under project_root, skipping ignored dirs."""
        source_files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            # Prune skip directories in-place so os.walk skips them entirely
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

            for fname in filenames:
                source_files.append(os.path.join(dirpath, fname))

        return source_files

    def _rebuild_engine(self) -> None:
        """Rebuild the RetrievalEngine (e.g. after a re-ingest)."""
        from mnemosyne.analytics import Analytics
        from mnemosyne.prefetch import Prefetcher
        from mnemosyne.retrieval import RetrievalEngine

        analytics = Analytics(self.store, self.config)
        analytics.start_session()
        prefetcher = Prefetcher(self.store)

        self.engine = RetrievalEngine(
            store=self.store,
            tfidf_backend=self.tfidf,
            config=self.config,
            analytics=analytics,
            prefetcher=prefetcher,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mnemosyne Benchmark — measures token reduction, retrieval precision, "
        "compression, speed, and storage overhead against a target project."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the project to benchmark (e.g. /path/to/policylens)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=4000,
        help="Token budget passed to each query (default: 4000)",
    )
    args = parser.parse_args()

    bench = MnemosyneBenchmark(args.project_root, args.budget)

    print("Setting up Mnemosyne...")
    ingest_stats = bench.setup()
    print(
        f"  Ingest complete — {ingest_stats.get('files_indexed', '?')} files indexed, "
        f"{ingest_stats.get('chunks_added', '?')} chunks added "
        f"({ingest_stats.get('elapsed_seconds', 0):.2f}s)"
    )

    print("Running benchmarks...")
    results = bench.run_all()

    report = bench.format_report(results)
    print(report)


if __name__ == "__main__":
    main()
