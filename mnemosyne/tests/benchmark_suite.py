# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
MnemoSync Benchmark Suite -- multi-project, chunk-level precision measurement.

Loads question sets from JSON files in ``tests/benchmark_questions/``, runs each
question against its target project's Mnemosyne index, and computes file-level
AND chunk-level precision/recall.  Produces a per-question detail report and
aggregate metrics.

Usage:
    cd /path/to/mnemosyne-repo
    python3 -m mnemosyne.tests.benchmark_suite
    python3 -m mnemosyne.tests.benchmark_suite --questions-dir /path/to/questions
    python3 -m mnemosyne.tests.benchmark_suite --budget 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Path bootstrap -- importable when run directly
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkQuestion:
    """A single retrieval-quality question with ground truth."""

    id: str
    question: str
    category: str  # e.g., "symbol_lookup", "cross_file", "config_setup"
    ground_truth_files: list[str]  # required -- file-level ground truth
    ground_truth_chunks: list[dict] | None = None  # optional -- chunk-level
    # Each chunk dict: {"file": str, "symbol_name": str}
    #              or: {"file": str, "line_start": int, "line_end": int}
    budget: int = 4000
    difficulty: str = "medium"  # easy / medium / hard


@dataclass
class BenchmarkProject:
    """A project to benchmark with its question set."""

    name: str
    root: str  # absolute path to project root
    questions_file: str  # path to JSON questions file


@dataclass
class QuestionResult:
    """Metrics for a single question."""

    question: BenchmarkQuestion
    file_precision: float
    file_recall: float
    chunk_precision: float
    chunk_recall: float
    retrieved_files: list[str]
    retrieved_chunks: list[dict]  # [{"file": str, "symbol_name": str|None, ...}]
    elapsed_ms: float
    hit_at_3: int = 0
    relevant_at_5: int = 0
    mrr_at_10: float = 0.0


@dataclass
class ProjectResult:
    """Aggregate results for one project."""

    name: str
    question_results: list[QuestionResult]
    avg_file_precision: float
    avg_file_recall: float
    avg_chunk_precision: float
    avg_chunk_recall: float


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """
    Multi-project benchmark runner with file-level and chunk-level metrics.

    Initialise with a list of :class:`BenchmarkProject` descriptors.  Call
    :meth:`run_all` to ingest each project, run its questions, and collect
    aggregate metrics.
    """

    def __init__(self, projects: list[BenchmarkProject]) -> None:
        self.projects = projects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """
        Run all projects and return aggregate results.

        Returns:
            Dict with keys ``projects`` (list of ProjectResult) and
            ``aggregate`` (global averages across all projects).
        """
        all_project_results: list[ProjectResult] = []

        for project in self.projects:
            pr = self._run_project(project)
            all_project_results.append(pr)

        # Compute global aggregates across all projects
        all_qr: list[QuestionResult] = []
        for pr in all_project_results:
            all_qr.extend(pr.question_results)

        n = len(all_qr) or 1
        aggregate = {
            "file_precision": sum(q.file_precision for q in all_qr) / n,
            "file_recall": sum(q.file_recall for q in all_qr) / n,
            "chunk_precision": sum(q.chunk_precision for q in all_qr) / n,
            "chunk_recall": sum(q.chunk_recall for q in all_qr) / n,
            "total_questions": len(all_qr),
        }

        return {
            "projects": all_project_results,
            "aggregate": aggregate,
        }

    # ------------------------------------------------------------------
    # Measurement methods (public for testability)
    # ------------------------------------------------------------------

    @staticmethod
    def measure_file_precision(
        results: list,  # list[QueryResult]
        ground_truth_files: list[str],
    ) -> tuple[float, float]:
        """
        Compute file-level precision and recall.

        Args:
            results:             List of QueryResult from engine.query().
            ground_truth_files:  Expected file paths (relative).

        Returns:
            ``(precision, recall)`` as floats in [0.0, 1.0].
        """
        retrieved_files: set[str] = set()
        for r in results:
            fp = r.file_path.replace("\\", "/")
            retrieved_files.add(fp)

        ground_truth = set(ground_truth_files)
        hits = retrieved_files & ground_truth

        precision = len(hits) / len(retrieved_files) if retrieved_files else 0.0
        recall = len(hits) / len(ground_truth) if ground_truth else 1.0

        return precision, recall

    @staticmethod
    def measure_chunk_precision(
        results: list,  # list[QueryResult]
        ground_truth_chunks: list[dict] | None,
    ) -> tuple[float, float]:
        """
        Compute chunk-level precision and recall.

        Ground truth chunks are identified by:
        - ``symbol_name``: matched against ``result.chunk.symbol_name``
        - ``line_start``/``line_end``: checked for overlap with result chunk lines

        Each ground truth chunk also includes ``file`` to disambiguate symbols
        with the same name across files.

        Args:
            results:              List of QueryResult from engine.query().
            ground_truth_chunks:  List of dicts with ground truth chunk info.

        Returns:
            ``(precision, recall)`` as floats in [0.0, 1.0].
            Returns (0.0, 0.0) if ground_truth_chunks is None or empty.
        """
        if not ground_truth_chunks:
            return 0.0, 0.0

        # Build a list of retrieved chunk descriptors for matching
        retrieved: list[dict] = []
        for r in results:
            fp = r.file_path.replace("\\", "/")
            retrieved.append({
                "file": fp,
                "symbol_name": r.chunk.symbol_name,
                "line_start": r.chunk.line_start,
                "line_end": r.chunk.line_end,
            })

        # Track which ground truth chunks were matched
        gt_matched: list[bool] = [False] * len(ground_truth_chunks)
        # Track which retrieved chunks match at least one ground truth
        retr_matched: list[bool] = [False] * len(retrieved)

        for gi, gt in enumerate(ground_truth_chunks):
            gt_file = gt.get("file", "")
            gt_symbol = gt.get("symbol_name")
            gt_line_start = gt.get("line_start")
            gt_line_end = gt.get("line_end")

            for ri, retr in enumerate(retrieved):
                # File must match (allow gt_file to be a suffix of retrieved path)
                retr_file = retr["file"]
                if not (retr_file == gt_file or retr_file.endswith("/" + gt_file)):
                    continue

                matched = False

                # Symbol name match
                if gt_symbol is not None and retr["symbol_name"] is not None:
                    # Exact match or suffix match (e.g., "ARCCache._evict"
                    # matches a chunk whose symbol_name is "_evict" if the
                    # file already matched)
                    r_sym = retr["symbol_name"]
                    if (
                        r_sym == gt_symbol
                        or gt_symbol.endswith("." + r_sym)
                        or r_sym.endswith("." + gt_symbol)
                        or gt_symbol.split(".")[-1] == r_sym
                        or r_sym.split(".")[-1] == gt_symbol.split(".")[-1]
                    ):
                        matched = True

                # Line range overlap match
                if (
                    not matched
                    and gt_line_start is not None
                    and gt_line_end is not None
                ):
                    r_start = retr["line_start"]
                    r_end = retr["line_end"]
                    # Check for any overlap
                    if r_start <= gt_line_end and r_end >= gt_line_start:
                        # Require at least 50% overlap of the ground truth range
                        overlap_start = max(r_start, gt_line_start)
                        overlap_end = min(r_end, gt_line_end)
                        overlap_lines = max(0, overlap_end - overlap_start + 1)
                        gt_lines = gt_line_end - gt_line_start + 1
                        if overlap_lines >= gt_lines * 0.5:
                            matched = True

                if matched:
                    gt_matched[gi] = True
                    retr_matched[ri] = True

        total_retrieved = len(retrieved)
        total_gt = len(ground_truth_chunks)
        matched_retrieved = sum(retr_matched)
        matched_gt = sum(gt_matched)

        precision = matched_retrieved / total_retrieved if total_retrieved else 0.0
        recall = matched_gt / total_gt if total_gt else 0.0

        return precision, recall

    # ------------------------------------------------------------------
    # Retrieval ranking metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hit_at_k(results: list, ground_truth_files: list[str], k: int = 3) -> int:
        """Return 1 if any of the top-k results match a gold file, else 0."""
        gt = set(ground_truth_files)
        for r in results[:k]:
            fp = r.file_path.replace("\\", "/")
            if fp in gt:
                return 1
        return 0

    @staticmethod
    def compute_relevant_at_k(results: list, ground_truth_files: list[str], k: int = 5) -> int:
        """Count how many of the top-k results match any gold file."""
        gt = set(ground_truth_files)
        count = 0
        for r in results[:k]:
            fp = r.file_path.replace("\\", "/")
            if fp in gt:
                count += 1
        return count

    @staticmethod
    def compute_mrr_at_k(results: list, ground_truth_files: list[str], k: int = 10) -> float:
        """Reciprocal rank of the first gold file in top-k results. 0.0 if none found."""
        gt = set(ground_truth_files)
        for i, r in enumerate(results[:k]):
            fp = r.file_path.replace("\\", "/")
            if fp in gt:
                return 1.0 / (i + 1)
        return 0.0

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _run_project(self, project: BenchmarkProject) -> ProjectResult:
        """Ingest and benchmark a single project."""
        print(f"\n--- Project: {project.name} ---")

        # Load questions
        questions = self._load_questions(project.questions_file)
        print(f"  Loaded {len(questions)} questions")

        # Set up Mnemosyne for this project
        engine, config = self._setup_mnemosyne(project.root)
        print("  Mnemosyne initialised")

        # Run each question
        question_results: list[QuestionResult] = []
        for q in questions:
            qr = self._run_question(engine, q)
            question_results.append(qr)

        # Compute project-level averages
        n = len(question_results) or 1
        avg_fp = sum(qr.file_precision for qr in question_results) / n
        avg_fr = sum(qr.file_recall for qr in question_results) / n
        avg_cp = sum(qr.chunk_precision for qr in question_results) / n
        avg_cr = sum(qr.chunk_recall for qr in question_results) / n

        return ProjectResult(
            name=project.name,
            question_results=question_results,
            avg_file_precision=avg_fp,
            avg_file_recall=avg_fr,
            avg_chunk_precision=avg_cp,
            avg_chunk_recall=avg_cr,
        )

    def _run_question(self, engine, q: BenchmarkQuestion) -> QuestionResult:
        """Run a single question and measure all metrics."""
        t0 = time.perf_counter()
        results = engine.query(q.question, budget=q.budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # File-level metrics
        fp, fr = self.measure_file_precision(results, q.ground_truth_files)

        # Chunk-level metrics
        cp, cr = self.measure_chunk_precision(results, q.ground_truth_chunks)

        # Retrieval ranking metrics
        hit3 = self.compute_hit_at_k(results, q.ground_truth_files, k=3)
        rel5 = self.compute_relevant_at_k(results, q.ground_truth_files, k=5)
        mrr10 = self.compute_mrr_at_k(results, q.ground_truth_files, k=10)

        # Collect retrieved info for reporting
        retrieved_files = sorted(set(
            r.file_path.replace("\\", "/") for r in results
        ))
        retrieved_chunks = [
            {
                "file": r.file_path.replace("\\", "/"),
                "symbol_name": r.chunk.symbol_name,
                "line_start": r.chunk.line_start,
                "line_end": r.chunk.line_end,
                "chunk_type": r.chunk.chunk_type,
            }
            for r in results
        ]

        return QuestionResult(
            question=q,
            file_precision=fp,
            file_recall=fr,
            chunk_precision=cp,
            chunk_recall=cr,
            retrieved_files=retrieved_files,
            retrieved_chunks=retrieved_chunks,
            elapsed_ms=elapsed_ms,
            hit_at_3=hit3,
            relevant_at_5=rel5,
            mrr_at_10=mrr10,
        )

    def _setup_mnemosyne(self, project_root: str) -> tuple:
        """
        Initialise Mnemosyne for a project and return (engine, config).

        Performs a clean ingest for reproducible benchmark results.
        """
        from mnemosyne.config import Config
        from mnemosyne.schema import open_store
        from mnemosyne.store import Store
        from mnemosyne.bloom import BloomFilter
        from mnemosyne.audit import AuditLog
        from mnemosyne.embeddings import get_backend
        from mnemosyne.ingest import Ingester
        from mnemosyne.analytics import Analytics
        from mnemosyne.prefetch import Prefetcher
        from mnemosyne.retrieval import RetrievalEngine

        project_root = os.path.abspath(project_root)
        mnemosyne_dir = os.path.join(project_root, ".mnemosyne")

        # Clean slate for reproducibility
        db_path = os.path.join(mnemosyne_dir, "mnemosyne.db")
        bloom_path = os.path.join(mnemosyne_dir, "bloom.bin")
        for f in (db_path, bloom_path):
            if os.path.isfile(f):
                os.remove(f)
        os.makedirs(mnemosyne_dir, exist_ok=True)

        # Build config
        config = Config(root=project_root)
        patterns = list(config.general.ignore_patterns)
        # Exclude non-source noise from benchmark projects
        for ignore in ("marketing", ".pytest_cache"):
            if ignore not in patterns:
                patterns.append(ignore)
        config.general.ignore_patterns = patterns
        config.embedding.tfidf_min_df = 1

        # Open DB and initialise components
        conn = open_store(mnemosyne_dir)
        store = Store(conn)
        bloom = BloomFilter()
        tfidf = get_backend(config, store)
        audit = AuditLog(os.path.join(mnemosyne_dir, "audit.log"))

        # Dense backend (optional -- requires onnxruntime)
        dense_backend = None
        try:
            from mnemosyne.embeddings.dense_backend import DenseBackend
            if DenseBackend.is_available():
                config.embedding.dense_model = "minilm-l6-code"
                dense_backend = DenseBackend(
                    config, store,
                    model_dir=os.path.join(mnemosyne_dir, "models"),
                )
        except Exception:
            pass

        # Ingest
        ingester = Ingester(
            project_root=project_root,
            config=config,
            store=store,
            bloom=bloom,
            tfidf_backend=tfidf,
            audit=audit,
            dense_backend=dense_backend,
        )
        stats = ingester.ingest(full=True)
        print(
            f"  Ingest: {stats.get('files_indexed', '?')} files, "
            f"{stats.get('chunks_added', '?')} chunks "
            f"({stats.get('elapsed_seconds', 0):.2f}s)"
        )

        # Build retrieval engine
        analytics = Analytics(store, config)
        analytics.start_session()
        prefetcher = Prefetcher(store)

        engine = RetrievalEngine(
            store=store,
            tfidf_backend=tfidf,
            config=config,
            analytics=analytics,
            prefetcher=prefetcher,
            dense_backend=dense_backend,
        )

        return engine, config

    @staticmethod
    def _load_questions(questions_file: str) -> list[BenchmarkQuestion]:
        """Load questions from a JSON file."""
        with open(questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions: list[BenchmarkQuestion] = []
        for q in data["questions"]:
            questions.append(BenchmarkQuestion(
                id=q["id"],
                question=q["question"],
                category=q.get("category", "general"),
                ground_truth_files=q["ground_truth_files"],
                ground_truth_chunks=q.get("ground_truth_chunks"),
                budget=q.get("budget", 4000),
                difficulty=q.get("difficulty", "medium"),
            ))

        return questions

    # ------------------------------------------------------------------
    # Report formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(results: dict) -> str:
        """Format all benchmark results into a human-readable report."""
        lines: list[str] = []
        sep = "=" * 76

        lines.append(sep)
        lines.append("  MNEMOSYNC BENCHMARK SUITE")
        lines.append(sep)

        for pr in results["projects"]:
            lines.append("")
            lines.append(f"Project: {pr.name} ({len(pr.question_results)} questions)")
            lines.append("-" * 76)

            # Header
            lines.append(
                f"  {'ID':<5} [{'Diff':^6}] {'Question':<50}  "
                f"{'F-Prec':>6}  {'F-Rec':>6}  {'C-Prec':>6}  {'C-Rec':>6}  {'ms':>6}"
            )
            lines.append(
                f"  {'-'*5} {'-'*8} {'-'*50}  "
                f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}"
            )

            for qr in pr.question_results:
                q = qr.question
                q_text = q.question
                if len(q_text) > 48:
                    q_text = q_text[:45] + "..."

                # Chunk columns show "n/a" when no chunk ground truth
                cp_str = f"{qr.chunk_precision:5.1%}" if q.ground_truth_chunks else "  n/a"
                cr_str = f"{qr.chunk_recall:5.1%}" if q.ground_truth_chunks else "  n/a"

                lines.append(
                    f"  {q.id:<5} [{q.difficulty:^6}] {q_text:<50}  "
                    f"{qr.file_precision:5.1%}  {qr.file_recall:5.1%}  "
                    f"{cp_str}  {cr_str}  {qr.elapsed_ms:5.0f}"
                )

            # Per-question detail
            lines.append("")
            lines.append("  Retrieved vs Ground Truth:")
            for qr in pr.question_results:
                q = qr.question
                lines.append(f"    {q.id} [{q.category}] {q.question}")
                lines.append(
                    f"      GT files : {', '.join(q.ground_truth_files)}"
                )
                lines.append(
                    f"      Ret files: {', '.join(qr.retrieved_files) or '(none)'}"
                )
                if q.ground_truth_chunks:
                    gt_syms = [
                        c.get("symbol_name", f"L{c.get('line_start')}-{c.get('line_end')}")
                        for c in q.ground_truth_chunks
                    ]
                    ret_syms = [
                        c.get("symbol_name") or f"L{c['line_start']}-{c['line_end']}"
                        for c in qr.retrieved_chunks
                    ]
                    lines.append(f"      GT chunks: {', '.join(str(s) for s in gt_syms)}")
                    lines.append(f"      Ret chunks: {', '.join(str(s) for s in ret_syms)}")

            # Project summary
            lines.append("")
            lines.append(f"  Project averages:")
            lines.append(f"    File precision:  {pr.avg_file_precision:5.1%}")
            lines.append(f"    File recall:     {pr.avg_file_recall:5.1%}")
            lines.append(f"    Chunk precision: {pr.avg_chunk_precision:5.1%}")
            lines.append(f"    Chunk recall:    {pr.avg_chunk_recall:5.1%}")

        # Global aggregates
        agg = results["aggregate"]
        lines.append("")
        lines.append(sep)
        lines.append(f"  AGGREGATE ({agg['total_questions']} questions)")
        lines.append(sep)
        lines.append(f"  File precision:  {agg['file_precision']:5.1%}")
        lines.append(f"  File recall:     {agg['file_recall']:5.1%}")
        lines.append(f"  Chunk precision: {agg['chunk_precision']:5.1%}")
        lines.append(f"  Chunk recall:    {agg['chunk_recall']:5.1%}")
        lines.append(sep)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discovery: find question files and build project list
# ---------------------------------------------------------------------------


def discover_projects(
    questions_dir: str,
    base_dir: str | None = None,
) -> list[BenchmarkProject]:
    """
    Scan *questions_dir* for JSON files and build BenchmarkProject descriptors.

    Each JSON file must have a ``project`` key (name) and a ``root`` key
    (path to the project, resolved relative to *base_dir*).

    Args:
        questions_dir: Directory containing ``*.json`` question files.
        base_dir:      Base directory for resolving relative ``root`` paths.
                       Defaults to the parent of the mnemosyne package.

    Returns:
        List of :class:`BenchmarkProject` descriptors.
    """
    if base_dir is None:
        # Default: the directory containing the mnemosyne package
        base_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

    projects: list[BenchmarkProject] = []
    if not os.path.isdir(questions_dir):
        return projects

    for fname in sorted(os.listdir(questions_dir)):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(questions_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = data.get("project", os.path.splitext(fname)[0])
        root_raw = data.get("root", ".")

        # Resolve root path
        if os.path.isabs(root_raw):
            root = root_raw
        else:
            # Relative to the mnemosyne package directory
            mnemosyne_pkg_dir = os.path.abspath(os.path.join(_THIS_DIR, ".."))
            root = os.path.abspath(os.path.join(mnemosyne_pkg_dir, root_raw))

        if not os.path.isdir(root):
            print(f"  WARNING: skipping {name} -- root not found: {root}")
            continue

        projects.append(BenchmarkProject(
            name=name,
            root=root,
            questions_file=fpath,
        ))

    return projects


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MnemoSync Benchmark Suite -- multi-project retrieval quality measurement "
        "with file-level and chunk-level precision/recall."
    )
    parser.add_argument(
        "--questions-dir",
        default=os.path.join(_THIS_DIR, "benchmark_questions"),
        help="Directory containing JSON question files (default: tests/benchmark_questions/)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Override token budget for all questions (0 = use per-question budgets)",
    )
    args = parser.parse_args()

    print("=== MnemoSync Benchmark Suite ===")
    print(f"Questions dir: {args.questions_dir}")

    projects = discover_projects(args.questions_dir)
    if not projects:
        print("ERROR: No valid projects found in questions directory.")
        sys.exit(1)

    print(f"Found {len(projects)} project(s): {', '.join(p.name for p in projects)}")

    # Apply budget override if specified
    if args.budget > 0:
        for p in projects:
            # We will override per-question budgets at run time
            pass

    suite = BenchmarkSuite(projects)

    # Monkey-patch budget if override specified
    if args.budget > 0:
        _orig_run_question = suite._run_question

        def _budget_override_run(engine, q):
            q = BenchmarkQuestion(
                id=q.id,
                question=q.question,
                category=q.category,
                ground_truth_files=q.ground_truth_files,
                ground_truth_chunks=q.ground_truth_chunks,
                budget=args.budget,
                difficulty=q.difficulty,
            )
            return _orig_run_question(engine, q)

        suite._run_question = _budget_override_run

    results = suite.run_all()
    report = suite.format_report(results)
    print(report)


if __name__ == "__main__":
    main()
