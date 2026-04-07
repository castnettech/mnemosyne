# Copyright 2026 Cast Rock Innovation L.L.C.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Mnemosyne command-line interface.

Entry point: ``mnemosyne`` (configured in pyproject.toml / setup.py as
``mnemosyne.cli:main``).

Commands
--------
init      -- Create ``.mnemosyne/`` and default config in the current directory.
ingest    -- Index files into the knowledge base.
query     -- Retrieve relevant context for a query string.
stats     -- Show index statistics and cache health.
compress  -- Preview compression of a single file.
cache     -- Manage the ARC cache (show / clear / warm).
delta     -- Show changes since the last index run.
audit     -- Print recent audit log entries.
analytics -- Show feedback precision metrics and top-used chunks.
gc        -- Garbage collect orphaned chunks and stale records.
health    -- Report index health with pass/fail indicators.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging(log_format: str, log_level: str) -> None:
    """Configure root logger based on CLI flags."""
    level = getattr(logging, log_level, logging.WARNING)

    if log_format == "json":
        import json as _json

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }, default=str)

        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    root = logging.getLogger("mnemosyne")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Helpers shared by multiple handlers
# ---------------------------------------------------------------------------


def _find_project_root() -> str:
    """
    Walk upward from cwd looking for a ``.mnemosyne/`` directory.

    Returns the directory containing ``.mnemosyne/``, or cwd if none found.
    """
    current = Path(os.getcwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / ".mnemosyne").is_dir():
            return str(parent)
    return str(current)


def _open_store_conn(project_root: str):
    """Open (and initialise if needed) the Mnemosyne SQLite connection."""
    from mnemosyne.schema import open_store
    db_dir = os.path.join(project_root, ".mnemosyne")
    return open_store(db_dir)


def _load_config(project_root: str):
    from mnemosyne.config import Config
    return Config(root=project_root)


def _make_store(conn):
    from mnemosyne.store import Store
    return Store(conn)


def _make_tfidf(store, config):
    from mnemosyne.embeddings import get_backend
    return get_backend(config, store)


def _make_bloom(project_root: str):
    from mnemosyne.bloom import BloomFilter
    bloom_path = os.path.join(project_root, ".mnemosyne", "bloom.bin")
    if os.path.isfile(bloom_path):
        try:
            return BloomFilter.load(bloom_path)
        except Exception:
            logger.warning("Could not load Bloom filter from %s -- creating new", bloom_path)
    return BloomFilter()


def _save_bloom(bloom, project_root: str) -> None:
    bloom_path = os.path.join(project_root, ".mnemosyne", "bloom.bin")
    try:
        bloom.save(bloom_path)
    except Exception:
        logger.warning("Could not save Bloom filter to %s", bloom_path)


def _make_audit(project_root: str):
    from mnemosyne.audit import AuditLog
    audit_path = os.path.join(project_root, ".mnemosyne", "audit.log")
    return AuditLog(audit_path)


def _make_analytics(store, config):
    from mnemosyne.analytics import Analytics
    return Analytics(store, config)


def _make_prefetcher(store):
    from mnemosyne.prefetch import Prefetcher
    return Prefetcher(store)


def _make_dense_backend(config, store, project_root: str):
    """Create the dense embedding backend if configured and available."""
    dense_model = getattr(config.embedding, "dense_model", None)
    if not dense_model:
        return None
    try:
        from mnemosyne.embeddings.dense_backend import DenseBackend
        model_dir = os.path.join(project_root, ".mnemosyne", "models")
        return DenseBackend(config, store, model_dir=model_dir)
    except ImportError:
        return None


def _make_cache(config):
    from mnemosyne.cache import ARCCache
    return ARCCache(
        capacity=config.cache.capacity,
        ghost_capacity=config.cache.ghost_capacity,
    )


def _require_mnemosyne_dir(project_root: str) -> bool:
    """Return True if .mnemosyne exists; print an error and return False otherwise."""
    if not os.path.isdir(os.path.join(project_root, ".mnemosyne")):
        print(
            "No .mnemosyne/ directory found. Run 'mnemosyne init' first.",
            file=sys.stderr,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """
    Initialise a ``.mnemosyne/`` workspace in the current directory.

    Creates:
    - ``.mnemosyne/`` directory
    - ``.mnemosyne/config.toml`` (default values)
    - ``.mnemosyne/mnemosyne.db`` (schema applied)
    """
    target_dir = Path(os.getcwd()).resolve()
    mnemosyne_dir = target_dir / ".mnemosyne"

    if mnemosyne_dir.exists():
        print(f"Already initialised: {mnemosyne_dir}")
        return 0

    mnemosyne_dir.mkdir(parents=True, exist_ok=True)

    # Write default config
    from mnemosyne.config import Config
    config = Config(root=str(target_dir))
    config.save()
    print(f"Created: {mnemosyne_dir / 'config.toml'}")

    # Initialise database
    try:
        conn = _open_store_conn(str(target_dir))
        conn.close()
        print(f"Created: {mnemosyne_dir / 'mnemosyne.db'}")
    except Exception as exc:
        print(f"Warning: could not create database: {exc}", file=sys.stderr)

    print(f"\nMnemosyne initialised at: {target_dir}")
    print("Run 'mnemosyne ingest' to index your project.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Index files into the knowledge base."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)

    # --code-only: remove documentation-type extensions from supported list
    if getattr(args, "code_only", False):
        doc_extensions = {".md", ".txt", ".rst", ".log", ".csv", ".tsv", ".ini", ".cfg", ".conf"}
        current = list(config.general.supported_extensions)
        filtered = [ext for ext in current if ext not in doc_extensions]
        config.general.supported_extensions = filtered

    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    bloom = _make_bloom(project_root)
    tfidf = _make_tfidf(store, config)
    audit = _make_audit(project_root)
    dense = _make_dense_backend(config, store, project_root)

    from mnemosyne.doc_store import DocStore
    from mnemosyne.embeddings import get_backend as _get_be

    doc_store = DocStore(conn)
    doc_tfidf = _get_be(config, store=None)

    from mnemosyne.ingest import Ingester
    ingester = Ingester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf_backend=tfidf,
        audit=audit,
        dense_backend=dense,
        doc_store=doc_store,
        doc_tfidf=doc_tfidf,
    )

    paths = args.paths if args.paths else None
    if getattr(args, "code_only", False):
        print("[code-only] Skipping documentation files (.md, .txt, .rst, .log, etc.)")
    if args.dry_run:
        print("[dry-run] Scanning files without writing...")

    try:
        stats = ingester.ingest(paths=paths, full=args.full, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        conn.close()
        return 1

    if not args.dry_run:
        _save_bloom(bloom, project_root)

    print(f"Files scanned:  {stats['files_scanned']}")
    print(f"Files indexed:  {stats['files_indexed']}")
    print(f"Files skipped:  {stats['files_skipped']}")
    print(f"Files failed:   {stats['files_failed']}")
    print(f"Chunks added:   {stats['chunks_added']}")
    print(f"Chunks deduped: {stats['chunks_deduped']}")
    print(f"Elapsed:        {stats['elapsed_seconds']:.2f}s")

    conn.close()
    return 0


def cmd_schema_ingest(args: argparse.Namespace) -> int:
    """Ingest database schema sources into the index."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    bloom = _make_bloom(project_root)
    tfidf = _make_tfidf(store, config)
    audit = _make_audit(project_root)
    dense = _make_dense_backend(config, store, project_root)

    from mnemosyne.schema_ingest import SchemaIngester
    ingester = SchemaIngester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf=tfidf,
        audit=audit,
        dense=dense,
    )

    source = getattr(args, "source", None)
    sqlite_db = getattr(args, "sqlite", None)
    env_tag = getattr(args, "env", "") or ""
    fmt = getattr(args, "format", "auto") or "auto"

    try:
        if sqlite_db:
            stats = ingester.introspect_sqlite(sqlite_db, env_tag=env_tag)
            print(f"SQLite DB:      {sqlite_db}")
            print(f"Environment:    {env_tag or '(none)'}")
            print(f"Tables found:   {stats.get('tables_found', 0)}")
            print(f"Chunks added:   {stats['chunks_added']}")
            print(f"Chunks deduped: {stats['chunks_deduped']}")
            print(f"Redactions:     {stats['redactions']}")
        elif source:
            stats = ingester.ingest_from_file(source, env_tag=env_tag, fmt=fmt)
            print(f"Source:         {source}")
            print(f"Environment:    {env_tag or '(none)'}")
            print(f"Chunks added:   {stats['chunks_added']}")
            print(f"Chunks deduped: {stats['chunks_deduped']}")
            print(f"Redactions:     {stats['redactions']}")
        else:
            stats = ingester.ingest_from_config()
            print(f"Sources processed: {stats['sources_processed']}")
            print(f"Sources failed:    {stats['sources_failed']}")
            print(f"Chunks added:      {stats['chunks_added']}")
            print(f"Chunks deduped:    {stats['chunks_deduped']}")
            print(f"Redactions:        {stats['redactions']}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        conn.close()
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        conn.close()
        return 1

    _save_bloom(bloom, project_root)
    conn.close()
    return 0


def cmd_schema_stats(args: argparse.Namespace) -> int:
    """Show statistics about ingested schema sources."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    bloom = _make_bloom(project_root)
    tfidf = _make_tfidf(store, config)
    audit = _make_audit(project_root)

    from mnemosyne.schema_ingest import SchemaIngester
    ingester = SchemaIngester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf=tfidf,
        audit=audit,
    )

    stats = ingester.get_schema_stats()
    print(f"Schema sources:  {stats['schema_sources']}")
    print(f"Environments:    {', '.join(stats['environments']) or '(none)'}")
    print(f"Total chunks:    {stats['total_chunks']}")
    if stats['chunk_types']:
        print("Chunk types:")
        for ctype, count in sorted(stats['chunk_types'].items()):
            print(f"  {ctype}: {count}")

    conn.close()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Retrieve relevant context chunks for a query string."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    tfidf = _make_tfidf(store, config)
    analytics = _make_analytics(store, config)
    prefetcher = _make_prefetcher(store)
    dense = _make_dense_backend(config, store, project_root)

    session_id = args.session
    if session_id is None:
        session_id = analytics.start_session()
    else:
        analytics.start_session(session_id)

    budget = args.budget
    effective_budget = budget if budget is not None else config.retrieval.token_budget

    from mnemosyne.formatter import Formatter

    search_docs = getattr(args, "docs", False)
    search_all = getattr(args, "search_all", False)

    results = []

    # Document partition search
    if search_docs or search_all:
        from mnemosyne.doc_store import DocStore
        from mnemosyne.doc_retrieval import DocRetrievalEngine
        from mnemosyne.embeddings import get_backend as _get_be

        doc_store = DocStore(conn)
        doc_tfidf = _get_be(config, store=None)
        doc_engine = DocRetrievalEngine(
            doc_store=doc_store, tfidf_backend=doc_tfidf, config=config,
        )
        doc_budget = effective_budget // 2 if search_all else effective_budget
        results.extend(doc_engine.query(query_text=args.text, budget=doc_budget))

    # Code partition search (default, or --all)
    if not search_docs or search_all:
        from mnemosyne.retrieval import RetrievalEngine
        engine = RetrievalEngine(
            store=store,
            tfidf_backend=tfidf,
            config=config,
            analytics=analytics,
            prefetcher=prefetcher,
            dense_backend=dense,
        )
        code_budget = effective_budget // 2 if search_all else effective_budget
        results.extend(engine.query(
            query_text=args.text,
            budget=code_budget,
            session_id=session_id,
            use_compression=not args.no_compress,
        ))

    if args.format == "json":
        output = Formatter.format_json(results, args.text, effective_budget, session_id)
    else:
        output = Formatter.format_plain(results, args.text, effective_budget, session_id)

    print(output)
    conn.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Display index and cache statistics."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)

    file_count = store.count_files()
    chunk_count = store.count_chunks()
    total_tokens = store.total_tokens()
    event_count = store.count_usage_events()

    print(f"Project root:   {project_root}")
    print(f"Files indexed:  {file_count:,}")
    print(f"Chunks:         {chunk_count:,}")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Usage events:   {event_count:,}")

    if args.detailed:
        avg_tokens = (total_tokens / chunk_count) if chunk_count else 0
        print(f"\nAvg tokens/chunk: {avg_tokens:.1f}")

        type_counts = store.chunk_type_counts()
        if type_counts:
            print("\nChunk types:")
            for chunk_type, count in sorted(type_counts.items()):
                print(f"  {chunk_type:15s}: {count:,}")

        lang_counts = store.language_counts()
        if lang_counts:
            print("\nLanguages:")
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                print(f"  {(lang or 'unknown'):15s}: {count:,} files")

        cache_stats = store.get_cache_state_counts()
        if cache_stats:
            print("\nCache state (persisted):")
            for tier, count in sorted(cache_stats.items()):
                print(f"  {tier}: {count:,} entries")

        pattern_count = store.count_patterns()
        print(f"\nTask patterns:  {pattern_count:,}")

    conn.close()
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    """Preview compression of a single file."""
    project_root = _find_project_root()
    config = _load_config(project_root)

    target = os.path.abspath(args.file)
    if not os.path.isfile(target):
        print(f"File not found: {target}", file=sys.stderr)
        return 1

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        print(f"Cannot read file: {exc}", file=sys.stderr)
        return 1

    if args.ratio is not None:
        config.compression.target_ratio = args.ratio

    try:
        from mnemosyne.compress import Compressor
        from mnemosyne.models import Chunk, estimate_tokens
        compressor = Compressor(config)
        # Compressor.compress() expects a Chunk object; wrap the source text
        preview_chunk = Chunk(
            chunk_id=None,
            file_id=0,
            content_hash="",
            chunk_type="generic",
            line_start=1,
            line_end=source.count("\n") + 1,
            token_count=estimate_tokens(source),
            content=source,
        )
        compressed = compressor.compress(preview_chunk)
    except ImportError:
        print("Compressor module not available.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Compression error: {exc}", file=sys.stderr)
        return 1

    from mnemosyne.models import estimate_tokens
    orig_tokens = estimate_tokens(source)
    comp_tokens = estimate_tokens(compressed)
    ratio = len(compressed) / max(1, len(source))

    print(f"File:            {os.path.relpath(target, project_root)}")
    print(f"Original tokens: {orig_tokens:,}")
    print(f"Compressed:      {comp_tokens:,}")
    print(f"Char ratio:      {ratio:.2%}")
    print(f"\n{'-' * 60}")
    print(compressed)
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """Manage the ARC cache."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    action = args.action

    if action == "show":
        cache_counts = store.get_cache_state_counts()
        print("Cache state (persisted tiers):")
        total = 0
        for tier in ("T1", "T2", "B1", "B2"):
            count = cache_counts.get(tier, 0)
            total += count
            print(f"  {tier}: {count:,} entries")
        print(f"  Total: {total:,}")

    elif action == "clear":
        store.clear_cache_state()
        print("Cache state cleared.")

    elif action == "warm":
        top_chunks = store.get_top_accessed_chunks(limit=config.cache.capacity)
        cache = _make_cache(config)
        from mnemosyne.tiers import TierManager
        TierManager(cache, store)
        warmed = 0
        for chunk in top_chunks:
            if chunk.chunk_id is not None:
                cache.put(chunk.chunk_id, chunk)
                warmed += 1
        print(f"Warmed cache with {warmed} chunks.")

    conn.close()
    return 0


def cmd_delta(args: argparse.Namespace) -> int:
    """Show file changes since the last index run."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    conn = _open_store_conn(project_root)
    store = _make_store(conn)

    from mnemosyne.delta import DeltaTracker
    tracker = DeltaTracker(store)
    changes = tracker.detect_changes(project_root)

    if args.paths:
        requested = {
            os.path.relpath(os.path.abspath(p), project_root).replace(os.sep, "/")
            for p in args.paths
        }
        changes = [c for c in changes if c["file"] in requested]

    if not changes:
        print("No changes detected since last index.")
        conn.close()
        return 0

    added = [c for c in changes if c["status"] == "added"]
    modified = [c for c in changes if c["status"] == "modified"]
    deleted = [c for c in changes if c["status"] == "deleted"]

    print(f"Changes detected: {len(changes)} file(s)\n")

    if added:
        print(f"Added ({len(added)}):")
        for c in added:
            print(f"  + {c['file']}")

    if deleted:
        print(f"\nDeleted ({len(deleted)}):")
        for c in deleted:
            print(f"  - {c['file']}")

    if modified:
        print(f"\nModified ({len(modified)}):")
        for c in modified:
            print(f"  ~ {c['file']}")
            if c.get("diff"):
                diff_lines = c["diff"].splitlines()
                additions = sum(
                    1 for ln in diff_lines
                    if ln.startswith("+") and not ln.startswith("+++")
                )
                removals = sum(
                    1 for ln in diff_lines
                    if ln.startswith("-") and not ln.startswith("---")
                )
                print(f"    +{additions}/-{removals} lines")

    conn.close()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Print recent audit log entries."""
    project_root = _find_project_root()
    audit_path = os.path.join(project_root, ".mnemosyne", "audit.log")

    if not os.path.isfile(audit_path):
        print("No audit log found.")
        return 0

    try:
        from mnemosyne.audit import AuditLog
        audit = AuditLog(audit_path)
        entries = audit.read(last_n=args.last)
    except Exception as exc:
        print(f"Could not read audit log: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("Audit log is empty.")
        return 0

    print(f"Last {len(entries)} audit entries:\n")
    for entry in entries:
        ts = entry.get("ts", "?")
        action = entry.get("op", "?")
        # All keys except "op" and "ts" are the detail payload
        data = {k: v for k, v in entry.items() if k not in ("op", "ts")}
        data_str = (
            "  ".join(f"{k}={v}" for k, v in data.items())
            if data
            else ""
        )
        print(f"[{ts}] {action}  {data_str}")

    return 0


def cmd_analytics(args: argparse.Namespace) -> int:
    """Display feedback analytics and precision metrics."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    analytics = _make_analytics(store, config)

    session_id = getattr(args, "session", None)
    top_chunks_n = getattr(args, "top_chunks", None)

    # Compute precision
    precision_data = analytics.compute_precision_at_k(session_id=session_id)

    # Count distinct sessions with feedback
    if session_id is not None:
        sessions_with_feedback = 1 if (
            precision_data["total_used"]
            + precision_data["total_discarded"]
            + precision_data["total_selected"]
            + precision_data["total_retrieved"]
        ) > 0 else 0
    else:
        events = store.get_usage_events()
        feedback_sessions: set[str] = set()
        for event in events:
            if event.session_id and event.event_type in (
                "used", "discarded", "selected", "retrieved",
            ):
                feedback_sessions.add(event.session_id)
        sessions_with_feedback = len(feedback_sessions)

    precision_pct = precision_data["precision"] * 100

    print("Feedback Analytics")
    if session_id:
        print(f"  Session:                {session_id}")
    print(f"  Sessions with feedback: {sessions_with_feedback}")
    print(f"  Precision (used/retrieved): {precision_pct:.1f}%")
    print(f"  Total retrieved:        {precision_data['total_retrieved']}")
    print(f"  Total used:             {precision_data['total_used']}")
    print(f"  Total discarded:        {precision_data['total_discarded']}")
    print(f"  Total selected:         {precision_data['total_selected']}")

    # Top chunks
    effective_top = top_chunks_n if top_chunks_n is not None else 5
    if top_chunks_n is not None or (
        precision_data["total_used"] > 0 and top_chunks_n is None
    ):
        top_used = analytics.get_top_used_chunks(limit=effective_top)
        if top_used:
            print(f"\nTop {len(top_used)} most-used chunks:")
            for i, entry in enumerate(top_used, 1):
                label = entry["file_path"]
                if entry["symbol_name"]:
                    label += f":{entry['symbol_name']}"
                label += f" (lines {entry['line_start']}-{entry['line_end']})"
                print(f"  {i}. {label} -- {entry['use_count']} uses")
        elif top_chunks_n is not None:
            print("\nNo 'used' events recorded yet.")

    conn.close()
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    """Garbage collect orphaned chunks and stale file records."""
    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    audit = _make_audit(project_root)

    # Orphaned chunks from soft-deleted file records
    deleted_file_ids = store.get_deleted_file_ids()
    orphan_chunk_count = sum(
        len(store.get_chunks_for_file(fid)) for fid in deleted_file_ids
    )

    # Stale records whose files no longer exist on disk
    stale_file_ids: list[int] = []
    for file_record in store.get_all_file_records():
        if file_record.is_deleted:
            continue
        abs_path = os.path.join(project_root, file_record.rel_path)
        if not os.path.isfile(abs_path):
            stale_file_ids.append(file_record.file_id)

    print(f"Orphaned chunks (deleted files): {orphan_chunk_count}")
    print(f"Stale file records (missing):    {len(stale_file_ids)}")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        conn.close()
        return 0

    removed_chunks = 0
    for file_id in deleted_file_ids:
        n = len(store.get_chunks_for_file(file_id))
        store.delete_chunks_for_file(file_id)
        removed_chunks += n

    for file_id in stale_file_ids:
        n = len(store.get_chunks_for_file(file_id))
        store.delete_chunks_for_file(file_id)
        removed_chunks += n
        store.mark_file_deleted(file_id)

    store.prune_cache_state()
    store.prune_usage_events()

    # Rebuild Bloom filter from surviving files and chunks so that stale
    # entries no longer cause false "already indexed" skips on future ingests.
    from mnemosyne.bloom import BloomFilter
    bloom = BloomFilter()
    bloom_entry_count = 0

    # Add surviving (non-deleted) file paths
    for file_record in store.list_files(include_deleted=False):
        bloom.add(file_record.rel_path)
        bloom_entry_count += 1

    # Add surviving chunk content hashes
    rows = conn.execute(
        "SELECT DISTINCT content_hash FROM chunks"
    ).fetchall()
    for row in rows:
        bloom.add(row[0])
        bloom_entry_count += 1

    _save_bloom(bloom, project_root)

    print(f"\nRemoved {removed_chunks} chunks.")
    print(f"Bloom filter rebuilt with {bloom_entry_count} entries.")
    print("Garbage collection complete.")

    try:
        audit.log("gc", removed_chunks=removed_chunks, stale_files=len(stale_file_ids))
    except Exception:
        logger.warning("Failed to write GC audit entry")

    conn.close()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the benchmark suite against configured projects."""
    from mnemosyne.tests.benchmark_suite import BenchmarkSuite, discover_projects

    questions_dir = args.questions_dir
    if questions_dir is None or not os.path.isdir(questions_dir):
        # Try relative to mnemosyne package
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        questions_dir = os.path.join(pkg_dir, "tests", "benchmark_questions")

    if not os.path.isdir(questions_dir):
        print("Error: questions directory not found", file=sys.stderr)
        return 1

    projects = discover_projects(questions_dir)
    if not projects:
        print("No benchmark projects found.", file=sys.stderr)
        return 1

    print(f"Found {len(projects)} project(s)")
    suite = BenchmarkSuite(projects)
    results = suite.run_all()
    report = suite.format_report(results)
    print(report)
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    """Manage the Mnemosyne background daemon."""
    import signal as _signal

    project_root = _find_project_root()
    action = args.action

    if action == "start":
        if not _require_mnemosyne_dir(project_root):
            return 1

        from mnemosyne.daemon import is_daemon_alive
        if is_daemon_alive(project_root):
            from mnemosyne.daemon import read_pid
            print(f"Daemon already running (PID {read_pid(project_root)}).")
            return 0

        if args.foreground:
            from mnemosyne.daemon import MnemosyneDaemon
            daemon = MnemosyneDaemon(project_root)
            daemon.start()
        else:
            pid = os.fork()
            if pid == 0:
                # Child: detach from terminal.
                os.setsid()
                # Redirect stdio to /dev/null so the child does not
                # hold the parent's terminal open.
                devnull = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull, 0)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                os.close(devnull)
                from mnemosyne.daemon import MnemosyneDaemon
                daemon = MnemosyneDaemon(project_root)
                daemon.start()
                os._exit(0)
            else:
                print(f"Daemon started (PID {pid}).")
        return 0

    elif action == "stop":
        from mnemosyne.daemon import read_pid, is_daemon_alive
        pid = read_pid(project_root)
        if pid is None:
            print("No daemon PID file found.")
            return 1
        if not is_daemon_alive(project_root):
            print(f"Daemon not running (stale PID {pid}).")
            # Clean up stale PID file.
            pid_path = os.path.join(project_root, ".mnemosyne", "daemon.pid")
            try:
                os.unlink(pid_path)
            except OSError:
                pass
            return 1
        os.kill(pid, _signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (PID {pid}).")
        return 0

    elif action == "status":
        from mnemosyne.daemon import read_pid, is_daemon_alive
        pid = read_pid(project_root)
        if pid is None:
            print("Daemon is not running (no PID file).")
            return 1
        if is_daemon_alive(project_root):
            print(f"Daemon is running (PID {pid}).")
            return 0
        else:
            print(f"Daemon is not running (stale PID {pid}).")
            return 1

    return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Report index health with pass/fail indicators."""
    import json as _json
    from datetime import datetime, timezone

    project_root = _find_project_root()
    if not _require_mnemosyne_dir(project_root):
        return 1

    config = _load_config(project_root)
    conn = _open_store_conn(project_root)
    store = _make_store(conn)
    report: dict = {}

    # -- Index age --------------------------------------------------------
    row = conn.execute(
        "SELECT MAX(last_indexed) FROM files WHERE is_deleted = 0"
    ).fetchone()
    last_indexed_str = row[0] if row else None
    if last_indexed_str:
        try:
            last_dt = datetime.fromisoformat(last_indexed_str)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - last_dt).total_seconds()
            hours, rem = divmod(int(max(age_sec, 0)), 3600)
            mins = rem // 60
            age_human = f"{hours}h {mins:02d}m" if hours else f"{mins}m"
            report["index_age"] = age_human
            report["last_indexed"] = last_dt.isoformat()
        except (ValueError, TypeError):
            report["index_age"] = "unknown"
            report["last_indexed"] = last_indexed_str
    else:
        report["index_age"] = "never"
        report["last_indexed"] = None

    # -- File and chunk counts -------------------------------------------
    file_count = store.count_files(include_deleted=False)
    chunk_count = store.count_chunks()
    total_tokens = store.total_tokens()
    report["files"] = file_count
    report["chunks"] = chunk_count
    report["tokens"] = total_tokens

    # -- Vocabulary size -------------------------------------------------
    vocab_data = store.load_vocabulary()
    vocab_size = len(vocab_data["vocabulary"]) if vocab_data and "vocabulary" in vocab_data else 0
    report["vocabulary"] = vocab_size

    # -- Stale file count ------------------------------------------------
    stale_count = 0
    missing_count = 0
    for rec in store.list_files(include_deleted=False):
        full_path = os.path.join(project_root, rec.rel_path)
        if not os.path.isfile(full_path):
            missing_count += 1
            continue
        try:
            disk_mtime = os.path.getmtime(full_path)
            if abs(disk_mtime - rec.last_modified) > 1.0:
                stale_count += 1
        except OSError:
            missing_count += 1
    report["stale_files"] = stale_count
    report["missing_files"] = missing_count

    # -- FTS5 integrity --------------------------------------------------
    try:
        ic_row = conn.execute("PRAGMA integrity_check").fetchone()
        fts_ok = ic_row and ic_row[0] == "ok"
    except Exception:
        fts_ok = False
    report["fts_integrity"] = "ok" if fts_ok else "FAIL"

    # -- Tokenizer hash --------------------------------------------------
    try:
        tfidf = _make_tfidf(store, config)
        stored_hash = store.get_index_metadata("tokenizer_hash")
        current_hash = getattr(tfidf, "_compute_tokenizer_hash", lambda: None)()
        if stored_hash and current_hash:
            tok_match = stored_hash == current_hash
            tok_label = "current" if tok_match else "STALE"
            report["tokenizer"] = tok_label
            report["tokenizer_hash"] = (current_hash or stored_hash)[:8]
        elif stored_hash:
            report["tokenizer"] = "stored-only"
            report["tokenizer_hash"] = stored_hash[:8]
        else:
            report["tokenizer"] = "not built"
            report["tokenizer_hash"] = None
    except Exception:
        report["tokenizer"] = "error"
        report["tokenizer_hash"] = None

    # -- Daemon status ---------------------------------------------------
    try:
        from mnemosyne.daemon import is_daemon_alive, read_pid
        pid = read_pid(project_root)
        if pid and is_daemon_alive(project_root):
            report["daemon"] = f"running (PID {pid})"
        else:
            report["daemon"] = "not running"
    except Exception:
        report["daemon"] = "unavailable"

    conn.close()

    # -- Output ----------------------------------------------------------
    if getattr(args, "json", False):
        print(_json.dumps(report, indent=2))
    else:
        print("Mnemosyne Health Check")
        print("\u2500" * 22)
        age_line = report["index_age"]
        if report["last_indexed"]:
            age_line += f" (last indexed: {report['last_indexed']})"
        print(f"Index age:       {age_line}")
        stale_suffix = f", {stale_count} stale" if stale_count else ""
        miss_suffix = f", {missing_count} missing" if missing_count else ""
        print(f"Files:           {file_count:,} indexed{stale_suffix}{miss_suffix}")
        print(f"Chunks:          {chunk_count:,}")
        print(f"Tokens:          {total_tokens:,}")
        print(f"Vocabulary:      {vocab_size:,} terms")
        print(f"FTS5 integrity:  {report['fts_integrity']}")
        tok_line = report["tokenizer"]
        if report.get("tokenizer_hash"):
            tok_line += f" (hash: {report['tokenizer_hash']})"
        print(f"Tokenizer:       {tok_line}")
        print(f"Daemon:          {report['daemon']}")

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the ``mnemosyne`` CLI."""
    parser = argparse.ArgumentParser(
        prog="mnemosyne",
        description="Mnemosyne -- LLM Context Compression & Retrieval Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-format",
        choices=["text", "json"],
        default="text",
        help="Log output format (default: text)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Log verbosity (default: WARNING)",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # init
    sub.add_parser(
        "init",
        help="Initialise .mnemosyne/ in the current directory",
    )

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index files into the knowledge base")
    p_ingest.add_argument(
        "paths", nargs="*",
        help="Files or directories to index (default: entire project)",
    )
    p_ingest.add_argument(
        "--full", action="store_true",
        help="Force full re-index, ignoring existing hash/mtime records",
    )
    p_ingest.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be indexed without writing any data",
    )
    p_ingest.add_argument(
        "--code-only", action="store_true",
        help="Skip documentation files (.md, .txt, .rst, .log, .csv, .tsv, .ini, .cfg, .conf)",
    )
    # query
    p_query = sub.add_parser("query", help="Retrieve relevant context for a query")
    p_query.add_argument("text", help="Query string")
    p_query.add_argument(
        "--budget", type=int, default=None,
        help="Maximum total tokens in the result set (default: from config)",
    )
    p_query.add_argument(
        "--format", choices=["plain", "json"], default="plain",
        help="Output format (default: plain)",
    )
    p_query.add_argument(
        "--session", default=None, metavar="SESSION_ID",
        help="Session ID for usage tracking (auto-generated if omitted)",
    )
    p_query.add_argument(
        "--no-compress", action="store_true",
        help="Disable compression fallback when a chunk exceeds remaining budget",
    )
    p_query.add_argument(
        "--show-scores", action="store_true",
        help="Include per-signal scores in plain-text output headers",
    )
    p_query.add_argument(
        "--docs", action="store_true",
        help="Search the document partition only (PDFs, DOCX, CSV, logs)",
    )
    p_query.add_argument(
        "--all", action="store_true", dest="search_all",
        help="Search both code and document partitions",
    )

    # stats
    p_stats = sub.add_parser("stats", help="Show index and cache statistics")
    p_stats.add_argument(
        "--detailed", action="store_true",
        help="Include chunk type breakdown, language counts, and cache state",
    )

    # compress
    p_compress = sub.add_parser("compress", help="Preview compression output for a file")
    p_compress.add_argument("file", help="Path to the file to compress")
    p_compress.add_argument(
        "--ratio", type=float, default=None, metavar="0.0-1.0",
        help="Override target compression ratio (default: from config)",
    )

    # cache
    p_cache = sub.add_parser("cache", help="Manage the ARC cache")
    p_cache.add_argument(
        "action", choices=["show", "clear", "warm"], nargs="?", default="show",
        help="show (default), clear, or warm",
    )

    # delta
    p_delta = sub.add_parser("delta", help="Show file changes since the last index run")
    p_delta.add_argument("paths", nargs="*", help="Limit delta check to specific paths")

    # audit
    p_audit = sub.add_parser("audit", help="Print recent audit log entries")
    p_audit.add_argument(
        "--last", type=int, default=20, metavar="N",
        help="Number of most-recent entries to show (default: 20)",
    )

    # analytics
    p_analytics = sub.add_parser(
        "analytics",
        help="Show feedback analytics and precision metrics",
    )
    p_analytics.add_argument(
        "--session", default=None, metavar="SESSION_ID",
        help="Show precision for a specific session only",
    )
    p_analytics.add_argument(
        "--top-chunks", type=int, default=None, metavar="N",
        help="Show the N most-used chunks (by 'used' event count)",
    )

    # gc
    p_gc = sub.add_parser("gc", help="Garbage collect orphaned chunks and stale file records")
    p_gc.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be removed without deleting anything",
    )

    # benchmark
    p_benchmark = sub.add_parser("benchmark", help="Run retrieval-quality benchmarks")
    p_benchmark.add_argument(
        "--questions-dir",
        default=None,
        help="Directory with benchmark question JSON files (default: built-in)",
    )

    # daemon
    p_daemon = sub.add_parser("daemon", help="Manage the background daemon")
    p_daemon.add_argument(
        "action", choices=["start", "stop", "status"],
        help="start, stop, or check status of the daemon",
    )
    p_daemon.add_argument(
        "--foreground", action="store_true",
        help="Run in foreground (don't fork to background)",
    )

    # schema-ingest
    p_schema = sub.add_parser(
        "schema-ingest",
        help="Ingest database schema sources (DDL, JSON, YAML)",
    )
    p_schema.add_argument(
        "--source", default=None, metavar="PATH",
        help="Path to a schema file (DDL, JSON, or YAML). "
             "If omitted, reads from config.database.schema_sources.",
    )
    p_schema.add_argument(
        "--env", default="", metavar="TAG",
        help="Environment tag (e.g. prod, dev, staging)",
    )
    p_schema.add_argument(
        "--format", choices=["auto", "ddl", "json", "yaml"], default="auto",
        help="Source format (default: auto-detect from extension)",
    )
    p_schema.add_argument(
        "--sqlite", default=None, metavar="DB_PATH",
        help="Introspect a local SQLite database file (must be within project root)",
    )

    # schema-stats
    sub.add_parser(
        "schema-stats",
        help="Show statistics about ingested schema sources",
    )

    # health
    p_health = sub.add_parser("health", help="Report index health with pass/fail indicators")
    p_health.add_argument(
        "--json", action="store_true",
        help="Output as JSON dict (useful for monitoring)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Parse arguments and dispatch to the appropriate command handler.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args()

    _configure_logging(
        getattr(args, "log_format", "text"),
        getattr(args, "log_level", "WARNING"),
    )

    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "schema-ingest": cmd_schema_ingest,
        "schema-stats": cmd_schema_stats,
        "query": cmd_query,
        "stats": cmd_stats,
        "compress": cmd_compress,
        "cache": cmd_cache,
        "delta": cmd_delta,
        "audit": cmd_audit,
        "analytics": cmd_analytics,
        "gc": cmd_gc,
        "benchmark": cmd_benchmark,
        "daemon": cmd_daemon,
        "health": cmd_health,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # stdout closed by a pager or head -- not an error
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if os.environ.get("MNEMOSYNE_DEBUG"):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
