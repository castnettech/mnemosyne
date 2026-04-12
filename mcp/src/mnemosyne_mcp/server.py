"""Mnemosyne MCP Server.

Exposes Mnemosyne's 6-signal hybrid code retrieval engine as MCP tools
for Claude Code and other MCP-compatible agents.

Install:
    pip install mnemosyne-mcp

Register with Claude Code:
    claude mcp add mnemosyne -- mnemosyne-mcp

Or add to your project's .mcp.json:
    {
      "mcpServers": {
        "mnemosyne": {
          "command": "mnemosyne-mcp",
          "args": []
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# ---------------------------------------------------------------------------
# Lazy engine initialization -- only imports mnemosyne when first tool is called
# ---------------------------------------------------------------------------

_engine_cache: dict[str, object] = {}


def _get_engine(project_root: str) -> tuple:
    """Return (RetrievalEngine, Store, Config, DocRetrievalEngine, DocStore) for project root.

    Initialises the index directory and loads existing data on first call.
    Caches per project root so repeated queries are fast.
    """
    resolved = str(Path(project_root).resolve())

    if resolved in _engine_cache:
        return _engine_cache[resolved]

    from mnemosyne.config import Config
    from mnemosyne.schema import open_store
    from mnemosyne.store import Store
    from mnemosyne.doc_store import DocStore
    from mnemosyne.embeddings import get_backend
    from mnemosyne.analytics import Analytics
    from mnemosyne.prefetch import Prefetcher
    from mnemosyne.retrieval import RetrievalEngine
    from mnemosyne.doc_retrieval import DocRetrievalEngine

    mnemosyne_dir = Path(resolved) / ".mnemosyne"
    if not mnemosyne_dir.exists():
        from mnemosyne.schema import init_db

        mnemosyne_dir.mkdir(parents=True, exist_ok=True)
        init_db(str(mnemosyne_dir))

    config = Config(root=resolved)
    conn = open_store(str(mnemosyne_dir))
    store = Store(conn)
    tfidf = get_backend(config, store)
    analytics = Analytics(store, config)
    prefetcher = Prefetcher(store)

    engine = RetrievalEngine(
        store=store,
        tfidf_backend=tfidf,
        config=config,
        analytics=analytics,
        prefetcher=prefetcher,
    )

    # Document partition -- same connection, separate tables
    doc_store = DocStore(conn)
    doc_tfidf = get_backend(config, store=doc_store)
    doc_engine = DocRetrievalEngine(
        doc_store=doc_store,
        tfidf_backend=doc_tfidf,
        config=config,
    )

    result = (engine, store, config, doc_engine, doc_store)
    _engine_cache[resolved] = result
    return result


def _resolve_project_root(requested: str | None) -> str:
    """Determine project root: explicit arg > env var > cwd."""
    if requested:
        p = Path(requested).resolve()
        if p.is_dir():
            return str(p)
        raise ValueError(f"Not a directory: {requested}")

    env_root = os.environ.get("MNEMOSYNE_PROJECT_ROOT")
    if env_root:
        return str(Path(env_root).resolve())

    return str(Path.cwd().resolve())


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("mnemosyne")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description=(
                "Search a codebase using Mnemosyne's 6-signal hybrid retrieval engine. "
                "Combines BM25, TF-IDF, symbol matching, usage frequency, predictive "
                "prefetch, and optional dense embeddings via Reciprocal Rank Fusion. "
                "Returns the most relevant code chunks within a configurable token budget, "
                "with AST-aware compression. Use this instead of grep/ripgrep for "
                "semantic code understanding."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query or keyword search.",
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Maximum token budget for results. Default 8000.",
                        "default": 8000,
                    },
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="index",
            description=(
                "Index or re-index a codebase for Mnemosyne retrieval. "
                "Performs incremental indexing by default (only changed files). "
                "Run this before your first search, or after significant code changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                    "full": {
                        "type": "boolean",
                        "description": "Force full re-index (not incremental). Default false.",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="stats",
            description=(
                "Show index statistics for a Mnemosyne-indexed codebase: "
                "file count, chunk count, total tokens, language breakdown, "
                "and cache hit rate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="search_docs",
            description=(
                "Search the document partition of a Mnemosyne index. Searches "
                "ingested PDFs, DOCX files, CSVs, logs, and other non-code documents "
                "using BM25 and TF-IDF with isolated vocabulary. Use this for "
                "organizational knowledge, documentation, and reference material."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query.",
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Maximum token budget for results. Default 8000.",
                        "default": 8000,
                    },
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="schema_ingest",
            description=(
                "Ingest database schema into the Mnemosyne index. Accepts DDL files, "
                "JSON schema snapshots, or SQLite database paths. Schema chunks flow "
                "through the same retrieval pipeline as code, enabling queries that "
                "correlate code behavior with database structure and configuration."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": (
                            "Path to schema source: a .sql DDL file, .json schema snapshot, "
                            "or .db/.sqlite SQLite database file."
                        ),
                    },
                    "environment": {
                        "type": "string",
                        "description": "Environment tag (e.g. prod, dev, staging). Default empty.",
                        "default": "",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["auto", "ddl", "json", "yaml", "sqlite"],
                        "description": "Source format. Default auto-detects from extension.",
                        "default": "auto",
                    },
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                },
                "required": ["source_path"],
            },
        ),
        Tool(
            name="schema_stats",
            description=(
                "Show statistics about ingested database schema sources: "
                "source count, environments indexed, chunk counts by type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {
                        "type": "string",
                        "description": (
                            "Absolute path to the project root. "
                            "Defaults to MNEMOSYNE_PROJECT_ROOT env var or cwd."
                        ),
                    },
                },
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search":
            return await _handle_search(arguments)
        elif name == "search_docs":
            return await _handle_search_docs(arguments)
        elif name == "index":
            return await _handle_index(arguments)
        elif name == "stats":
            return await _handle_stats(arguments)
        elif name == "schema_ingest":
            return await _handle_schema_ingest(arguments)
        elif name == "schema_stats":
            return await _handle_schema_stats(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {type(exc).__name__}: {exc}")]


async def _handle_search(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    budget = arguments.get("budget", 8000)
    project_root = _resolve_project_root(arguments.get("project_root"))
    engine, store, config, doc_engine, doc_store = _get_engine(project_root)

    if store.count_files() == 0:
        return [TextContent(
            type="text",
            text=(
                "No files indexed yet. Run the 'index' tool first, or run:\n"
                "  mnemosyne ingest\n"
                f"in {project_root}"
            ),
        )]

    loop = asyncio.get_event_loop()

    # Code partition search
    code_results = await loop.run_in_executor(
        None,
        lambda: engine.query(query_text=query, budget=budget, use_compression=True),
    )

    # Document partition search (half budget -- LLM decides relevance)
    doc_results = await loop.run_in_executor(
        None,
        lambda: doc_engine.query(query_text=query, budget=budget // 2),
    )

    if not code_results and not doc_results:
        return [TextContent(type="text", text=f"No results found for: {query}")]

    from mnemosyne.formatter import Formatter

    max_chars = 65_536
    has_code = bool(code_results)
    has_docs = bool(doc_results)

    if has_code and has_docs:
        per_partition = max_chars // 2
    else:
        per_partition = max_chars

    parts: list[str] = []
    if has_code:
        code_text = Formatter.format_plain(code_results, query, budget, session_id=None)
        if len(code_text) > per_partition:
            code_text = code_text[:per_partition] + "\n... [code results truncated]"
        parts.append("## Code Results\n\n" + code_text)
    if has_docs:
        doc_text = Formatter.format_plain(doc_results, query, budget // 2, session_id=None)
        if len(doc_text) > per_partition:
            doc_text = doc_text[:per_partition] + "\n... [document results truncated]"
        parts.append("## Document Results\n\n" + doc_text)

    output = "\n\n".join(parts)

    return [TextContent(type="text", text=output)]


async def _handle_search_docs(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "").strip()
    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    budget = arguments.get("budget", 8000)
    project_root = _resolve_project_root(arguments.get("project_root"))
    _, _, config, doc_engine, doc_store = _get_engine(project_root)

    if doc_store.count_chunks() == 0:
        return [TextContent(
            type="text",
            text="No documents indexed yet. Run 'index' to ingest PDF, DOCX, CSV, and other document files.",
        )]

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: doc_engine.query(query_text=query, budget=budget),
    )

    if not results:
        return [TextContent(type="text", text=f"No document results for: {query}")]

    from mnemosyne.formatter import Formatter

    output = Formatter.format_plain(results, query, budget, session_id=None)

    max_chars = 65_536
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... [truncated to 64KB]"

    return [TextContent(type="text", text=output)]


async def _handle_index(arguments: dict) -> list[TextContent]:
    project_root = _resolve_project_root(arguments.get("project_root"))
    full = arguments.get("full", False)

    from mnemosyne.config import Config
    from mnemosyne.schema import open_store, init_db
    from mnemosyne.store import Store
    from mnemosyne.embeddings import get_backend
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog
    from mnemosyne.ingest import Ingester

    mnemosyne_dir = Path(project_root) / ".mnemosyne"
    if not mnemosyne_dir.exists():
        mnemosyne_dir.mkdir(parents=True, exist_ok=True)
        init_db(str(mnemosyne_dir))

    from mnemosyne.doc_store import DocStore

    config = Config(root=project_root)
    conn = open_store(str(mnemosyne_dir))
    store = Store(conn)
    tfidf = get_backend(config, store)
    bloom = BloomFilter()
    audit = AuditLog(str(mnemosyne_dir / "audit.jsonl"))

    doc_store = DocStore(conn)
    doc_tfidf = get_backend(config, store=doc_store)

    ingester = Ingester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf_backend=tfidf,
        audit=audit,
        doc_store=doc_store,
        doc_tfidf=doc_tfidf,
    )

    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: ingester.ingest(full=full),
    )

    elapsed = time.monotonic() - t0

    # Invalidate engine cache so next search picks up new data
    resolved = str(Path(project_root).resolve())
    _engine_cache.pop(resolved, None)

    lines = [
        f"Indexing complete ({elapsed:.1f}s)",
        f"  Files scanned:  {stats.get('files_scanned', 0)}",
        f"  Files indexed:  {stats.get('files_indexed', 0)}",
        f"  Files skipped:  {stats.get('files_skipped', 0)}",
        f"  Chunks added:   {stats.get('chunks_added', 0)}",
        f"  Chunks deduped: {stats.get('chunks_deduped', 0)}",
    ]
    if stats.get("files_failed", 0) > 0:
        lines.append(f"  Files failed:   {stats['files_failed']}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_stats(arguments: dict) -> list[TextContent]:
    project_root = _resolve_project_root(arguments.get("project_root"))
    _, store, config, _, doc_store = _get_engine(project_root)

    file_count = store.count_files()
    chunk_count = store.count_chunks()
    total_tokens = store.total_tokens()
    lang_counts = store.language_counts()
    type_counts = store.chunk_type_counts()

    lines = [
        f"Mnemosyne Index: {project_root}",
        f"  Files:    {file_count}",
        f"  Chunks:   {chunk_count}",
        f"  Tokens:   {total_tokens:,}",
        "",
        "  Languages:",
    ]
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {lang}: {count} files")

    lines.append("")
    lines.append("  Chunk types:")
    for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {ctype}: {count}")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_schema_ingest(arguments: dict) -> list[TextContent]:
    source_path = arguments.get("source_path", "").strip()
    if not source_path:
        return [TextContent(type="text", text="Error: source_path is required")]

    environment = arguments.get("environment", "")
    fmt = arguments.get("format", "auto")
    project_root = _resolve_project_root(arguments.get("project_root"))

    from mnemosyne.config import Config
    from mnemosyne.schema import open_store
    from mnemosyne.store import Store
    from mnemosyne.embeddings import get_backend
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog
    from mnemosyne.schema_ingest import SchemaIngester

    mnemosyne_dir = Path(project_root) / ".mnemosyne"
    if not mnemosyne_dir.exists():
        mnemosyne_dir.mkdir(parents=True, exist_ok=True)

    config = Config(root=project_root)
    conn = open_store(str(mnemosyne_dir))
    store = Store(conn)
    tfidf = get_backend(config, store)
    bloom = BloomFilter()
    audit = AuditLog(str(mnemosyne_dir / "audit.jsonl"))

    ingester = SchemaIngester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf=tfidf,
        audit=audit,
    )

    loop = asyncio.get_event_loop()

    if fmt == "sqlite" or source_path.endswith((".db", ".sqlite", ".sqlite3")):
        stats = await loop.run_in_executor(
            None,
            lambda: ingester.introspect_sqlite(source_path, env_tag=environment),
        )
        lines = [
            f"Schema ingested from SQLite: {source_path}",
            f"  Tables found:   {stats.get('tables_found', 0)}",
            f"  Chunks added:   {stats['chunks_added']}",
            f"  Chunks deduped: {stats['chunks_deduped']}",
            f"  Redactions:     {stats['redactions']}",
        ]
    else:
        stats = await loop.run_in_executor(
            None,
            lambda: ingester.ingest_from_file(source_path, env_tag=environment, fmt=fmt),
        )
        lines = [
            f"Schema ingested: {source_path}",
            f"  Environment:    {environment or '(none)'}",
            f"  Chunks added:   {stats['chunks_added']}",
            f"  Chunks deduped: {stats['chunks_deduped']}",
            f"  Redactions:     {stats['redactions']}",
        ]

    # Invalidate engine cache
    resolved = str(Path(project_root).resolve())
    _engine_cache.pop(resolved, None)

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_schema_stats(arguments: dict) -> list[TextContent]:
    project_root = _resolve_project_root(arguments.get("project_root"))

    from mnemosyne.config import Config
    from mnemosyne.schema import open_store
    from mnemosyne.store import Store
    from mnemosyne.embeddings import get_backend
    from mnemosyne.bloom import BloomFilter
    from mnemosyne.audit import AuditLog
    from mnemosyne.schema_ingest import SchemaIngester

    mnemosyne_dir = Path(project_root) / ".mnemosyne"
    config = Config(root=project_root)
    conn = open_store(str(mnemosyne_dir))
    store = Store(conn)
    tfidf = get_backend(config, store)
    bloom = BloomFilter()
    audit = AuditLog(str(mnemosyne_dir / "audit.jsonl"))

    ingester = SchemaIngester(
        project_root=project_root,
        config=config,
        store=store,
        bloom=bloom,
        tfidf=tfidf,
        audit=audit,
    )

    stats = ingester.get_schema_stats()
    lines = [
        f"Schema Index: {project_root}",
        f"  Sources:      {stats['schema_sources']}",
        f"  Environments: {', '.join(stats['environments']) or '(none)'}",
        f"  Total chunks: {stats['total_chunks']}",
    ]
    if stats["chunk_types"]:
        lines.append("  Chunk types:")
        for ctype, count in sorted(stats["chunk_types"].items()):
            lines.append(f"    {ctype}: {count}")

    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run():
    """Synchronous entry point for the console script."""
    asyncio.run(main())


async def main() -> None:
    """Run the Mnemosyne MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    run()
