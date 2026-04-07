<p align="center">
  <img src="https://raw.githubusercontent.com/castnettech/mnemosyne/main/docs/assets/mnemosyne.png" alt="Mnemosyne" width="400">
</p>

<h1 align="center">Mnemosyne</h1>

<p align="center">
  Intelligent code retrieval engine -- index, search, and compress any codebase with zero dependencies.
</p>

<p align="center">
  <a href="https://pypi.org/project/mnemosyne-engine/"><img src="https://img.shields.io/pypi/v/mnemosyne-engine" alt="PyPI"></a>
  <a href="https://github.com/castnettech/mnemosyne/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11%2B-yellow" alt="Python"></a>
  <a href="https://github.com/castnettech/mnemosyne/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/runtime_deps-zero-brightgreen" alt="Dependencies"></a>
</p>

---

<p align="center">
  <img src="https://raw.githubusercontent.com/castnettech/mnemosyne/main/docs/assets/diagrams/mnemosyne-ecosystem.gif" alt="Mnemosyne Ecosystem -- three packages, zero cloud" width="800">
</p>

Mnemosyne indexes your codebase and documents into a local SQLite store, scores every chunk with a 6-signal hybrid retriever, compresses results with AST awareness, and returns exactly what you need within a token or result budget. Supports source code (Python, JS/TS, Go, Rust, C#, Java, Kotlin), documents (PDF, DOCX, CSV, plaintext), and database schemas (SQL DDL, JSON snapshots, SQLite introspection). It runs entirely locally -- no API keys, no cloud, no runtime dependencies beyond Python 3.11+.

## Install

```bash
pip install mnemosyne-engine
```

## Quick Start

```bash
mnemosyne init                                    # create .mnemosyne/ workspace
mnemosyne ingest                                  # index your codebase
mnemosyne query "How does authentication work?"   # search
```

## Performance

| Metric | Result |
|---|---|
| Query latency | **<20ms** warm, <500ms cold |
| Token reduction | **73%** on 829-file production repo |
| File retrieval accuracy | **100%** across all test sets |
| Ingestion speed | **167 files/sec** (~0.5s for 87 files) |
| Compression | **40-70%** per chunk, AST-aware |
| Memory footprint | **10-30 MB** total |
| Storage overhead | **~4.2 bytes** per indexed token |

## Features

- **Hybrid 6-signal search** -- BM25, TF-IDF, symbol matching, usage frequency, predictive prefetch, and optional dense embeddings fused via Reciprocal Rank Fusion
- **Cost-model ranking** -- results ranked by value-per-token, not just relevance. Like a query optimizer for code retrieval
- **AST-aware compression** -- four-stage pipeline preserves signatures, docstrings, and control flow while collapsing boilerplate (20-60% reduction)
- **Self-tuning ARC cache** -- adapts between recency and frequency patterns automatically, persisted across sessions
- **Delta-aware tracking** -- detects file and chunk-level changes, delivers diffs instead of full content (80-95% savings on incremental queries)
- **Content deduplication** -- SHA-256 addressed storage eliminates duplicate chunks across files
- **7-language structural chunking** -- Python (AST), JavaScript/TypeScript, Go, C#, Rust, Java, Kotlin
- **Document ingestion** -- PDF, DOCX, CSV, and plaintext extraction into an isolated document partition with independent BM25 + TF-IDF retrieval. Optional `mnemosyne-engine[pdf]` extra for PDF support
- **Schema ingestion** -- DDL files, JSON/YAML snapshots, and live SQLite introspection indexed alongside code for cross-domain queries
- **Daemon mode** -- JSON-RPC over Unix socket keeps indexes warm for sub-20ms queries
- **Full audit trail** -- append-only JSON-lines log of every operation
- **Zero runtime dependencies** -- pure Python 3.11+ stdlib. One `pip install`, no conflicts

## Use Cases

**Code search and navigation** -- Natural language queries return ranked, deduplicated results with function-level precision. Symbol-aware search finds implementations directly, not just string matches.

**LLM context optimization** -- Feed Claude, GPT, Cursor, or any LLM agent the right tokens from a 100K+ codebase. Drop-in integration via instruction files cuts API spend 70%+ on context-heavy workflows.

**Developer onboarding** -- New team members query "how does X work?" and get ranked results spanning models, middleware, and routes -- complete function signatures with context, not random line hits.

**PR review and CI/CD** -- Delta tracking identifies which functions changed and pulls their callers and tests into a review bundle. Pipe query output into automated review pipelines.

**Legacy codebase archaeology** -- Before a rewrite or migration, index a large monolith to answer "what calls this table?" or "which modules depend on this API?" Hybrid search beats grep for cross-cutting queries.

**Security audit surface mapping** -- Query for patterns like `exec(`, `eval(`, `subprocess.call` with usage-frequency ranking to prioritize the most-called dangerous patterns. Audit log provides evidence trail for compliance.

**Incident response** -- On-call engineer searches "payment timeout retry" at 3am. Gets ranked, compressed results across the codebase instead of grepping blindly.

**Migration impact analysis** -- Planning a framework upgrade or library swap? Query every usage of the old API, ranked by call frequency, to estimate effort and prioritize high-traffic paths.

## MCP Server (Claude Code Integration)

For native Claude Code integration, install the MCP server addon:

```bash
pip install mnemosyne-mcp
claude mcp add mnemosyne -- mnemosyne-mcp
```

Claude Code will automatically call `mnemosyne.search` for code understanding, `mnemosyne.index` to build the index, and `mnemosyne.stats` for index info -- no manual CLI steps required. Everything runs locally over stdio. No API calls, no data egress.

See the full [MCP Server Reference](https://github.com/castnettech/mnemosyne/blob/main/MCP.md) for configuration, tools, and integration details.

## Ollama Integration (Local LLMs)

For local LLM code search via [Ollama](https://ollama.com), install the Ollama bridge:

```bash
pip install mnemosyne-ollama
cd /your/project
mnemosyne-ollama "how does authentication work"
```

Auto-detects your Ollama model (Qwen, Llama, Phi, etc.), indexes if needed, searches with hybrid retrieval, and returns answers with file citations. Zero config, zero cloud.

See the [Ollama bridge README](https://github.com/castnettech/mnemosyne/blob/main/ollama/README.md) for details.

## LLM Agent Integration (CLI)

For agents that run shell commands (Cursor, Aider, Copilot, etc.), add to your `CLAUDE.md`, `.cursorrules`, or equivalent instruction file:

```
Before answering questions about this codebase, run:
  ! mnemosyne query "<question>" --budget 8000
Use the returned chunks as primary context. Only read additional files if needed.
```

Works with any agent that can execute shell commands.

## CLI

| Command | Purpose |
|---|---|
| `init` | Create workspace and config |
| `ingest` | Index files (incremental, `--full` to rebuild) |
| `query` | Search with token budget (`--docs`, `--all` for document partition) |
| `stats` | Index and cache statistics |
| `schema-ingest` | Import database schema (DDL, JSON, SQLite) |
| `schema-stats` | Schema index statistics |
| `compress` | Preview compression for a file |
| `delta` | Show changes since last index |
| `cache` | Manage ARC cache (`show`, `clear`, `warm`) |
| `daemon` | Persistent server for warm-start queries |
| `analytics` | Precision metrics and usage patterns |
| `audit` | Operation log |
| `health` | Index integrity checks |
| `gc` | Garbage collect stale data |
| `benchmark` | Run precision benchmarks |

## Documentation

| Document | Contents |
|---|---|
| [MCP.md](https://github.com/castnettech/mnemosyne/blob/main/MCP.md) | MCP server reference -- installation, tools, configuration, architecture |
| [ollama/README.md](https://github.com/castnettech/mnemosyne/blob/main/ollama/README.md) | Ollama bridge -- local LLM code search via tool-calling models |
| [REFERENCE.md](https://github.com/castnettech/mnemosyne/blob/main/REFERENCE.md) | Full CLI reference, configuration, architecture, integration guides |
| [ALGORITHMS.md](https://github.com/castnettech/mnemosyne/blob/main/ALGORITHMS.md) | Algorithm details with academic paper references |
| [TUNING.md](https://github.com/castnettech/mnemosyne/blob/main/TUNING.md) | Precision tuning guide |
| [CHANGELOG.md](https://github.com/castnettech/mnemosyne/blob/main/CHANGELOG.md) | Version history |

## Trademarks

All third-party product names (Claude Code, Ollama, Qwen, Llama, Gemma, Phi, Mistral, Command-R, Cursor, Aider, Copilot) are trademarks of their respective owners. Mnemosyne is an independent project and is not endorsed by or affiliated with any of these companies. Product names are used solely to describe compatibility.

## License

Dual-licensed: [AGPL-3.0](https://github.com/castnettech/mnemosyne/blob/main/LICENSE) for open-source use | [Commercial license](https://github.com/castnettech/mnemosyne/blob/main/COMMERCIAL-LICENSE.md) for proprietary embedding.

Copyright 2026 Cast Rock Innovation L.L.C. (DBA: [Cast Net Technology](https://castnettechnology.com))
