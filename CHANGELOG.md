# Changelog

All notable changes to Mnemosyne are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Default `general.ignore_patterns` extended to cover common build-output
  directories and runtime artifacts that the prior set silently allowed
  into the index. Triggering case: a .NET project whose
  `bin/Release/*.deps.json` and `*.runtimeconfig.json` files (NuGet
  build manifests, hundreds of KB of high-IDF JSON) ended up dominating
  BM25 scoring on unrelated queries -- the doc-retrieval lane would
  surface a build manifest as a top cite for prompts that had nothing
  to do with .NET.

  Additions:
  - Generic build dirs: `bin`, `obj`, `target`, `build`, `dist`, `out`,
    `vendor`, `cmake-build-*`, `htmlcov`, `*.egg-info`
  - Compiled artifacts: `*.class`, `*.jar`, `*.war`, `*.ear`, `*.dll`,
    `*.exe`, `*.pdb`, `*.dylib`, `*.so`, `*.o`, `*.a`
  - .NET runtime descriptors: `*.deps.json`, `*.runtimeconfig.json`,
    `*.runtimeconfig.dev.json`
  - TypeScript build cache: `*.tsbuildinfo`

  The hidden-dir glob (`.*`) already covered `.venv`, `.pytest_cache`,
  `.mypy_cache`, `.tox`, `.gradle`, `.next`, `.nuxt`, `.cache`,
  `.dotnet`, etc. -- comment expanded for clarity, no behavior change
  there.

  User-config semantics unchanged: a project that defines its own
  `general.ignore_patterns` in `.mnemosyne/config.toml` still wins via
  the existing list-union deep-merge.

  **Behavior change for already-indexed projects:** a re-index
  (`mnemosyne ingest --full`) is required to evict polluted rows.
  Projects that intentionally indexed files matching the new patterns
  (rare but possible: a `bin/` directory of shell scripts, or
  compliance scans against `*.deps.json`) can override by setting
  `general.ignore_patterns` in `.mnemosyne/config.toml`.

## [1.1.0] - 2026-04-07

### Added
- Document ingestion (Tier 0) -- extract and index PDFs, DOCX, CSV, and
  plaintext (.log, .cfg, .ini, .conf, .rst, .xml). PDF requires optional
  `mnemosyne-engine[pdf]` extra (pypdf, pure Python, BSD). All other
  extractors use stdlib only. Zero-dep core preserved.
- Partitioned document index -- documents stored in isolated tables
  (doc_chunks, doc_chunks_fts, doc_sparse_embeddings, doc_vocabulary) within
  the same SQLite database. Code retrieval pipeline is completely untouched.
  New DocStore and DocRetrievalEngine with BM25 + TF-IDF fused via RRF.
- Schema ingestion pipeline -- DDL-aware SQL chunking (Phase 1), JSON/YAML
  schema import with environment tagging (Phase 2), SQLite introspection via
  PRAGMA queries with `--sqlite` CLI flag (Phase 3).
- CLI `--docs` flag -- search document partition only.
- CLI `--all` flag -- search both code and document partitions with split budget.
- CLI `schema-ingest` and `schema-stats` commands for database DDL ingestion.
- CLI `--version` flag -- reads from package `__version__`.
- One-time upgrade hint -- on first query or ingest after schema migration,
  a stderr message reminds users about the new `--all` and `--docs` flags.
  Fires once, then clears. Fresh installs are not affected.
- Progress bar on `mnemosyne ingest` with live counter and file path display.
- Document partition benchmark -- 18-test regression gate covering document
  retrieval, schema lookup, cross-partition queries, and partition isolation.

### Fixed
- Document file size limit -- documents were silently skipped using the 512 KB
  code limit. Now uses extraction.max_file_size_kb (default 10 MB).
- DocStore vocabulary adapter -- doc_tfidf was created with store=None at all
  call sites, so doc_vocabulary was never populated. Added save_vocabulary()
  and load_vocabulary() adapter methods, wired all call sites. Re-ingest
  required after upgrade to populate document vocabulary.

### Changed
- .md and .txt files reclassified as documents -- route to the doc partition
  instead of competing with source code for ranking slots. Use `--all` to
  search both partitions.
- Dotfile ignore consolidation -- replaced 12 individual patterns with a
  single `.*` glob that catches all dotfiles and dotdirs.
- Schema migration 2->3: files table gains extraction_method,
  extraction_quality, page_count; chunks table gains page_number.
- Schema migration 3->4: creates doc_chunks, doc_chunks_fts,
  doc_sparse_embeddings, doc_vocabulary tables.

### Upgrade notes
- Run `mnemosyne ingest --full` after upgrading to populate document
  partitions and vocabulary. Existing code indexes are preserved.
- Queries that previously found .md files via default `mnemosyne query` must
  now use `--all` or `--docs`.

---

### mnemosyne-mcp [0.2.0]

### Added
- `search_docs` tool -- document-only MCP search for LLM agents.
- `schema_ingest` tool -- import DDL files or introspect SQLite databases.
- `schema_stats` tool -- report indexed schema sources and statistics.
- Federated search -- `search` tool now queries both code and document
  partitions, returns labeled sections ("Code Results" / "Document Results").

### Fixed
- Per-partition truncation -- the 64KB output cap was applied after
  concatenation, silently removing document results when code results
  exceeded the cap. Each partition now gets an independent budget (32KB each
  when both exist, full 64KB when only one exists).
- Stats handler tuple unpack -- _handle_stats unpacked 3 values but
  _get_engine returns 5 since the partition refactor.
- doc_tfidf store wiring at 2 server.py call sites.

---

### mnemosyne-ollama [0.1.1]

### Added
- Partition visibility in verbose mode -- `-v` output shows code/docs chunk
  counts and top file paths per partition for `search` and `search_docs` calls.

### Fixed
- Version flag now imports from package `__version__` instead of hardcoded
  string.

---

## [1.0.5] - 2026-04-04

### Fixed
- README.md -- all relative links (image, doc references, badge hrefs) converted
  to absolute GitHub URLs. PyPI was showing broken image and dead links because it
  cannot resolve relative paths.
- REFERENCE.md -- replaced all 10 Mermaid code blocks (5 unique diagrams x 2
  duplicate sections) with pre-rendered PNG images. GitHub strips foreignObject
  elements from SVGs for security, which caused all node text to vanish. PNGs
  render identically on GitHub and PyPI. Diagrams stored in docs/assets/diagrams/.

### Added
- mnemosyne-ollama (v0.1.0) -- lightweight MCP host that bridges Ollama's
  tool-calling API to mnemosyne-mcp. Zero new dependencies beyond mnemosyne-mcp.
  Auto-detects tool-capable models (Gemma 3/4, Llama 3.x/4, Qwen 2.5/3, Phi-4,
  Mistral-Nemo, Command-R). Single-shot and interactive CLI modes. ~210 lines of
  implementation code. Includes OIDC publish workflow (publish-ollama.yml).
- Ollama Integration section in README.md with quickstart.
- ollama/README.md with full usage documentation.
- docs/mnemosyne_ollama_plan.md -- design document for the Ollama bridge.

### Changed
- Version bump: 1.0.4 -> 1.0.5.

## [1.0.4] - 2026-04-02

### Added
- PyPI trusted publisher workflow -- OIDC authentication, SHA-pinned actions,
  environment-gated deployment with required reviewer approval, signed SLSA
  artifact attestation. Zero secrets or tokens in CI.

### Fixed
- Pinned SHA for `pypa/gh-action-pypi-publish` action (v1.12.4).

## [1.0.2] - 2026-04-02

### Fixed
- Self-ingestion bug -- mnemosyne's own source package (`mnemosyne/`) was not
  excluded from default `ignore_patterns`, causing the engine to index itself
  when run from the repo root or when the package directory existed in a project.
  Added `"mnemosyne"` to the default ignore list alongside `.mnemosyne`.
- Benchmark suite root path -- `httpx.json` had an empty `root` field that
  resolved to the mnemosyne package directory instead of the httpx corpus.
  Updated to use the same `/tmp` clone path as `httpx_holdback.json`.
- Removed manual `"mnemosyne"` exclusion workaround from `benchmark_suite.py`
  `_setup_mnemosyne()` -- no longer needed with the default ignore fix.

### Changed
- README.md rewritten -- condensed from 1,280 lines to ~120 lines. Repositioned
  as a standard GitHub project page with logo, install, benchmarks, features,
  and use cases. Detailed documentation moved to REFERENCE.md.
- Added REFERENCE.md -- full CLI reference, configuration, architecture,
  key innovations, and integration guides (content preserved from original README).
- Replaced `pipeline.png` with `mnemosyne.png` as centered repo logo.

### Benchmark results (post-fix)
- httpx (6 questions): File recall 0% -> **80.6%** (was indexing wrong project)
- Aggregate (46 questions): File recall 76.8% -> **87.3%**
- Regression gate: 5/5 passing

## [1.0.0] - 2026-03-30

### Added
- Dense embedding backend -- optional 23MB ONNX model (all-MiniLM-L6-v2-code-search-512)
  as 6th retrieval signal via Reciprocal Rank Fusion. Bridges vocabulary gap
  when query terms don't match code identifiers. Opt-in via config, downloads
  on first use, runs 100% locally. No data egress.
- Porter stemmer for TF-IDF tokenizer -- aligns with BM25/FTS5 porter stemming.
- Code-aware stopwords -- `get`, `set`, `for`, `is`, `has` preserved in code
  identifier splitting instead of being stripped as English stopwords.
- Two-pass soft file filter -- files containing top-50 individual chunks survive
  at 0.7x penalty even if their aggregate file score is low.
- Symbol match file promotion -- files with strong symbol matches get guaranteed
  filter slots.
- Decorator inclusion -- function/class chunks now include decorator blocks.
- Chunk enrichment -- TF-IDF embeddings include file path and module docstring
  context for better semantic matching.

## [0.4.0] - 2026-03-29

### Added
- `mnemosyne health` command -- reports index age, file/chunk/token counts,
  vocabulary size, stale files, FTS5 integrity, tokenizer hash, and daemon
  status. `--json` flag for programmatic monitoring.
- `--log-format` global flag (text/json) for structured logging on all commands.
- `--log-level` global flag (DEBUG/INFO/WARNING/ERROR) for log verbosity.
- `mnemosyne benchmark` command for running retrieval-quality benchmarks
  against any corpus with JSON question sets.
- httpx 0.28.1 benchmark regression gate (5 assertions, `@pytest.mark.benchmark`).
- Retrieval metrics: `hit_at_3`, `relevant_at_5`, `mrr_at_10` in BenchmarkSuite.
- Symbol type-aware ranking -- `_symbol_search` returns chunk_type; class
  definitions receive 4x boost when PascalCase/TitleCase query detected.
- Hybrid file aggregation in `_file_level_filter` -- max + 0.1*sum prevents
  files with many weak chunks from volume-dominating over precise matches.
- Ingest directory walk -- `mnemosyne ingest src/` now expands directories
  instead of silently dropping them.
- Path containment validation -- ingest CLI rejects paths outside project root,
  resolves symlinks with `os.path.realpath()`.

### Fixed
- FTS5 escape regex -- commas, periods, hyphens, and other punctuation silently
  caused BM25 to return 0 results for natural language queries.
- `_split_class_chunk` now preserves parent chunk_type instead of hardcoding
  "block", enabling class-name boost for split class definitions.
- Daemon socket permissions set to 0600 after bind (CVE-class prevention).
- Daemon ingest path validation -- rejects paths outside project root.
- Six silent `except:pass` blocks replaced with proper WARNING/ERROR logging.

### Changed
- Ingest `_scan_files()` refactored into `_scan_dir()` shared helper for
  both full-scan and targeted path resolution.
- `_file_level_filter` aggregation changed from pure sum to hybrid
  (max + 0.1*sum) -- improves precision for single-file queries.

### Retrieval quality (httpx 0.28.1 benchmark)
- hit@3: 0.667 -> 1.000
- relevant@5: 3.17 -> 3.83
- mrr@10: 0.597 -> 0.778
- All 6 queries now return at least one gold file in top 3.

## [0.3.0] - 2026-03-28

Initial public release on PyPI as `mnemosyne-engine`.

- Hybrid retrieval: BM25 (FTS5) + TF-IDF + usage frequency + symbol search
- Reciprocal Rank Fusion across all signals
- Python/Markdown/Go/Rust/C#/Java/Kotlin chunkers
- ARC cache with ghost lists and tiered storage
- Token compression (40-70% reduction)
- Daemon mode with Unix socket RPC
- Zero runtime dependencies

[1.1.0]: https://github.com/castnettech/mnemosyne/compare/v1.0.5...v1.1.0
[1.0.5]: https://github.com/castnettech/mnemosyne/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/castnettech/mnemosyne/compare/v1.0.2...v1.0.4
[1.0.2]: https://github.com/castnettech/mnemosyne/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/castnettech/mnemosyne/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/castnettech/mnemosyne/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/castnettech/mnemosyne/releases/tag/v0.3.0
