# Changelog

All notable changes to Mnemosyne are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.4]: https://github.com/castnettech/mnemosyne/compare/v1.0.2...v1.0.4
[1.0.2]: https://github.com/castnettech/mnemosyne/compare/v1.0.0...v1.0.2
[1.0.0]: https://github.com/castnettech/mnemosyne/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/castnettech/mnemosyne/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/castnettech/mnemosyne/releases/tag/v0.3.0
