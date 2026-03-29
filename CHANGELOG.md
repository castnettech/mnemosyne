# Changelog

All notable changes to Mnemosyne are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-03-29

### Added
- `mnemosyne health` command — reports index age, file/chunk/token counts,
  vocabulary size, stale files, FTS5 integrity, tokenizer hash, and daemon
  status. `--json` flag for programmatic monitoring.
- `--log-format` global flag (text/json) for structured logging on all commands.
- `--log-level` global flag (DEBUG/INFO/WARNING/ERROR) for log verbosity.
- `mnemosyne benchmark` command for running retrieval-quality benchmarks
  against any corpus with JSON question sets.
- httpx 0.28.1 benchmark regression gate (5 assertions, `@pytest.mark.benchmark`).
- Retrieval metrics: `hit_at_3`, `relevant_at_5`, `mrr_at_10` in BenchmarkSuite.
- Symbol type-aware ranking — `_symbol_search` returns chunk_type; class
  definitions receive 4x boost when PascalCase/TitleCase query detected.
- Hybrid file aggregation in `_file_level_filter` — max + 0.1*sum prevents
  files with many weak chunks from volume-dominating over precise matches.
- Ingest directory walk — `mnemosyne ingest src/` now expands directories
  instead of silently dropping them.
- Path containment validation — ingest CLI rejects paths outside project root,
  resolves symlinks with `os.path.realpath()`.

### Fixed
- FTS5 escape regex — commas, periods, hyphens, and other punctuation silently
  caused BM25 to return 0 results for natural language queries.
- `_split_class_chunk` now preserves parent chunk_type instead of hardcoding
  "block", enabling class-name boost for split class definitions.
- Daemon socket permissions set to 0600 after bind (CVE-class prevention).
- Daemon ingest path validation — rejects paths outside project root.
- Six silent `except:pass` blocks replaced with proper WARNING/ERROR logging.

### Changed
- Ingest `_scan_files()` refactored into `_scan_dir()` shared helper for
  both full-scan and targeted path resolution.
- `_file_level_filter` aggregation changed from pure sum to hybrid
  (max + 0.1*sum) — improves precision for single-file queries.

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

[0.4.0]: https://github.com/castnettech/mnemosyne/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/castnettech/mnemosyne/releases/tag/v0.3.0
