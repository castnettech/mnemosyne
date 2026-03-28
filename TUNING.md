# Mnemosyne — Precision Tuning Guide

## Version 0.2.0 — Updated 2026-03-22

---

## Clearing the Index (Reset)

```bash
# Full reset — delete everything and start fresh
rm -rf .mnemosyne
python3 -m mnemosyne init
python3 -m mnemosyne ingest

# Or re-ingest from scratch (purges stale files automatically)
python3 -m mnemosyne ingest --full
```

The `--full` flag now purges file records for files that no longer match
the scan criteria (deleted files, newly-ignored patterns). Stale chunks
from previous runs are automatically removed.

---

## How Retrieval Ranking Works (v0.2.0)

Understanding the pipeline helps you tune it. Each query passes through:

```
BM25 (FTS5)  +  TF-IDF vector  +  Symbol name match  +  Usage frequency
                        |
                   RRF fusion (weighted merge of ranked lists)
                        |
                Symbol match multiplier (3x for exact matches)
                        |
                Filename boost (1.5x when query terms match filename)
                        |
           Import/namespace graph injection (connected files added)
                        |
              File-level filter (top 6 files by aggregate score)
                        |
        Cost-model re-rank (boilerplate penalty, code boost, test penalty)
                        |
                Budget cut (greedy selection within token budget)
```

Each stage has tunable parameters. The defaults work well for most
codebases. Tune only when benchmark data shows a specific gap.

---

## What Improves Retrieval Precision

### 1. Exclude non-source directories (biggest win)

Edit `.mnemosyne/config.toml`:

```toml
[general]
ignore_patterns = ["marketing", "docs", "vendor", "dist", "build"]
```

These patterns are **added to** the hardened defaults (which include `.git`,
`node_modules`, `__pycache__`, `.env`, `*.pem`, `*.key`, `credentials.json`,
`package-lock.json`, etc.). You cannot accidentally remove security patterns —
the config merge uses list union.

### 2. Use language-aware chunking

Mnemosyne v0.2.0 includes dedicated chunkers:

| Language | Chunker | Symbol extraction |
|---|---|---|
| Python | AST-based (`CodeChunker`) | function/class names via `ast.parse` |
| JavaScript/TypeScript | Regex-structural (`JSChunker`) | function, class, const/let/var declarations, object literals |
| Markdown/Text | Heading-based (`TextChunker`) | paragraph boundaries |
| Everything else | Sliding window (`GenericChunker`) | none |

The JS chunker is new in v0.2.0. It extracts `symbol_name` from:
- `function foo()` / `async function foo()`
- `class Foo`
- `const foo = () =>` / `const foo = function()`
- `const PATTERNS = { ... }` (object/array constants)
- Class method definitions

Chunks with `symbol_name` receive a **2x ranking boost** in the cost model.
If your project uses a language not listed above, chunks default to
`GenericChunker` with no symbol names and no boost.

### 3. Tune retrieval weights for your workload

```toml
[retrieval]
# Code-heavy projects: boost BM25 (exact keyword match)
bm25_weight = 0.5
vector_weight = 0.3
usage_weight = 0.2

# Documentation-heavy projects: boost TF-IDF (semantic similarity)
bm25_weight = 0.3
vector_weight = 0.5
usage_weight = 0.2
```

### 4. Adjust the token budget

```bash
# Tight budget = fewer results, higher precision
python3 -m mnemosyne query "auth middleware" --budget 2000

# Generous budget = more results, higher recall
python3 -m mnemosyne query "auth middleware" --budget 12000
```

The default is 8000 tokens. For single-function lookups, 2000 is enough.
For architectural questions spanning multiple files, 8000-12000 is better.

### 5. Lower `tfidf_min_df` for small projects

```toml
[embedding]
tfidf_min_df = 1
```

Default is 1 (changed from 2 in v0.2.0). This keeps terms that appear in
only one file, which are often the most discriminative for retrieval.
Increase to 2 if your project has many one-off junk tokens inflating the
vocabulary.

---

## What Improves Retrieval Recall

### 1. The import/dependency graph (automatic)

v0.2.0 scans retrieved files for `import`, `require()`, and runtime
namespace access patterns (e.g., `MyApp.Utils`). Connected files are
injected into results even if they share no keywords with the query.

This is how `utils.js` gets surfaced when `analyzer.js` is found — the
graph detects `var utils = PrivacyPeep.Utils` and injects utils.js.

No configuration needed. Works for any JS/TS/Python project with standard
import patterns.

### 2. Filename boost (automatic)

If query terms match a file's name (4-char prefix matching), all chunks
from that file get a 1.5x score boost. Query "scoring pipeline" boosts
`scorer.js`. Query "comparison" boosts `comparator.js`.

No configuration needed.

### 3. Symbol name search (automatic)

If the query contains an identifier like `isNegated`, `calculateScore`,
or `analyzePolicy`, chunks with matching `symbol_name` get a 3x multiplier
after RRF fusion.

No configuration needed. Requires the language-aware chunker to extract
symbol names (Python and JS/TS supported).

---

## What Reduces Noise

### Automatic penalties (no configuration)

| Signal | Effect |
|---|---|
| HTML/CSS/Markdown/TXT chunks | 0.85 boilerplate penalty — demoted below code |
| Test directory chunks (`tests/`, `test/`) | 0.5 boilerplate penalty — secondary to source |
| Boilerplate code patterns (imports, logging, assignments) | Detected by `density.py`, penalized proportionally |
| Chunks without `symbol_name` | No 2x structured code boost — lose to named chunks |

### Manual exclusions

For project-specific noise (e.g., generated code, vendor directories):

```toml
[general]
ignore_patterns = ["generated/", "vendor/", "*.generated.ts"]
```

---

## Benchmarking Your Project

Run the built-in benchmark to measure retrieval quality:

```bash
python3 -m mnemosyne.tests.benchmark --project-root /path/to/project --budget 4000
```

The benchmark reports:
- **Token reduction** — raw tokens vs. mnemosyne tokens per query
- **Retrieval precision** — fraction of retrieved files that are ground truth
- **Retrieval recall** — fraction of ground truth files that are retrieved
- **Compression ratios** — per-file compression effectiveness
- **Speed** — ingest time, query latency, baseline read time
- **Storage** — index size vs. raw source size

### Current benchmark results (policylens, 25 files, no project-specific overrides)

| Metric | Value |
|---|---|
| Precision | 40.7% |
| Recall | 91.7% |
| Queries at 100% recall | 8 of 10 |
| Token reduction (large files) | 50-77% |
| Compression ratio | 38.9% |
| Ingest time | 0.5s |
| Query latency | ~60ms |
| Storage overhead | 0.25x |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Security patterns missing from index | TOML was overriding defaults | Fixed in v0.2.0 — list union, not replacement |
| Test files dominate all queries | Tests exercise many features | Automatic 0.5 test penalty in v0.2.0 |
| HTML/legal pages outrank source code | Prose matches query keywords | Automatic 0.85 HTML penalty in v0.2.0 |
| Utility files never found | No keyword overlap | Import graph auto-injects connected files |
| `package-lock.json` indexed | Missing from ignore list | Now in hardened defaults |
| Query returns 0 results | Index empty or stale | Run `python3 -m mnemosyne ingest --full` |
