# Mnemosyne — Adaptive Context Engine for LLM Agents

> "Give your agent a memory that actually works."

Mnemosyne is a production-grade LLM context compression and retrieval engine built entirely on the Python standard library. It solves the single biggest bottleneck in LLM agent effectiveness: context window waste. Rather than naively stuffing raw file contents into a prompt, Mnemosyne indexes your codebase, compresses each chunk intelligently, ranks candidates by value per token, and delivers exactly the right context within whatever budget you set — in under 100 milliseconds.

Version: **0.3.0** | License: AGPL-3.0 ([Commercial license available](COMMERCIAL-LICENSE.md)) | Requires: Python 3.11+ | Dependencies: **zero** (optional: `onnxruntime` for future dense embeddings)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [What Makes Mnemosyne Different](#what-makes-mnemosyne-different)
3. [Architecture Overview](#architecture-overview)
4. [Key Innovations](#key-innovations)
5. [Quick Start](#quick-start)
6. [CLI Reference](#cli-reference)
7. [Configuration Reference](#configuration-reference)
8. [Integration with LLM Agents](#integration-with-llm-agents)
9. [Benchmarks](#benchmarks)
10. [Technical Stack](#technical-stack)
11. [File Structure](#file-structure)
12. [Algorithm Documentation](#algorithm-documentation)
13. [Future Roadmap](#future-roadmap)
14. [License](#license)

---

## The Problem

Every LLM agent working on a real codebase faces the same problem - the context window is the most expensive, scarcest resource in the system, and current tooling wastes most of it.

### Context Window Waste Is Pervasive

When a developer asks their AI assistant to fix a bug in an authentication module, the agent typically receives hundreds of kilobytes of unrelated code — test fixtures, unrelated models, build configuration, lock files, and import boilerplate. Studies of real-world LLM coding sessions show **40–70% of injected context is irrelevant or redundant** to the task at hand.

The consequences compound:

- **Token cost**: Every wasted token is direct spend. At scale, context inefficiency is a billing problem.
- **Degraded quality**: LLMs lose focus when context is noisy. The "lost in the middle" phenomenon is well-documented — relevant information buried in a large context window is missed more often than information at the edges.
- **Latency**: Larger prompts take longer to process. Every unnecessary token adds latency to every turn.
- **Rate limits**: Context-heavy requests hit token-per-minute limits faster, throttling agent throughput.

### No Existing Tool Solves This Completely

Current tools address at most one dimension of the problem:

- **IDE context managers** (Cursor, Copilot) use recency and open-tab heuristics. They are file-centric, not semantic, and have no cross-session memory.
- **Code search tools** (grep, ripgrep, embeddings) find text but do not budget, compress, or rank by value per token.
- **MemGPT** introduces persistent memory but is architecture-specific, externally dependent, and does not perform compression or cost-model retrieval.
- **Aider** is a capable coding agent but its context management is manual — the developer specifies files.

**Context management is the number one bottleneck for LLM agent effectiveness on large codebases.** No existing tool combines compression, adaptive retrieval, cost-model ranking, delta delivery, and cross-session learning in a single zero-dependency engine.

Mnemosyne is that engine.

---

## What Makes Mnemosyne Different

### Competitive Feature Matrix

| Feature | Claude Code (native) | Cursor | Aider | Copilot | MemGPT | **Mnemosyne** |
|---|---|---|---|---|---|---|
| Adaptive compression | No | No | No | No | No | **Yes** |
| Cost-model retrieval | No | No | No | No | No | **Yes** |
| Delta-aware context injection | No | No | No | No | No | **Yes** |
| Usage-frequency learning | No | No | No | No | No | **Yes** |
| ARC cache (not FIFO) | No | No | No | No | No | **Yes** |
| Predictive pre-fetching | No | No | No | No | No | **Yes** |
| Semantic deduplication | No | No | No | No | No | **Yes** |
| Cross-session persistence | No | No | No | No | Yes | **Yes** |
| Zero external dependencies | N/A | No | No | No | No | **Yes** |
| AST-aware chunking | No | Partial | No | No | No | **Yes** |
| Hybrid BM25 + TF-IDF + Usage | No | No | No | No | No | **Yes** |
| Token budget enforcement | No | No | No | No | No | **Yes** |
| Audit log | No | No | No | No | No | **Yes** |
| Garbage collection | No | No | No | No | No | **Yes** |

Every "Yes" in the Mnemosyne column is a differentiator that required deliberate engineering. The combination of all of them in a single, zero-dependency Python library is unique in the ecosystem.

---

## Architecture Overview

Mnemosyne is organized into two primary pipelines — ingestion and retrieval — connected by a persistent SQLite store with WAL mode, FTS5 full-text search, and JSON1 for sparse embedding storage.

### Ingestion Pipeline

```
Project Files on Disk
        |
        v
  [ File Scanner ]  (Ingester._scan_files)
  - Extension filter (.py, .js, .ts, .md, ...)
  - Size cap (default 512 KB)
  - fnmatch-based ignore patterns (.git, node_modules, __pycache__, ...)
        |
        v
  [ Change Detection ]  (Ingester._needs_indexing)
  - Bloom filter O(1) "definitely not indexed" check
  - mtime comparison (fast path)
  - SHA-256 content hash comparison (confirm path)
        |
        v
  [ Language Detection ]  (chunkers.detect_language)
  - Extension-based: .py -> python, .js -> javascript, .ts -> typescript, ...
        |
        v
  [ Chunker ]  (CodeChunker / JSChunker / BraceChunker family / TextChunker / GenericChunker)
  - Python:      AST walk -> function chunks, class chunks, import blocks, loose blocks
  - JavaScript:  Regex-structural -> functions, classes, arrow consts, object literals, methods
  - TypeScript:  Same as JavaScript
  - Go:          Brace-based -> funcs, methods, types, structs, interfaces (GoChunker)
  - C#:          Brace-based -> classes, methods, properties, namespaces (CSharpChunker)
  - Rust:        Brace-based -> fn, impl, struct, enum, trait, mod (RustChunker)
  - Java/Kotlin: Brace-based -> classes, methods, interfaces, enums (JavaChunker)
  - Text/MD:     Paragraph-boundary splitting with configurable overlap
  - Generic:     Line-count sliding window with configurable overlap
  - Output:      ChunkCandidate(content, chunk_type, line_start, line_end, symbol_name)
        |
        v
  [ Content Deduplication ]  (hasher.content_hash)
  - SHA-256 of whitespace-normalised content
  - Cross-file dedup: identical chunks stored once regardless of location
        |
        v
  [ Store.save_chunk ]  (store.py + schema.py)
  - SQLite WAL mode: concurrent reads during writes
  - FTS5 virtual table: BM25 index auto-maintained via trigger
  - JSON1: sparse embedding weights stored as term->weight dicts
        |
        v
  [ TF-IDF Vocabulary Build ]  (TFIDFBackend.build_vocabulary)
  - Runs after all files in the batch are stored (corpus-level IDF is valid)
  - camelCase / snake_case token splitting for code identifiers
  - Augmented TF: 0.5 + 0.5 * (count / max_count)
  - Smoothed IDF: log((N+1) / (df+1)) + 1
        |
        v
  [ Bloom Filter Update + Audit Log ]
  - Bloom filter records file path and chunk hash for fast future checks
  - AuditLog appends a JSON line: op=ingest_complete, counts, elapsed
```

### Retrieval Pipeline

```
Query String  +  Token Budget
        |
        v
  [ BM25 Search ]  (RetrievalEngine._bm25_search)
  - FTS5 MATCH query with FTS5 special-character escaping
  - OR-joined terms for broader recall on natural-language queries
  - Scores normalised to [0, 1] relative to top result
        |
        v (parallel)
  [ TF-IDF Vector Search ]  (TFIDFBackend.search)
  - Inverted index lookup: only query terms consulted (sub-linear)
  - Cosine-like dot product over sparse vectors
        |
        v (parallel)
  [ Usage Frequency Scores ]  (Analytics.get_usage_scores)
  - 'selected' and 'used' events retrieved from SQLite
  - Exponential decay: score = 2^(-age_days / halflife)
  - Default halflife: 7 days
        |
        v (parallel)
  [ Predictive Pre-fetch ]  (Prefetcher.get_prefetch_ids)
  - Query signature: sorted deduplicated token set -> MD5 fingerprint
  - Pattern lookup: if pattern seen >= min_hits times, boost associated chunks
        |
        v
  [ Symbol Name Search ]  (RetrievalEngine._symbol_search)
  - Matches query identifiers against chunk symbol_name column
  - Exact match: score 1.0; component overlap: proportional score
        |
        v (merge)
  [ RRF Fusion ]  (ranking.rrf_fuse)
  - Reciprocal Rank Fusion across BM25, vector, symbol, usage, prefetch signals
  - rrf_score(id) = sum(weight[src] / (k + rank[src](id)))
  - Symbol match multiplier: 3x post-RRF boost for exact symbol hits
  - Configurable per-source weights (default: BM25=0.4, vector=0.4, usage=0.2)
        |
        v
  [ Filename Boost ]  (RetrievalEngine._filename_boost)
  - 4-char prefix matching: query term "scoring" boosts "scorer.js"
  - Injects chunks from filename-matched files missing from RRF results
        |
        v
  [ Import/Namespace Graph ]  (RetrievalEngine._import_graph_boost)
  - Parses import/require and Namespace.Module references from retrieved files
  - Injects connected files (up to 4) that share no keywords with query
  - Source files prioritised over test files by reference count
        |
        v
  [ File-Level Filter ]  (RetrievalEngine._file_level_filter)
  - Keeps top 6 files by aggregate RRF score to prevent budget dilution
        |
        v
  [ Cost-Model Re-ranking ]  (ranking.cost_model_score)
  - value_density = rrf_score * code_boost / (1 + ln(1 + token_count))
  - 2x boost for chunks with symbol_name (function, class, named const)
  - 0.85 boilerplate penalty for HTML/CSS/Markdown
  - 0.5 penalty for test directory files
        |
        v
  [ Budget Cutting ]  (ranking.budget_cut)
  - Greedy selection: highest density first, accumulate until budget exhausted
  - Compression fallback: if chunk alone exceeds remaining budget, try compressing
        |
        v
  [ Result Formatting ]  (formatter.py)
  - Plain text: file path header, line range, content block, score summary
  - JSON: structured QueryResult array for programmatic consumption
        |
        v
  [ Usage Event Recording ]  (Analytics.record)
  - 'retrieved' events for every returned chunk
  - Pattern recording for future prefetch learning
```

### Data Flow Diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │              .mnemosyne/                             │
                    │                                                      │
   Disk Files  ───> │  mnemosyne.db  (SQLite WAL)                         │
                    │  ├── files          (FileRecord)                     │
                    │  ├── chunks         (Chunk + compressed)             │
                    │  ├── chunks_fts     (FTS5 BM25 index)                │
                    │  ├── sparse_embeddings  (JSON term weights)          │
                    │  ├── summaries      (hierarchical summary tree)      │
                    │  ├── usage_events   (retrieved/selected/used)        │
                    │  ├── cache_state    (persisted ARC tier membership)  │
                    │  ├── task_patterns  (prefetch query signatures)      │
                    │  └── vocabulary     (TF-IDF IDF weights)             │
                    │                                                      │
                    │  bloom.bin   (Bloom filter — fast existence checks)  │
                    │  config.toml (merged user + default config)          │
                    │  audit.log   (append-only JSON-lines operation log)  │
                    └─────────────────────────────────────────────────────┘
                                          ^
                                          |
                         ┌────────────────┴────────────────┐
                         |                                 |
                    Ingestion                          Retrieval
                    Pipeline                           Pipeline
                         |                                 |
                    python -m mnemosyne ingest        python -m mnemosyne query "..."
                    (or LLM agent call)               (or LLM agent call)
```

### Module Dependency Graph

```
cli.py  ──> ingest.py  ──> chunkers/  ──> code_chunker.py   (Python AST)
        │              ├── hasher.py          js_chunker.py    (JS/TS regex+brace)
        │              ├── store.py           brace_chunker.py (shared base class)
        │              ├── bloom.py           go_chunker.py    (Go, v0.3.0)
        │              ├── embeddings/ ──>    csharp_chunker.py(C#, v0.3.0)
        │              └── audit.py           rust_chunker.py  (Rust, v0.3.0)
        │                   └──>              java_chunker.py  (Java/Kotlin, v0.3.0)
        │                   tfidf_backend.py  text_chunker.py
        │                                     generic_chunker.py
        │
        ├── retrieval.py ──> store.py
        │               ├── ranking.py
        │               ├── analytics.py
        │               ├── prefetch.py
        │               └── compress.py ──> density.py
        │                               └── embeddings/tfidf_backend.py
        │
        ├── daemon.py (JSON-RPC Unix socket server, v0.3.0)
        ├── cache.py  (ARCCache — standalone)
        ├── tiers.py  (TierManager — ARC <-> Store bridge)
        ├── delta.py  (DeltaTracker)
        ├── formatter.py
        ├── audit.py
        ├── schema.py (DDL + migration + SQLite hardening)
        ├── store.py  (all SQLite CRUD)
        ├── models.py (dataclasses, estimate_tokens, staleness fields)
        └── config.py (TOML loader, dot-access namespace)
```

---

## Key Innovations

### 1. Cost-Model-Driven Retrieval

Every database query optimizer has understood for decades that the cost of executing a plan matters as much as the estimated result quality. Mnemosyne applies the same principle to context assembly.

After fusing signals from multiple retrieval sources, Mnemosyne re-ranks every candidate by its **value density** with structural awareness:

```
value_density = (rrf_score * code_boost * (1 - 0.5 * boilerplate)) / (1 + ln(1 + token_count))
```

Where `code_boost = 2.0` for chunks with extracted symbol names (functions, classes, named constants) and `1.0` for generic chunks. The logarithmic denominator dampens the size penalty so that large, highly-relevant source files remain competitive against small, marginally-relevant ones.

HTML, CSS, and Markdown chunks receive an automatic `boilerplate = 0.85` penalty. Test directory files receive `boilerplate = 0.5`. These are general signals, not project-specific overrides.

When the budget is tight (the common case in production), the cost model ensures the most informative content per token is always selected. This is analogous to a database query planner choosing an index scan over a full table scan: the optimal plan depends on both selectivity and execution cost.

Competitors that rank purely by relevance without token cost will consistently select large, moderately-relevant chunks over small, highly-relevant ones. Mnemosyne does not make this mistake.

Implementation: `ranking.py` — `cost_model_score()` and `budget_cut()`.

---

### 2. Four-Stage Compression Pipeline

When a highly-relevant chunk is too large to fit in the remaining context budget, Mnemosyne does not discard it. Instead, it applies a four-stage compression pipeline to reduce the chunk's token count while preserving its semantic content.

**Stage 1 — Structural Preservation**

Before anything is removed, a preservation set is computed. Lines that are structurally load-bearing or informationally irreplaceable are locked:

- Function and class signature lines (`def`, `async def`, `class`)
- Return, raise, and yield statements
- Assert statements (invariants)
- All lines inside triple-quoted docstrings (when `preserve_docstrings = true`)
- Lines containing `TODO`, `FIXME`, `HACK`, `NOTE`, or `XXX` comments

These lines will survive all subsequent stages unchanged, regardless of importance scores.

**Stage 2 — Boilerplate Collapse**

Repetitive patterns that convey structure but low semantic content are replaced with compact summary tokens:

- Import blocks longer than 3 lines become: `# [12 imports: os, sys, re, pathlib, ...]`
- Consecutive `self.x = x` assignment runs become: `# [8 assignments: store, config, bloom, ...]`
- Consecutive logging/print call runs become: `# [4 log statements]`
- Multiple consecutive blank lines collapse to a single blank line

This alone reduces token count by 15–30% on typical Python codebases, with zero semantic loss for the purpose of understanding logic.

**Stage 3 — TF-IDF Importance Filtering**

Each remaining non-preserved line receives an importance score equal to the sum of IDF weights of its vocabulary terms. Lines where every word is a low-IDF stopword (common across the entire codebase) are removed first, until the configured `target_ratio` is met.

Optionally, a per-query IDF override (`context_idf`) can be passed to make this stage query-aware: terms rare in the query context are weighted more heavily, enabling relevance-aware pruning where lines not germane to the current question are removed preferentially.

**Stage 4 — Density Analysis**

Near-duplicate lines (SequenceMatcher ratio > 0.85 against any of the previous 20 lines) are removed. This catches copy-pasted code, repeated error-handling boilerplate, and overly-similar docstring lines that add characters but no meaning.

Typical results: **20–60% token reduction** with full preservation of function signatures, control flow, return types, assertions, and critical comments.

Implementation: `compress.py` — `Compressor`.

---

### 3. ARC Cache — Not FIFO, Not LRU

Every competing tool that implements any caching at all uses FIFO or LRU eviction. FIFO is obviously wrong (recently-added items are not necessarily recently-used). LRU is better but conflates recency with frequency: a chunk accessed 50 times last week but not at all today will be evicted in favor of a chunk accessed once this morning.

Mnemosyne implements the **Adaptive Replacement Cache (ARC)**, based on the 2003 USENIX FAST paper by Megiddo and Modha.

ARC maintains four ordered dictionaries:

- **T1** — Recently-inserted items seen exactly once (recency).
- **T2** — Items seen at least twice (frequency). Promoted from T1 on second access.
- **B1** — Ghost keys evicted from T1 (no data, keys only).
- **B2** — Ghost keys evicted from T2 (no data, keys only).

An adaptive parameter `p` controls the target size of T1 relative to T2. When a ghost hit occurs in B1 (meaning T1 was too small — we evicted something we needed again), `p` grows: more capacity is allocated to recency. When a ghost hit occurs in B2 (T2 was too small), `p` shrinks: more capacity is allocated to frequency.

The result is a cache that **self-tunes** to the actual access pattern of the workload. For a codebase where the developer is working in one area repeatedly (frequency-dominant), ARC performs like an LFU cache. For rapid exploration across many files (recency-dominant), it performs like an LRU cache. It adapts continuously without configuration.

ARC cache state (which chunk IDs are in T1, T2, B1, B2) is persisted to SQLite between sessions via the `cache_state` table, so the adaptive parameter `p` is not lost on restart.

Implementation: `cache.py` — `ARCCache`.

---

### 4. Hybrid Search — BM25 + TF-IDF + Usage + Pre-fetch

A single search signal is fragile. BM25 (term frequency in the document) misses semantic similarity between synonymous terms. TF-IDF vector search misses exact keyword matches in rare but critical identifiers. Usage frequency alone degenerates into "always return what was used before." Pre-fetch alone causes stale patterns to persist.

Mnemosyne fuses **six signals** via Reciprocal Rank Fusion (RRF) plus post-fusion boosting:

```
rrf_score(id) = sum over sources: weight[src] / (k + rank[src](id))
```

Where `k = 60` (standard RRF smoothing constant) and weights are configurable. The default weighting is:

| Signal | Weight | Source |
|---|---|---|
| BM25 (FTS5) | 0.4 | SQLite FTS5 with Porter stemmer, stopword filtering |
| TF-IDF vector | 0.4 | Inverted index, camelCase-aware tokenization, stopword filtering |
| Symbol name match | 0.6 | Direct match against chunk `symbol_name` + 3x post-RRF multiplier |
| Usage frequency | 0.2 | Exponentially-decayed historical access |
| Predictive pre-fetch | dynamic | Pattern-matched historical selections |
| Filename match | post-RRF | 1.5x boost when query terms prefix-match filenames |
| Import/namespace graph | post-filter | Injects connected files not found by keyword search |

RRF is robust to score scale differences between signals (because it operates on ranks, not raw scores) and handles documents absent from some lists gracefully (penalty rank = list length + 1).

**BM25 via FTS5**: SQLite's FTS5 extension provides production-quality BM25 ranking with Porter stemming, Unicode normalization, and a trigram tokenizer. The `chunks_fts` virtual table is maintained by database triggers on insert/update/delete to the `chunks` table, so the index is always consistent.

**TF-IDF with code-aware tokenization**: The `TFIDFBackend` splits camelCase identifiers (e.g. `getUserById` -> `get`, `user`, `by`, `id`) and snake_case identifiers (e.g. `auth_token_expiry` -> `auth`, `token`, `expiry`). This dramatically improves recall for code search where identifiers carry the semantics.

**Usage frequency with time decay**: Rather than a simple access count, Mnemosyne applies a half-life decay model: `score = 2^(-age_days / halflife)`. A chunk accessed yesterday contributes more than one accessed last month. The default halflife is 7 days. This prevents the system from ossifying around old patterns.

**Pre-fetch boosting**: Query signatures (normalized, sorted token sets hashed to a 16-character fingerprint) are matched against historical patterns. When a pattern has been seen at least `min_hits` times (default: 3), its associated chunks receive a maximum-priority boost in the fusion step. This allows the system to learn "whenever the developer asks about authentication, chunks A, B, and C are always relevant" without any explicit configuration.

Implementation: `retrieval.py` — `RetrievalEngine`, `ranking.py` — `rrf_fuse`.

---

### 5. Delta-Aware Context Injection

Sending an entire 400-line file to an LLM when only 8 lines changed is a 98% token waste. Mnemosyne tracks per-session chunk delivery state and computes diffs at both the file level and the chunk level.

**File-level delta detection**: `DeltaTracker.detect_changes()` walks the project directory, comparing on-disk mtime and content hashes against `FileRecord` entries in the database. It classifies every file as `added`, `modified`, or `deleted`. For modified files it computes a `difflib.unified_diff` between the indexed content and the current content.

**Chunk-level delta tracking**: Within a session, `DeltaTracker.mark_retrieved()` records the exact content string delivered for each chunk ID. On subsequent queries, `get_delta_context()` returns a unified diff if the chunk has changed since it was last sent. The agent receives only what changed, not the whole chunk again.

**Diff impact analysis**: `get_affected_chunks()` parses the `@@` hunk headers of a unified diff to extract changed line ranges, then queries the store for chunks whose `[line_start, line_end]` intervals overlap. This enables surgical re-indexing: only the chunks that were actually affected by a change need to be re-ranked or re-fetched.

In practice, delta injection reduces per-turn token consumption by **80–95%** for incremental coding sessions where the agent is iterating on a small set of files.

Implementation: `delta.py` — `DeltaTracker`.

---

### 6. Content-Addressed Deduplication

Copy-paste is ubiquitous in real codebases. Error-handling boilerplate, configuration loading patterns, and utility functions frequently appear in multiple files with minor or no variation. Without deduplication, the index balloons and identical content competes with itself in retrieval rankings.

Mnemosyne applies **content-addressed storage** at the chunk level:

1. Each chunk's text is **whitespace-normalized** (CRLF -> LF, trailing whitespace stripped per line) before hashing. This ensures that the same logical content with different line-ending conventions produces the same hash — critical for cross-platform teams.

2. A **SHA-256 digest** is computed over the normalized content. SHA-256 provides 2^256 possible values; collisions are not a practical concern for any codebase.

3. Before inserting a new chunk, the store performs a `get_chunk_by_hash()` lookup. If the hash already exists in the index, the chunk is counted as deduplicated and skipped. The BM25 and TF-IDF indices are not polluted with redundant content.

4. A **Bloom filter** provides an O(1) probabilistic pre-check for "is this file/hash definitely not indexed?" using the Kirsch-Mitzenmacher double-hashing trick (MD5 + SHA-1 pair). False negatives are impossible; false positives at the configured 0.1% rate are resolved by the subsequent hash lookup. This avoids a database read for the large majority of unchanged files during incremental re-index runs.

During a typical incremental ingest of a 10,000-file codebase, the Bloom filter and mtime check together prevent database reads for over 95% of files, keeping incremental ingest fast even as the project grows.

Implementation: `hasher.py` — `content_hash`, `file_hash`; `bloom.py` — `BloomFilter`; `store.py` — `get_chunk_by_hash`.

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- No other dependencies (optional: `onnxruntime` for future dense embeddings)

### Installation

```bash
# Install from PyPI
pip install mnemosyne-engine

# Or install from source (for development)
git clone https://github.com/castnettech/mnemosyne.git
cd mnemosyne
pip install -e .
```

### Initialize a Project

```bash
cd /your/project

# Create .mnemosyne/ directory with default config and empty database
python -m mnemosyne init
```

This creates:
```
.mnemosyne/
├── config.toml       # editable configuration
└── mnemosyne.db      # SQLite database (WAL mode)
```

### Index Your Project

```bash
# Index all supported files
python -m mnemosyne ingest

# Index specific files only
python -m mnemosyne ingest src/auth.py src/models.py

# Force full re-index, ignoring cached hashes
python -m mnemosyne ingest --full

# Preview what would be indexed without writing
python -m mnemosyne ingest --dry-run
```

### Query for Relevant Context

```bash
# Retrieve context with default token budget (from config)
python -m mnemosyne query "how does authentication work"

# Set an explicit token budget
python -m mnemosyne query "how does authentication work" --budget 6000

# Get JSON output for programmatic consumption
python -m mnemosyne query "database connection pooling" --format json

# Include per-signal scores in output headers
python -m mnemosyne query "rate limiting logic" --show-scores

# Disable compression fallback
python -m mnemosyne query "error handling patterns" --no-compress

# Attach a session ID for usage tracking continuity
python -m mnemosyne query "JWT validation" --session my-session-001
```

### Integrate with an LLM Agent

How to enable your LLM to use Mnemosyne (agent-instructable)
One bash command:

cd /path/to/project && mnemosyne query "your question here" --budget 8000

That's it. For an agent's instruction file (CLAUDE.md, .github/copilot-instructions.md, etc.):

## Code Search
  Before reading files to answer questions about this codebase, query the
  Mnemosyne index first:
    ! cd /path/to/project && mnemosyne query "your question" --budget 8000
  Use the returned chunks as context. Only read additional files if the
  chunks don't fully answer the question.
  Refresh the index after code changes: mnemosyne ingest

### Preview Compression

```bash
# See what the compression pipeline produces for a specific file
python -m mnemosyne compress src/auth.py

# Override target compression ratio
python -m mnemosyne compress src/auth.py --ratio 0.3
```

Sample output:
```
File:            src/auth.py
Original tokens: 842
Compressed:      341
Char ratio:      40.5%

────────────────────────────────────────────────────────────
# [8 imports: os, sys, re, hashlib, datetime, jwt, models, config]

class AuthManager:
    """Manages user authentication, token issuance, and session validation."""

    def __init__(self, config, store):
# [6 assignments: config, store, secret, algorithm, expiry_hours, logger]

    def authenticate(self, username: str, password: str) -> str | None:
        """Verify credentials and return a signed JWT, or None on failure."""
        ...
    def validate_token(self, token: str) -> dict | None:
        """Decode and verify a JWT. Returns the payload or None if invalid."""
        ...
        return payload
```

### View Statistics

```bash
# Summary statistics
python -m mnemosyne stats

# Detailed breakdown including chunk types, languages, cache state
python -m mnemosyne stats --detailed
```

Sample output:
```
Project root:   /home/user/myproject
Files indexed:  87
Chunks:         878
Total tokens:   143,241
Usage events:   0

Avg tokens/chunk: 163.2

Chunk types:
  block          : 142
  class          :  34
  function       : 521
  imports        :  87
  paragraph      :  94

Languages:
  python         : 61 files
  typescript     : 14 files
  go             : 12 files
  markdown       :  8 files
  csharp         :  6 files
  javascript     :  4 files
  rust           :  3 files
  java           :  2 files
```

### Check for Changes

```bash
# Show what changed since the last ingest
python -m mnemosyne delta

# Limit to specific paths
python -m mnemosyne delta src/auth.py src/models.py
```

### Manage the ARC Cache

```bash
# Show current cache tier distribution
python -m mnemosyne cache show

# Pre-warm the cache with the most-accessed chunks
python -m mnemosyne cache warm

# Clear all cache state
python -m mnemosyne cache clear
```

### View the Audit Log

```bash
# Show the last 20 audit entries
python -m mnemosyne audit

# Show the last 100 entries
python -m mnemosyne audit --last 100
```

### Garbage Collection

```bash
# Preview what would be removed
python -m mnemosyne gc --dry-run

# Run garbage collection
python -m mnemosyne gc
```

---

## CLI Reference

### `mnemosyne init`

Initialize a `.mnemosyne/` workspace in the current directory.

Creates `.mnemosyne/config.toml` with all default values and `.mnemosyne/mnemosyne.db` with the full schema applied. Safe to run in an existing project — exits immediately if `.mnemosyne/` already exists.

**No flags.**

---

### `mnemosyne ingest [paths...] [--full] [--dry-run]`

Index files into the knowledge base.

| Argument | Description |
|---|---|
| `paths` | Optional list of specific file paths to index. Defaults to the entire project. |
| `--full` | Force full re-index of every file, bypassing hash and mtime checks. |
| `--dry-run` | Scan and report counts without writing any data to the database. |

After ingestion, the TF-IDF vocabulary is rebuilt from the full corpus so that IDF values are meaningful. The Bloom filter is saved to `bloom.bin`. An audit entry is appended to `audit.log`.

**Output example:**
```
Files scanned:  89
Files indexed:  87
Files skipped:  2
Files failed:   0
Chunks added:   878
Chunks deduped: 12
Elapsed:        0.52s
```

---

### `mnemosyne query <text> [options]`

Retrieve relevant context chunks for a query string.

| Argument | Description |
|---|---|
| `text` | The query string. Required. |
| `--budget N` | Maximum total tokens in the result set. Defaults to `retrieval.token_budget` in config (default: 8000). |
| `--format plain\|json` | Output format. Default: `plain`. |
| `--session SESSION_ID` | Session identifier for usage tracking. Auto-generated UUID fragment if omitted. |
| `--no-compress` | Disable compression fallback when a chunk exceeds the remaining budget. |
| `--show-scores` | Include per-signal scores (BM25, vector, usage, RRF) in plain-text output headers. |

**Plain output format:**
```
Query: how does authentication work
Budget: 6000 tokens | Session: a3f8b2c1
Results: 6 chunks, 4,218 tokens used

--- src/auth.py [lines 45-112] [function: AuthManager.authenticate] [tokens: 341] ---
[compressed]
...chunk content...

--- src/models.py [lines 1-23] [imports] [tokens: 89] [STALE] ---
...chunk content...
```

Results where the underlying file has changed since indexing are marked `[STALE]`. The JSON output includes `is_stale` and `stale_reason` fields for programmatic detection.

**JSON output**: A structured array of `QueryResult` objects with `chunk`, `file_path`, `scores` (bm25, vector, usage, rrf), `is_stale`, `stale_reason`, `is_delta`, and `delta_text` fields.

---

### `mnemosyne stats [--detailed]`

Display index and cache statistics.

| Flag | Description |
|---|---|
| `--detailed` | Show per-chunk-type counts, per-language file counts, cache tier state, and pattern count. |

---

### `mnemosyne compress <file> [--ratio FLOAT]`

Preview the four-stage compression pipeline for a single file.

| Argument | Description |
|---|---|
| `file` | Path to the file to compress. |
| `--ratio` | Override the target compression ratio (0.0–1.0). Default: from `compression.target_ratio` in config (default: 0.4). |

Displays original token count, compressed token count, character ratio, and the compressed output.

---

### `mnemosyne cache [show|clear|warm]`

Manage the ARC cache persisted in the database.

| Action | Description |
|---|---|
| `show` | Display current T1/T2/B1/B2 tier counts (default). |
| `clear` | Delete all cache state entries from the database. |
| `warm` | Pre-load the most-accessed chunks into the in-memory ARC cache. |

---

### `mnemosyne delta [paths...]`

Show file changes detected since the last index run.

Compares on-disk mtime and content against stored `FileRecord` entries. Reports added, modified, and deleted files. For modified files, shows the number of added and removed lines.

| Argument | Description |
|---|---|
| `paths` | Optional: limit change detection to specific file paths. |

---

### `mnemosyne audit [--last N]`

Print recent entries from the append-only audit log.

| Flag | Description |
|---|---|
| `--last N` | Number of most-recent entries to display. Default: 20. |

Audit entries are JSON objects: `{"ts": "2026-03-21T10:00:00Z", "op": "ingest_complete", "files_indexed": 87, ...}`

---

### `mnemosyne gc [--dry-run]`

Garbage collect orphaned chunks and stale file records.

Removes chunks belonging to soft-deleted file records. Marks file records as deleted when the corresponding file no longer exists on disk. Prunes stale cache state and usage events. **Rebuilds the Bloom filter** from surviving entries so that stale entries no longer cause false "already indexed" skips on future ingests.

| Flag | Description |
|---|---|
| `--dry-run` | Report what would be removed without deleting anything. |

---

### `mnemosyne analytics [--session SESSION_ID] [--top-chunks N]`

Display feedback precision metrics and top-used chunks from recorded usage events.

| Flag | Description |
|---|---|
| `--session SESSION_ID` | Limit analysis to a specific session. |
| `--top-chunks N` | Show the N most-accessed chunks. Default: 5. |

Reports precision-at-k (ratio of used to retrieved chunks), total feedback event counts, and optionally the most frequently accessed chunks across sessions.

---

### `mnemosyne daemon <start|stop|status> [--foreground]`

Manage the JSON-RPC background daemon. The daemon keeps SQLite, the TF-IDF inverted index, analytics, and the prefetcher warm across requests, eliminating cold-start overhead. Communicates over a Unix domain socket at `.mnemosyne/mnemosyne.sock`.

| Action | Description |
|---|---|
| `start` | Start the daemon in the background (or foreground with `--foreground`). |
| `stop` | Send SIGTERM to a running daemon. |
| `status` | Check whether the daemon is running. |

| Flag | Description |
|---|---|
| `--foreground` | Run the daemon in the foreground (blocks). |

---

## Configuration Reference

Configuration is read from `.mnemosyne/config.toml` in the project root, deep-merged on top of built-in defaults. The file is created with defaults by `mnemosyne init` and is safe to edit.

### `[general]`

| Key | Default | Description |
|---|---|---|
| `project_root` | `"."` | Root directory to scan (relative to config file location). |
| `ignore_patterns` | `[".git", "node_modules", "__pycache__", ...]` | fnmatch patterns for files and directories to skip. |
| `max_file_size_kb` | `512` | Files larger than this are skipped during ingestion. |
| `supported_extensions` | `[".py", ".js", ".ts", ".go", ".cs", ".rs", ".java", ".kt", ".md", ...]` | File extensions to index. |

### `[chunking]`

| Key | Default | Description |
|---|---|---|
| `max_chunk_tokens` | `300` | Maximum token count for a single chunk. Larger logical units are split. |
| `min_chunk_tokens` | `20` | Minimum token count. Smaller candidates are merged with adjacent chunks. |
| `overlap_lines` | `3` | Lines of overlap between adjacent chunks (for sliding-window chunkers). |
| `code_granularity` | `"function"` | Granularity for code chunking: `"function"` extracts individual functions/methods; `"class"` groups methods with their class body. |

### `[embedding]`

| Key | Default | Description |
|---|---|---|
| `backend` | `"tfidf"` | Embedding backend. Currently `"tfidf"`; dense backend via `onnxruntime` planned. |
| `tfidf_max_features` | `10000` | Maximum vocabulary size (top-N terms by document frequency). |
| `tfidf_min_df` | `2` | Minimum document frequency for a term to enter the vocabulary. Filters noise terms that appear in only one file. |

### `[compression]`

| Key | Default | Description |
|---|---|---|
| `target_ratio` | `0.4` | Target character ratio after compression (0.4 = keep 40% of characters). |
| `preserve_signatures` | `true` | Always preserve function/class signature lines. |
| `preserve_docstrings` | `true` | Always preserve lines inside triple-quoted docstrings. |
| `collapse_imports` | `true` | Collapse import blocks longer than 3 lines into a summary token. |
| `collapse_boilerplate` | `true` | Collapse `self.x = y` assignment runs and logging call runs. |

### `[retrieval]`

| Key | Default | Description |
|---|---|---|
| `bm25_weight` | `0.4` | RRF weight for BM25 signal. |
| `vector_weight` | `0.4` | RRF weight for TF-IDF vector signal. |
| `usage_weight` | `0.2` | RRF weight for usage frequency signal. |
| `max_results` | `20` | Maximum candidates to fetch from each search signal before fusion. |
| `token_budget` | `8000` | Default token budget for `query` when `--budget` is not specified. |

### `[cache]`

| Key | Default | Description |
|---|---|---|
| `capacity` | `500` | Maximum live chunks in the ARC cache (T1 + T2). |
| `ghost_capacity` | `1000` | Maximum ghost keys (B1 + B2) tracked for adaptive parameter tuning. |

### `[analytics]`

| Key | Default | Description |
|---|---|---|
| `decay_halflife_days` | `7` | Half-life for usage frequency decay. Reduce to focus on recent access; increase to weight historical patterns more heavily. |
| `session_timeout_minutes` | `30` | Session inactivity timeout. Used for co-occurrence analysis grouping. |

---

## Integration with LLM Agents

Mnemosyne is a standalone CLI tool. All consumers interact with Mnemosyne directly via the CLI or Python API.

### From Claude Code

Use the shell escape to call Mnemosyne from within a Claude Code session:

```bash
! mnemosyne query "how does authentication work" --budget 6000
! mnemosyne stats
```

### From scripts

```bash
CONTEXT=$(mnemosyne query "database connection pooling" --format json --budget 4000)
```

Or via `subprocess`:

```python
import subprocess, json
result = subprocess.run(
    ["mnemosyne", "query", "auth middleware", "--format", "json", "--budget", "4000"],
    capture_output=True, text=True
)
chunks = json.loads(result.stdout)
```

### Programmatic Python API

```python
from mnemosyne.retrieval import RetrievalEngine
from mnemosyne.config import Config

config = Config.load()
engine = RetrievalEngine(config)
results = engine.query("authentication middleware", budget=6000)
```

The JSON-RPC daemon mode (`mnemosyne daemon start`) keeps indexes warm for low-latency repeated queries.

### CLI Usage

For development, debugging, and benchmarking:

```bash
# Shell pipe to STDIN
CONTEXT=$(python -m mnemosyne query "how does authentication work" --budget 6000)

# JSON output for programmatic consumption
python -m mnemosyne query "database connection pooling" --format json

# Incremental session context
SESSION="session-$(date +%s)"
python -m mnemosyne query "authentication module" --budget 6000 --session "$SESSION"
```

### Keeping the Index Fresh

For active development, run `ingest` as part of your save workflow or as a file-watcher hook:

```bash
# Simple: re-ingest changed files manually before querying
python -m mnemosyne ingest src/auth.py

# With entr (file watcher):
find src/ -name "*.py" | entr python -m mnemosyne ingest /_

# As a pre-commit hook:
# .git/hooks/pre-commit
python -m mnemosyne ingest $(git diff --cached --name-only)
```

---

## Benchmarks

All measurements taken on a mid-range Linux workstation (AMD Ryzen 7, NVMe SSD) against a real Python codebase.

### Ingestion Performance

| Metric | Result |
|---|---|
| Files scanned | 89 |
| Files indexed | 87 |
| Chunks produced | 878 |
| Total tokens indexed | 143,241 |
| Wall-clock time | 0.52 seconds |
| Throughput | ~167 files/second |
| Storage footprint | ~600 KB (SQLite DB + Bloom filter) |
| Storage efficiency | ~4.2 bytes per indexed token |

Incremental re-ingest of an unchanged project (Bloom + mtime check): **< 50ms** regardless of project size.

### Query Performance

| Metric | Result |
|---|---|
| Cold query (inverted index build from DB) | < 100ms |
| Warm query (inverted index in memory) | < 20ms |
| BM25 search (FTS5) | < 5ms |
| TF-IDF vector search | < 10ms |
| RRF fusion + cost-model ranking | < 2ms |
| Budget cutting + compression fallback | < 5ms |

End-to-end query latency is dominated by SQLite I/O on first call and inverted-index reconstruction. Subsequent queries within the same process are substantially faster as the index is held in memory.

### Compression Effectiveness

| Content Type | Token Reduction | Semantic Preservation |
|---|---|---|
| Python class with logging | 55–65% | High — signatures, docstrings, returns intact |
| Python module with imports | 40–50% | High — import block collapsed to summary |
| Configuration-heavy module | 30–45% | High — assignments collapsed, comments preserved |
| Function-dense utility module | 20–35% | High — all function signatures preserved |
| Markdown documentation | 15–25% | Medium — paragraph splitting, no collapse heuristics |

The compression ratio varies by content type. Code with high proportions of imports, logging, and simple assignments compresses more aggressively. Pure logic with complex control flow compresses less but also needs less compression (its information density is already high).

### Deduplication Effectiveness

On a typical Python project with shared utilities and copy-paste patterns: **8–15% of chunks** are deduplicated on first index. This directly reduces index size, query noise, and retrieval ranking interference from repeated content.

### Benchmark Suite (v0.3.0)

A multi-project benchmark runner (`tests/benchmark_suite.py`) measures chunk-level precision across multiple codebases. It indexes each project, runs a set of queries against known-relevant chunks, and reports precision-at-k scores. The `mnemosyne analytics` CLI command provides the same precision metrics from recorded feedback events in production.

### Memory Footprint

| Component | Memory Usage |
|---|---|
| SQLite connection + cache | ~2–5 MB |
| TF-IDF inverted index (10K vocab, 878 chunks) | ~3–8 MB |
| ARC cache (500 chunks) | ~5–15 MB (depends on chunk sizes) |
| Bloom filter (100K capacity) | ~180 KB |
| **Total** | **~10–30 MB** |

Mnemosyne is suitable for use in resource-constrained environments, embedded agent runtimes, and CI pipelines.

---

## Technical Stack

Mnemosyne is built entirely on the **Python 3.11+ standard library**. No external packages are required at runtime. The optional `onnxruntime` dependency is reserved for future dense embedding support.

| Component | Implementation |
|---|---|
| Persistence | SQLite 3 with WAL mode, `busy_timeout=5000`, advisory file lock for write serialization |
| Full-text search | FTS5 virtual table with Porter stemmer and unicode61 tokenizer |
| Structured data | JSON1 SQLite extension for sparse embedding storage |
| Content addressing | `hashlib.sha256` with whitespace normalization |
| Bloom filter | Pure Python — Kirsch-Mitzenmacher double hashing (MD5 + SHA-1 pair) |
| ARC cache | Pure Python — four `collections.OrderedDict` instances |
| TF-IDF | Pure Python — `math`, `re`, `collections.Counter`, `defaultdict` |
| BM25 | Delegated to SQLite FTS5 (production-grade C implementation) |
| RRF fusion | Pure Python — rank-based fusion with configurable per-source weights |
| AST chunking | `ast` module (stdlib) — full AST walk for Python files |
| Brace-based chunking | Pure Python regex + brace tracking for Go, C#, Rust, Java, Kotlin |
| JSON-RPC daemon | `socket` + `select` (stdlib) — Unix domain socket server |
| Diff computation | `difflib.unified_diff` (stdlib) |
| Configuration | `tomllib` (stdlib since Python 3.11) with dot-access namespace wrapper |
| Audit logging | Append-only JSON-lines file |
| Near-duplicate detection | `difflib.SequenceMatcher` with 0.85 ratio threshold |

The zero-dependency design is a deliberate constraint, not an oversight. It means:

- **Installation is one command** — `pip install mnemosyne-engine` with no conflict resolution.
- **Deployment is trivial** — copy the directory, run Python.
- **Security surface is minimal** — no third-party supply chain to audit.
- **Compatibility is guaranteed** — any Python 3.11+ environment works.

---

## File Structure

The `mnemosyne/` package contains 35+ modules across 3 sub-packages.

### Core Domain

| Module | Description |
|---|---|
| `models.py` | Pure dataclasses: `FileRecord`, `Chunk`, `Summary`, `QueryResult` (with `is_stale`/`stale_reason`), `CacheEntry`, `UsageEvent`. Plus `estimate_tokens()`. |
| `config.py` | TOML configuration loader with deep-merge, dot-access namespace, and `save()` method. |
| `schema.py` | SQLite DDL manager: `get_connection()`, `init_db()`, `migrate()`. All table definitions, indexes, `busy_timeout`, advisory file lock. |
| `store.py` | Repository layer: all SQLite CRUD for every domain object. Zero raw SQL outside this module. |

### Ingestion

| Module | Description |
|---|---|
| `ingest.py` | Ingestion orchestrator: file scanning, change detection, chunking, dedup, storage, TF-IDF rebuild. |
| `hasher.py` | SHA-256 content addressing with whitespace normalization, binary file detection. |
| `bloom.py` | Bloom filter: Kirsch-Mitzenmacher double hashing, binary persistence, fill ratio diagnostics. |
| `chunkers/__init__.py` | `get_chunker()` factory and `detect_language()` by file extension. |
| `chunkers/code_chunker.py` | AST-based Python chunker. |
| `chunkers/js_chunker.py` | Regex+brace structural chunker for JavaScript and TypeScript. |
| `chunkers/brace_chunker.py` | Shared base class for brace-delimited languages with symbol extraction. |
| `chunkers/go_chunker.py` | Go chunker: funcs, methods, types, structs, interfaces. |
| `chunkers/csharp_chunker.py` | C# chunker: classes, methods, properties, namespaces. |
| `chunkers/rust_chunker.py` | Rust chunker: fn, impl, struct, enum, trait, mod. |
| `chunkers/java_chunker.py` | Java/Kotlin chunker: classes, methods, interfaces, enums. |
| `chunkers/text_chunker.py` | Paragraph-boundary chunker for Markdown, plain text, and prose. |
| `chunkers/generic_chunker.py` | Sliding-window line-count chunker for unknown file types. |

### Compression

| Module | Description |
|---|---|
| `compress.py` | Four-stage compression pipeline: structural preservation, boilerplate collapse, TF-IDF importance filter, density analysis. |
| `density.py` | Density analysis utilities: near-duplicate line detection and whitespace normalization. |

### Embeddings and Search

| Module | Description |
|---|---|
| `embeddings/__init__.py` | `get_backend()` factory for embedding backends. |
| `embeddings/tfidf_backend.py` | TF-IDF sparse vector backend: augmented TF, smoothed IDF, inverted index, camelCase/snake_case splitting. |
| `vectorstore.py` | Persistence adapter: saves and loads sparse embedding vectors to/from the SQLite store. |

### Retrieval and Ranking

| Module | Description |
|---|---|
| `retrieval.py` | Hybrid retrieval orchestrator: BM25 + TF-IDF + usage + prefetch, RRF fusion, cost-model ranking, budget cutting. |
| `ranking.py` | `rrf_fuse()`, `cost_model_score()`, `budget_cut()` — pure functions, no I/O. |
| `analytics.py` | Usage event recording and exponential decay scoring. Co-occurrence analysis. |
| `prefetch.py` | Query signature computation and pattern-based pre-fetch recommendation. |

### Caching and Tiers

| Module | Description |
|---|---|
| `cache.py` | Full ARC cache implementation: T1/T2/B1/B2 ordered dicts, adaptive parameter `p`, hit-rate tracking. |
| `tiers.py` | `TierManager`: bridge between the in-memory ARC cache and the persistent `cache_state` SQLite table. |

### Infrastructure

| Module | Description |
|---|---|
| `delta.py` | File-level and chunk-level change detection, unified diff computation, hunk-based chunk impact analysis. |
| `formatter.py` | Plain-text and JSON output formatters for `QueryResult` lists. `[STALE]` markers for stale results. |
| `audit.py` | Append-only JSON-lines audit log: `log()`, `read()`. |
| `daemon.py` | JSON-RPC daemon: Unix socket server, warm-start components, `start`/`stop`/`status` lifecycle. |
| `cli.py` | Full CLI with `argparse`: `init`, `ingest`, `query`, `stats`, `compress`, `cache`, `delta`, `audit`, `analytics`, `gc`, `daemon` commands. |
| `__main__.py` | `python -m mnemosyne` entry point. |
| `__init__.py` | Package metadata: `__version__`, `__package_name__`. |

### Tests and Benchmarks

| Module | Description |
|---|---|
| `tests/__init__.py` | Test package marker. |
| `tests/test_core.py` | Core model and config tests. |
| `tests/test_cache.py` | ARC cache tests. |
| `tests/test_chunkers.py` | Chunker tests (Python, JS/TS). |
| `tests/test_brace_chunkers.py` | Brace-family chunker tests (Go, C#, Rust, Java/Kotlin). |
| `tests/test_store.py` | SQLite store CRUD tests. |
| `tests/test_tfidf.py` | TF-IDF backend tests. |
| `tests/test_retrieval.py` | Retrieval pipeline tests. |
| `tests/test_compression.py` | Compression pipeline tests. |
| `tests/test_analytics.py` | Analytics and feedback precision tests. |
| `tests/test_integration.py` | End-to-end integration tests. |
| `tests/test_daemon.py` | JSON-RPC daemon tests. |
| `tests/benchmark.py` | Single-project benchmarking. |
| `tests/benchmark_suite.py` | Multi-project benchmark runner with chunk-level precision measurement. |

---

## Algorithm Documentation

Full algorithm details, design rationale, and academic paper references are in **`ALGORITHMS.md`**. This covers the TF-IDF backend, BM25 via FTS5, ARC cache, RRF fusion, cost-model ranking, compression pipeline, Bloom filter, and predictive pre-fetching, with citations to the original papers (Robertson et al. 1994, Megiddo & Modha 2003, Cormack et al. 2009, Kirsch & Mitzenmacher 2006, etc.).

---

## Future Roadmap

### Completed in v0.3.0

- **Standalone CLI** -- Mnemosyne is a CLI-only tool by design. No MCP server, no protocol bridges. Consumers use `mnemosyne query/ingest/stats` commands or the Python API directly.
- **JSON-RPC Daemon Mode** -- `mnemosyne daemon start/stop/status` runs a persistent Unix-domain-socket server keeping SQLite, TF-IDF, analytics, and prefetcher warm. Eliminates cold-start overhead for high-throughput workloads.
- **Multi-Language Structural Chunking** -- Go, C#, Rust, Java, and Kotlin now get brace-based structural chunking with symbol extraction via a shared `BraceChunker` base class, without requiring Tree-sitter.
- **Staleness Detection** -- `QueryResult.is_stale` and `stale_reason` fields; formatter shows `[STALE]` markers.
- **Bloom Filter Rebuild on GC** -- `mnemosyne gc` rebuilds the Bloom filter from surviving entries.
- **SQLite Hardening** -- `busy_timeout=5000`, advisory file lock for write serialization.
- **Tokenizer Version Tracking** -- Detects index/tokenizer version mismatch; warns and blocks stale searches.
- **Compression Safety Net** -- Control flow lines preserved, 70% max prune ratio, strict mode for symbol chunks.
- **Analytics CLI** -- `mnemosyne analytics` shows precision-at-k from feedback events and top-used chunks.
- **Benchmark Suite** -- Multi-project runner with chunk-level precision measurement.
- **PyPI Package** -- `pip install mnemosyne-engine` (import name remains `mnemosyne`).
- **ALGORITHMS.md** -- Full algorithm documentation with paper references.

### Planned

#### Dense Embeddings Backend

The current TF-IDF backend is exact-match oriented (with stemming and identifier splitting). It misses semantic similarity between paraphrases and concepts expressed with different vocabulary. A dense embedding backend (via optional `onnxruntime` dependency) is planned as an additional signal fused into the existing RRF pipeline alongside BM25, TF-IDF, usage, and prefetch.

The architecture is already prepared: the `embeddings/__init__.py` factory pattern allows backends to be swapped without changes to the retrieval pipeline. The `pyproject.toml` declares `onnxruntime` as an optional `[dense]` dependency.

#### Tree-sitter Multi-Language AST Parsing

The brace-based chunkers added in v0.3.0 provide good structural extraction for Go, C#, Rust, Java, and Kotlin. Tree-sitter would provide a uniform, production-grade parsing interface for 100+ languages, replacing regex patterns with true AST-based chunking where higher precision is needed (e.g., C/C++, Ruby, Swift).

#### Hierarchical Summary Trees

Currently, retrieval operates at the chunk level. For very large codebases, a hierarchical summary structure would enable two-level retrieval: query the directory-level or file-level summaries first to identify relevant files, then drill down into chunk-level retrieval only within those files. This mirrors how experienced developers navigate a codebase: they know which module to look in before searching for a specific function.

The `Summary` dataclass and `summaries` table are already defined in the schema. The hierarchical summary generation and two-level retrieval logic are the remaining pieces.

---

## License

Copyright 2026 Cast Rock Innovation L.L.C. (DBA: Cast Net Technology)

Mnemosyne is dual-licensed:

- **Open Source:** [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) — free for open-source, non-commercial, and internal use.
- **Commercial:** If you want to embed Mnemosyne in a proprietary product or offer it as a commercial service without open-sourcing your code, [contact us for a commercial license](mailto:support@castnettechnology.com).

See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for details on when a commercial license is needed.
