# Mnemosyne Algorithm Reference

Technical documentation of every custom algorithm in the Mnemosyne retrieval system.
Formulas and parameters are extracted directly from source code.

---

## 1. BM25 via FTS5

**Source:** `retrieval.py` (`_bm25_search`, `_escape_fts5`)

Mnemosyne delegates lexical scoring to SQLite FTS5's built-in BM25 implementation,
configured with the Porter stemmer and Unicode61 tokenizer.

**Query preprocessing:**
1. Escape FTS5 special characters: `" ' ( ) * : ^`
2. Filter stopwords (60 common English words) and tokens shorter than 2 characters.
3. Join remaining terms with `OR` for broad recall (FTS5 default AND is too strict for code search).

**Score normalization:**
FTS5 `bm25()` returns negative values (lower = better). Raw scores are converted to
positive magnitudes and normalized to `[0, 1]` relative to the top score:

```
normalized_score = abs(raw_score) / max(abs(raw_score) for all results)
```

**Result limit:** `config.retrieval.max_results * 3` (default: 60 candidates).

**Config parameters:**
- `retrieval.bm25_weight` = `0.4` (RRF fusion weight)
- `retrieval.max_results` = `20`

**Reference:** Robertson, S.E. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

---

## 2. TF-IDF (Augmented Term Frequency)

**Source:** `embeddings/tfidf_backend.py` (`TFIDFBackend`)

Pure-Python sparse vector backend with no external dependencies.

### Tokenization

1. Extract tokens matching `[a-zA-Z_][a-zA-Z0-9_]{2,}` (minimum 3 characters).
2. Split camelCase: `getUserById` -> `get`, `user`, `by`, `id` (via `re.sub(r"([a-z])([A-Z])", r"\1_\2", tok)`).
3. Split snake_case: handled by underscore splitting after camelCase insertion.
4. Filter stopwords (60-word set, synchronized with `retrieval._STOPWORDS`).
5. Minimum sub-token length: 2 characters (`_MIN_SUBTOKEN_LEN`).
6. Both the original lowercased token and its sub-parts are retained, so exact-match queries still score highly.

### Augmented TF

Prevents bias toward long documents while preserving relative frequency signal:

```
tf(t, d) = 0.5 + 0.5 * (count(t, d) / max_count(d))
```

Where `max_count(d)` is the frequency of the most common term in document `d`.

### Smoothed IDF

Avoids division-by-zero and gives non-zero weight to terms appearing in every document:

```
idf(t) = log((N + 1) / (df(t) + 1)) + 1.0
```

Where `N` = total documents, `df(t)` = number of documents containing term `t`.

### Final weight

```
w(t, d) = tf(t, d) * idf(t)
```

### Vocabulary building

1. Count document frequency (each term counted once per document).
2. Filter: terms must appear in at least `min_df` documents.
3. Cap at `max_features` terms, keeping the rarest terms (lowest df = highest IDF = most discriminative).
4. Version tracking: tokenizer config is hashed (SHA-256 of sorted stopwords + regex + min sub-token length). Stale vocabulary triggers a rebuild warning.

### Search via inverted index

The inverted index maps each vocabulary term to `list[(chunk_id, weight)]`. At query time,
only terms present in the query vector are consulted (sub-linear search). Scores are
accumulated partial dot products:

```
score(d) = sum(q_weight[t] * d_weight[t] for t in query_terms intersect doc_terms)
```

Query vector norm is constant across all candidates so it is omitted from ranking comparison.

### Sparse vector storage

Vectors are stored as JSON `{term: weight}` dicts. Cosine similarity uses standard
dot-product / (norm_a * norm_b) over the common-term intersection.

**Config parameters:**
- `embedding.backend` = `"tfidf"` (default; disables dense vectors entirely)
- `embedding.tfidf_max_features` = `10000`
- `embedding.tfidf_min_df` = `1`
- `retrieval.vector_weight` = `0.4` (RRF fusion weight)

**Reference:** Salton, G. & Buckley, C. (1988). "Term-Weighting Approaches in Automatic Text Retrieval." *Information Processing & Management*, 24(5), 513-523.

---

## 3. Reciprocal Rank Fusion (RRF)

**Source:** `ranking.py` (`rrf_fuse`)

Merges multiple ranked signal lists into a single fused ranking without requiring score calibration across sources.

### Formula

```
rrf_score(d) = sum( weight_s / (k + rank_s(d)) )  for each source s
```

Where:
- `k` = 60 (smoothing constant)
- `rank_s(d)` = 1-based rank of document `d` in source `s`
- If `d` is absent from source `s`, penalty rank = `len(source_s) + 1`
- `weight_s` = per-source weighting factor

### Default source weights

| Source    | Weight | Description                    |
|-----------|--------|--------------------------------|
| `bm25`    | 0.4    | Lexical BM25 via FTS5          |
| `vector`  | 0.4    | TF-IDF sparse vector search    |
| `usage`   | 0.2    | Decay-weighted usage frequency  |

### Dynamic sources (added when available)

| Source     | Weight             | Condition                   |
|------------|--------------------|-----------------------------|
| `symbol`   | `bm25_weight * 1.5` (= 0.6) | Symbol name matches found |
| `prefetch` | `max(bm25_weight, vector_weight)` (= 0.4) | Pre-fetch pattern hit |

**Config parameters:**
- `retrieval.bm25_weight` = `0.4`
- `retrieval.vector_weight` = `0.4`
- `retrieval.usage_weight` = `0.2`

**Reference:** Cormack, G.V., Clarke, C.L.A. & Butt, S. (2009). "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *Proceedings of SIGIR 2009*, 758-759.

---

## 4. Post-Fusion Boosts

**Source:** `retrieval.py` (`_symbol_search`, `_filename_boost`, `_import_graph_boost`, `_file_level_filter`)

Applied sequentially after RRF fusion to inject structural signals that pure term matching misses.

### 4a. Symbol Name Multiplier

Chunks whose `symbol_name` matches a query term receive a **3x RRF score multiplier**.

- Query terms are extracted as identifiers matching `[a-zA-Z_][a-zA-Z0-9_]{2,}`.
- camelCase is split into sub-parts for partial matching.
- Exact match: score 1.0. Partial/component overlap: score = `overlap_count / max(symbol_parts, search_terms) * 0.8`.
- Only chunks with score >= 0.5 qualify for the 3x boost.

### 4b. Filename Boost

All chunks from files whose basename matches query terms receive a **1.5x RRF score multiplier**.

Matching rules:
- Query terms must be at least 5 characters and not stopwords.
- Filename components are split on `[-_.]` separators after removing the extension.
- Stem-prefix matching: a query term and filename component must share a common prefix of >= **5 characters**.
- The prefix must cover at least **60%** of the shorter string's length.
- Only files already present in fused results are boosted (no injection of new files).

### 4c. Import Graph Injection

Injects files connected to already-retrieved files via dependency edges. Runs **after**
the file-level filter so keyword-matched results are never displaced.

Three reference types scanned:
1. ES6 imports / CommonJS `require` (static module references).
2. Runtime namespace access (`Namespace.Module` patterns mapped to indexed basenames).
3. Path references (quoted strings matching indexed file paths).

Injection rules:
- Minimum **2** references required (`config.retrieval.import_inject_threshold`).
- Maximum **2** files injected per query (dynamic cap: `min(2, max(0, 10 - current_file_count))`).
- Source files prioritized over test files.
- High-confidence injections (3+ references): score = `top_score * 0.9`.
- Standard injections (2 references): score = `p75_score * 0.7`.

### 4d. Adaptive File-Level Filter

Retains only chunks from the top-scoring files by aggregate RRF score.

```
max_files = min(max(3, n_unique_files // 3), 8)
```

Scales from 3 files (small projects) to 8 files (large corpora). Overridable via
`config.retrieval.max_files` (positive value = hard cap; 0 = adaptive).

**Config parameters:**
- `retrieval.max_files` = `0` (adaptive)
- `retrieval.import_inject_threshold` = `2`

---

## 5. Cost-Model Value Density

**Source:** `ranking.py` (`cost_model_score`), `retrieval.py` (`_cost_model_rank`)

Re-ranks fused results by information value per token, demoting bulky low-value chunks.

### Formula

```
relevance = rrf_score * (1.0 - 0.5 * boilerplate_ratio)
if is_structured_code:
    relevance *= 2.0
density = relevance / (1.0 + log1p(max(1, token_count)))
```

Where:
- `boilerplate_ratio` = fraction of lines detected as boilerplate + `repetition_score * 0.3`, capped at 1.0
- `is_structured_code` = True when `symbol_name is not None` or `chunk_type in ("function", "class", "imports")`
- `log1p(x)` = `log(1 + x)` (logarithmic dampening so chunk size influences ranking gently)

### Boilerplate penalties (path-based)

| File type / path pattern | Minimum `boilerplate_ratio` |
|--------------------------|----------------------------|
| `.html`, `.css`, `.md`, `.txt` (non-code chunks) | 0.85 |
| `tests/` or `test/` prefix | 0.50 |

### Structured code boost

Named symbol chunks (functions, classes) receive a **2x** relevance multiplier, since
they carry concrete code semantics that generic/block chunks lack.

---

## 6. Budget Cut

**Source:** `ranking.py` (`budget_cut`)

Greedy token-budget selection. Chunks are consumed in pre-ranked value-density order.

1. Iterate candidates in order from `_cost_model_rank` (already sorted by density).
2. If `tokens <= remaining_budget`: include chunk, subtract tokens.
3. If chunk exceeds remaining budget and a compressor is available:
   - Named symbol chunks use `strict=True` compression (skip TF-IDF thinning).
   - If compressed version fits: include with `compression_ratio = len(compressed) / len(original)`.
   - If compressed version still exceeds budget: skip chunk.
4. Stop when budget is exhausted.

**Config parameter:**
- `retrieval.token_budget` = `8000`

---

## 7. ARC Cache (Adaptive Replacement Cache)

**Source:** `cache.py` (`ARCCache`)

Self-tuning cache that balances recency and frequency without manual parameter tuning.

### Four-tier structure

| Tier | Contents | Purpose |
|------|----------|---------|
| **T1** | Recently-inserted items (seen once) | Recency signal |
| **T2** | Frequently-accessed items (seen 2+ times) | Frequency signal |
| **B1** | Ghost keys evicted from T1 (no data) | Recency history |
| **B2** | Ghost keys evicted from T2 (no data) | Frequency history |

### Adaptive parameter p

`p` = target size of T1 (how much of total capacity is allocated to recency).

**B1 ghost hit** (recency was underrepresented):
```
delta = max(1, len(B2) // max(1, len(B1)))
p = min(capacity, p + delta)
```

**B2 ghost hit** (frequency was underrepresented):
```
delta = max(1, len(B1) // max(1, len(B2)))
p = max(0, p - delta)
```

### Eviction rule

When `len(T1) + len(T2) >= capacity`:
- If `len(T1) > p` (or T2 is empty): evict LRU of T1, promote key to B1.
- Else: evict LRU of T2, promote key to B2.

### Hit behavior

- T1 hit: move item to MRU end of T2 (first re-access promotes to frequent).
- T2 hit: move item to MRU end of T2 (refresh LRU clock).

### Ghost list trimming

Combined B1 + B2 capped at `ghost_capacity`. When exceeded, the oldest entry from the
larger ghost list is evicted first.

**Config parameters:**
- `cache.capacity` = `500` (max live entries in T1 + T2)
- `cache.ghost_capacity` = `1000` (max ghost entries in B1 + B2)

**Reference:** Megiddo, N. & Modha, D.S. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache." *Proceedings of USENIX FAST 2003*, 115-130.

---

## 8. Bloom Filter

**Source:** `bloom.py` (`BloomFilter`, `_double_hash`, `_optimal_size`, `_optimal_hash_count`)

Probabilistic set membership using the Kirsch-Mitzenmacher double hashing construction.

### Double hashing

Two independent hash functions simulate `k` hash functions:

```
h1(x) = first 8 bytes of MD5(x) as uint64 LE
h2(x) = first 8 bytes of SHA-1(x) as uint64 LE
h_i(x) = (h1(x) + i * h2(x)) mod m    for i in 0..k-1
```

### Optimal sizing

Bit-array size for target false-positive rate:

```
m = ceil(-n * ln(p) / (ln 2)^2)
```

Optimal hash function count:

```
k = round((m / n) * ln 2)
```

Where `n` = expected capacity, `p` = target false-positive rate.

### Storage

Bit array stored as `bytearray` (1 byte per 8 bits). Binary persistence format:
- 32-byte header: capacity (uint64 LE), fp_rate (float64 LE), size_bits (uint64 LE), hash_count (uint64 LE).
- Followed by raw bit array bytes.

### Default parameters

- `capacity` = `100,000`
- `fp_rate` = `0.001` (0.1%)

**Memory example:** 100K capacity at 0.1% FP rate -> `m = -100000 * ln(0.001) / ln(2)^2` = ~1,437,759 bits = ~175 KB.

**Reference:** Kirsch, A. & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter." *Proceedings of ESA 2006*, LNCS 4168, 456-467.

---

## 9. Four-Stage Compression Pipeline

**Source:** `compress.py` (`Compressor`)

Reduces chunk token count for LLM context injection while preserving structural information.

### Stage 1: Structural Preservation

Identifies lines that must never be removed. Detected via regex patterns:

| Pattern | Regex |
|---------|-------|
| Function/class signatures | `^\s*(async\s+)?def\s+\w+` or `^\s*class\s+\w+` |
| Return/raise/yield | `^\s*(return\|raise\|yield)\b` |
| Assert | `^\s*assert\b` |
| Control flow | `^\s*(if\|elif\|else\s*:\|for\|while\|try\s*:\|except\|finally\s*:\|with\|switch\|case)\b` |
| Docstrings | Lines inside `"""` or `'''` blocks (when `preserve_docstrings=True`) |
| Annotated comments | `#\s*(TODO\|FIXME\|HACK\|NOTE\|XXX)\b` |

### Stage 2: Boilerplate Collapse

Replaces repetitive low-value patterns with compact inline summaries:

| Pattern | Threshold | Summary format |
|---------|-----------|---------------|
| Import blocks | > 3 consecutive | `# [N imports: a, b, ...]` |
| `self.x = y` assignments | > 2 consecutive | `# [N assignments: x, y, ...]` |
| Logging/print calls | > 1 consecutive | `# [N log statements]` |
| Consecutive blank lines | > 1 | Collapsed to single blank |

### Stage 3: TF-IDF Importance Filter

Removes non-preserved lines with the lowest IDF-weighted scores until the target token ratio is reached.

1. Score each removable line: `importance = sum(idf[term] for term in line_tokens)`.
2. Sort ascending (lowest importance = removed first).
3. Remove lines until `remaining_tokens <= total_tokens * target_ratio`.
4. Hard cap: never remove more than `max_prune_ratio` of removable lines.

Accepts optional `context_idf` override for query-aware pruning where terms rare
in the current query context are weighted more heavily.

**Strict mode:** When `strict=True` (used for named symbol chunks), Stage 3 is skipped
entirely. Semantically dense code is not thinned by IDF.

### Stage 4: Density Filter (Near-Duplicate Removal)

1. Collapse consecutive blank lines to at most one.
2. Compare each non-preserved line against the previous 20 lines using `SequenceMatcher`.
3. Remove line if `similarity_ratio > 0.85` with any recent line.
4. Preserved lines are always kept regardless of similarity.

**Config parameters:**
- `compression.target_ratio` = `0.4` (target token reduction)
- `compression.preserve_signatures` = `true`
- `compression.preserve_docstrings` = `true`
- `compression.collapse_imports` = `true`
- `compression.collapse_boilerplate` = `true`
- `compression.max_prune_ratio` = `0.7` (max 70% of removable lines pruned)
- `compression.strict_for_symbols` = `true`

---

## 10. Usage Decay Scoring

**Source:** `analytics.py` (`Analytics`)

Exponential time-decay model that makes recently-used chunks rank higher in retrieval.

### Decay formula (half-life model)

```
contribution(event) = 2 ^ (-age_days / halflife)
score(chunk) = sum(contribution(e) for e in qualifying_events(chunk))
```

Where:
- `age_days` = `(now_utc - event_timestamp).total_seconds() / 86400`
- `halflife` = `config.analytics.decay_halflife_days` (default: 7 days)
- Qualifying event types: `'selected'` and `'used'` only
- `'retrieved'` and `'discarded'` events are stored but do not contribute to scoring

### Behavior

| Age (days) | Contribution (halflife=7) |
|------------|--------------------------|
| 0          | 1.000                    |
| 7          | 0.500                    |
| 14         | 0.250                    |
| 28         | 0.063                    |

**Config parameters:**
- `analytics.decay_halflife_days` = `7`
- `analytics.session_timeout_minutes` = `30`
- `retrieval.usage_weight` = `0.2` (RRF fusion weight)

---

## 11. Injection Heuristics (Post-Phase-1 Tuning)

Design rationale for the structural precision fixes applied after initial evaluation.

### Filename boost tightening

**Problem:** Short prefix matches (3-4 chars) caused false-positive file boosting.

**Fix:** Minimum 5-character shared prefix with 60% coverage requirement. Both query term
and filename component must be >= 5 characters. Only files already in fused results are
boosted (no injection of files missed by BM25/TF-IDF).

### Import graph thresholds

**Problem:** Single-reference imports injected marginally relevant files that consumed budget.

**Fix:** Minimum 2 references required (`import_inject_threshold`), maximum 2 injections per
query. Source files prioritized over test files. Scoring differentiated by reference count
(3+ refs = 90% of top score; 2 refs = 70% of p75 score).

### Adaptive file cap

**Problem:** Fixed file limits are suboptimal across project sizes.

**Fix:** `max_files = min(max(3, n_files // 3), 8)`. Scales from 3 (small) to 8 (large).
Config override available via `retrieval.max_files`.

**Rationale:** These are structural precision fixes that improve retrieval quality without
algorithmic changes. They address false positives from overly broad matching rather than
modifying the core ranking algorithms.

---

## 12. Memory Characteristics

### Sparse TF-IDF vectors

Storage per chunk: JSON dict of `{term: weight}` for vocabulary terms present in that chunk.
Total footprint: `O(vocabulary_size * avg_terms_per_chunk)`.

| Corpus size | Typical vocabulary | Avg terms/chunk | Estimated size |
|-------------|-------------------|-----------------|----------------|
| 5K chunks   | 5,000-10,000      | 20-50           | 2-10 MB        |
| 50K chunks  | 10,000 (capped)   | 20-50           | 20-100 MB      |

### Dense embeddings (optional, not default)

When enabled (requires switching `embedding.backend` from `"tfidf"`):
`O(chunks * 384 * 4 bytes)` for 384-dimensional float32 vectors.

| Corpus size | Estimated size |
|-------------|----------------|
| 5K chunks   | ~7 MB          |
| 50K chunks  | ~73 MB         |

### ARC cache

`O(capacity * avg_chunk_size)` for live entries. Ghost lists store keys only (negligible).
Default: 500 live entries + 1000 ghost keys.

### Bloom filter

Optimal sizing at default parameters (100K capacity, 0.1% FP):
`m = ceil(-100000 * ln(0.001) / ln(2)^2)` = ~1,437,759 bits = **~175 KB**.

### SQLite configuration

- WAL mode (write-ahead logging) for concurrent read/write access.
- 64 MB page cache (`PRAGMA cache_size`).
- 256 MB memory-mapped I/O (`PRAGMA mmap_size`).

### Config toggle

`embedding.backend = "tfidf"` (default) keeps the system sparse-only, disabling dense
vector allocation entirely. This is the recommended configuration for codebases under
50K chunks where TF-IDF recall is sufficient.
