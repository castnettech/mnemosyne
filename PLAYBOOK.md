# Mnemosyne — Setup & Test Playbook

## Prerequisites

- Python 3.12+ installed
- A project directory you want to index (any codebase)

That's it. No API keys, no Docker.

---

## Step 1: Install Mnemosyne

### Option A: Editable install (recommended)

```bash
cd /path/to/mnemosyne
pip install -e .
```

### Option B: Copy the package into your project

```bash
cp -r /path/to/mnemosyne /your/project/mnemosyne

# Or just set PYTHONPATH to wherever it lives
export PYTHONPATH="/path/to/parent/of/mnemosyne:$PYTHONPATH"
```

Verify it's accessible:
```bash
python3 -c "import mnemosyne; print(mnemosyne.__version__)"
# Expected: 1.0.4
```

---

## Step 2: Initialize Mnemosyne in Your Target Project

```bash
cd /your/project
python3 -m mnemosyne init
```

**What happens:**
- Creates `.mnemosyne/` directory
- Writes `config.toml` with sensible defaults
- Creates empty `mnemosyne.db` (SQLite with WAL mode)

**Expected output:**
```
Created: /your/project/.mnemosyne/config.toml
Created: /your/project/.mnemosyne/mnemosyne.db

Mnemosyne initialised at: /your/project
Run 'mnemosyne ingest' to index your project.
```

---

## Step 3: Review and Tune Configuration (Optional)

```bash
cat .mnemosyne/config.toml
```

Key settings to consider adjusting:

```toml
[general]
# Add patterns to ignore (node_modules, .git already ignored by default)
ignore_patterns = [".git", "node_modules", "__pycache__", ".mnemosyne", "*.pyc", "*.lock"]

# Max file size to index (skip large generated files)
max_file_size_kb = 512

# File types to index
supported_extensions = [".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".sql", ".sh", ".html", ".css"]

[chunking]
# Smaller chunks = higher precision retrieval, more chunks
max_chunk_tokens = 300

[retrieval]
# Default token budget for query results
token_budget = 8000

[compression]
# Lower = more aggressive compression (0.3 = keep 30%)
target_ratio = 0.4
```

---

## Step 4: Ingest Your Codebase

```bash
python3 -m mnemosyne ingest
```

**What happens:**
1. Scans all files matching supported extensions
2. Skips files matching ignore patterns and size limits
3. Detects language per file (Python gets AST chunking, Markdown gets heading-based chunking, others get sliding window)
4. Splits each file into semantic chunks
5. Deduplicates identical chunks via SHA-256
6. Stores everything in SQLite with FTS5 full-text index
7. Builds TF-IDF vocabulary and embeds every chunk
8. Updates Bloom filter for fast re-index checks

**Expected output:**
```
Files scanned:  142
Files indexed:  142
Files skipped:  0
Files failed:   0
Chunks added:   1,203
Chunks deduped: 24
Elapsed:        1.12s
```

**To index specific files only:**
```bash
python3 -m mnemosyne ingest src/auth.py src/models.py
```

**To force full re-index (ignore cache):**
```bash
python3 -m mnemosyne ingest --full
```

**To preview what would be indexed without writing:**
```bash
python3 -m mnemosyne ingest --dry-run
```

---

## Step 5: Query Your Codebase

### Basic query
```bash
python3 -m mnemosyne query "how does authentication work"
```

Returns the most relevant chunks within the default 8,000 token budget, formatted as readable text with file paths, line numbers, and relevance scores.

### With a custom token budget
```bash
python3 -m mnemosyne query "database connection pooling" --budget 4000
```

### JSON output (for programmatic use)
```bash
python3 -m mnemosyne query "error handling patterns" --format json
```

Returns structured JSON with chunk IDs, scores, file paths, line ranges, and content -- ready for an LLM agent to parse and inject.

### With session tracking (enables delta-aware context)
```bash
export SESSION_ID="my-task-001"
python3 -m mnemosyne query "user model" --session $SESSION_ID
# ... do some work, modify files, re-ingest ...
python3 -m mnemosyne ingest
python3 -m mnemosyne query "user model" --session $SESSION_ID
# Second query will send diffs instead of full content for unchanged chunks
```

### Without compression (raw chunks)
```bash
python3 -m mnemosyne query "caching layer" --no-compress
```

---

## Step 6: Test Compression

Preview how Mnemosyne compresses a specific file:

```bash
python3 -m mnemosyne compress src/engine.py
```

**Expected output:**
```
File:            src/engine.py
Original tokens: 948
Compressed:      412
Char ratio:      43.2%

---
# [5 imports: os, sys, hashlib, pathlib, typing]

class Engine:
    def __init__(self, name: str, config: Config):
        # [4 assignments: self.name, self.config, self.cache, self.store]

    def process(self, query: str, budget: int = 8000) -> list[Chunk]:
        """Run hybrid retrieval against the index."""
        candidates = self.store.search_fts(query, limit=50)
        ...
```

**With custom compression ratio:**
```bash
python3 -m mnemosyne compress src/engine.py --ratio 0.3
```

---

## Step 7: Check Statistics

```bash
python3 -m mnemosyne stats
```

**Expected output:**
```
Project root:   /your/project
Files indexed:  142
Chunks:         1,203
Total tokens:   198,432
Usage events:   57
```

---

## Step 8: Detect Changes

After modifying files, see what changed before re-indexing:

```bash
python3 -m mnemosyne delta
```

**Expected output:**
```
Changes detected: 3 file(s)

Modified (2):
  ~ src/auth.py
  ~ src/models.py

Added (1):
  + src/new_feature.py
```

Then re-ingest to update the index:
```bash
python3 -m mnemosyne ingest
```

Only changed files are re-processed (Bloom filter + hash check skips unchanged files).

---

## Step 9: View Audit Log

```bash
python3 -m mnemosyne audit --last 10
```

Shows the last 10 operations with timestamps, files processed, chunks added, and query metrics.

---

## Step 10: Garbage Collect

Remove orphaned data from deleted files:

```bash
python3 -m mnemosyne gc
```

Preview first:
```bash
python3 -m mnemosyne gc --dry-run
```

---

## Step 11: Daemon Mode (Warm-Start Performance)

The daemon keeps indexes loaded in memory, eliminating cold-start latency on
every query. Recommended for active development sessions and production
deployments.

```bash
# Start the daemon (background process)
python3 -m mnemosyne daemon start

# Check status
python3 -m mnemosyne daemon status

# Stop when done
python3 -m mnemosyne daemon stop
```

When the daemon is running, all CLI commands (`query`, `ingest`, etc.)
automatically connect to it instead of loading indexes from scratch. Typical
query latency drops from ~200ms to <20ms on warm indexes.

---

## Step 12: View Analytics

Track feedback precision and retrieval quality over time:

```bash
python3 -m mnemosyne analytics
```

Shows feedback counts, precision metrics, and trending retrieval patterns.
Use this to validate that queries are returning useful results and to tune
retrieval weights.

---

## Step 13: Health Check

Verify the Mnemosyne installation and index integrity:

```bash
python3 -m mnemosyne health
```

Reports database status, index freshness, daemon connectivity, and any
configuration issues.

---

## Integration Test Script

Run this end-to-end validation in any project:

```bash
#!/bin/bash
set -e

echo "=== Mnemosyne Integration Test ==="

# Clean slate
rm -rf .mnemosyne

# Init
python3 -m mnemosyne init
echo "PASS: init"

# Ingest
python3 -m mnemosyne ingest
echo "PASS: ingest"

# Stats
python3 -m mnemosyne stats
echo "PASS: stats"

# Query (plain text)
python3 -m mnemosyne query "main entry point" --budget 2000
echo "PASS: query (plain)"

# Query (JSON)
python3 -m mnemosyne query "configuration" --format json --budget 1000 > /dev/null
echo "PASS: query (json)"

# Compress (pick first .py file found)
FIRST_PY=$(find . -name "*.py" -not -path "./.mnemosyne/*" | head -1)
if [ -n "$FIRST_PY" ]; then
    python3 -m mnemosyne compress "$FIRST_PY"
    echo "PASS: compress"
fi

# Delta
python3 -m mnemosyne delta
echo "PASS: delta"

# Re-ingest (should skip all -- nothing changed)
python3 -m mnemosyne ingest
echo "PASS: re-ingest (incremental)"

# Audit
python3 -m mnemosyne audit --last 5
echo "PASS: audit"

# GC dry run
python3 -m mnemosyne gc --dry-run
echo "PASS: gc"

# Health check
python3 -m mnemosyne health
echo "PASS: health"

# Analytics
python3 -m mnemosyne analytics
echo "PASS: analytics"

echo ""
echo "=== ALL TESTS PASSED ==="
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: mnemosyne` | Set `PYTHONPATH` to the parent directory of the `mnemosyne/` package |
| `0 chunks, 0 tokens` on query | Run `mnemosyne ingest` first -- the index is empty |
| All files skipped on ingest | Run `mnemosyne ingest --full` to force re-index |
| File type not indexed | Add the extension to `supported_extensions` in `.mnemosyne/config.toml` |
| Large files skipped | Increase `max_file_size_kb` in config (default: 512 KB) |
| Query returns too few results | Lower the `--budget` or adjust `retrieval.max_results` in config |
| `python3 -m mnemosyne` not found | Ensure you're in the right directory or `PYTHONPATH` is set |
