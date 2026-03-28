# Mnemosyne Benchmark: Complex Domain-Specific Query

**Query:** "How can I improve the accuracy of the detection and classification pipeline?"
**Target:** Production Python codebase (~844 source files, FastAPI backend, React frontend, PostgreSQL, Celery workers, multi-layer detection pipeline)
**Date:** 2026-03-29
**Query type:** Complex, domain-specific, requires understanding of multi-layer detection architecture

---

## Method

Two runs answering the same domain-specific technical question:

- **Run 1 (Baseline):** Standard AI agent tools — Grep, Glob, Read — with the same ignore patterns as Mnemosyne's config (excluding .venv, tests, eval, data, models, etc.)
- **Run 2 (Mnemosyne):** Single `mnemosyne query` command against a pre-built index

This query is significantly harder than a simple "what does this project do" question. It requires finding specific detection patterns, understanding how they're classified, identifying the ML pipeline vs regex pipeline, and understanding the negation and gating architecture.

---

## Results

| Metric | Baseline (Standard Tools) | Mnemosyne | Improvement |
|---|---|---|---|
| **Tool calls** | 13 | 1 | **-92%** |
| **Wall clock time** | ~72 seconds | ~1.2 seconds | **-98%** |
| **Context tokens consumed** | ~18,500 | 4,131 | **-78%** |
| **Files navigated** | 6 files, 13 read operations | 0 (auto-retrieved) | **-100%** |
| **Chunks returned** | N/A (raw file reads) | 7 ranked chunks | Structured |
| **Answer completeness** | Deep | Deep | Equivalent |
| **Navigation required** | Yes (manual file discovery) | None | Eliminated |

---

## Run 1: Baseline Detail

The standard approach required 13 sequential tool calls across 72 seconds:

| # | Tool | Target | Lines/Results | Purpose |
|---|---|---|---|---|
| 1 | Grep | Detection-related symbols across source tree | 32 files found | Find relevant files |
| 2 | Glob | Core detector filenames | 2 files found | Narrow to core files |
| 3 | Read | Primary detector module lines 1-100 | 100 lines | Understand pipeline |
| 4 | Read | Detection entry point lines 1-80 | 80 lines | Entry point |
| 5 | Read | Detector module lines 100-220 | 120 lines | Sentence processing |
| 6 | Read | ML pipeline module lines 1-80 | 80 lines | ML pipeline branch |
| 7 | Read | Detector module lines 220-370 | 150 lines | Negation patterns |
| 8 | Read | Data models lines 1-60 | 60 lines | Data models |
| 9 | Read | Detector module lines 370-570 | 200 lines | Gated patterns |
| 10 | Grep | Pattern index in detector module | 40 lines | Pattern index |
| 11 | Read | Detector module lines 590-690 | 100 lines | Pattern definitions |
| 12 | Read | Detector module lines 1857-1933 | 76 lines | Main function |
| 13 | Read | Architecture documentation | 40 lines | Architecture context |

**Total context ingested:** ~18,500 tokens (file listings + 1,046 lines of source code read across 13 operations)

The agent had to: discover which files contain detection logic, navigate into the correct layer, read the file in multiple passes (too large for single read), trace the pattern definitions, understand the ML vs regex branching, and read the architecture overview separately.

## Run 2: Mnemosyne Detail

One command:
```bash
mnemosyne query "how can I improve the accuracy of the detection and classification pipeline?" --budget 8000
```

Returned **7 ranked chunks, 4,131 tokens** from the most relevant sections — covering detection patterns, classification logic, the ML pipeline, the detection entry point, architecture documentation, technical specifications, and data models.

Every chunk directly answered the question. The retrieval engine identified both the implementation code AND the architectural documentation — providing the "what it does" and "how to improve it" context in a single query.

---

## Analysis

### This query was harder than the first benchmark

The first benchmark ("tell me the purpose of this project") was a broad conceptual query. This query requires:
- Finding the specific detection implementation (buried in a 1,933-line module)
- Understanding the dual regex/ML pipeline architecture
- Identifying the pattern classification system
- Finding the negation detection and gating logic
- Connecting the implementation to the architecture documentation

The baseline approach took 13 tool calls because it had to navigate this complexity manually. Mnemosyne's retrieval engine identified the relevant sections automatically.

### Key differences from Benchmark 1

| Metric | Benchmark 1 (Simple Query) | Benchmark 2 (Complex Query) | Trend |
|---|---|---|---|
| Baseline tool calls | 6 | 13 | +117% (complexity scales linearly for baseline) |
| Baseline time | ~27s | ~72s | +167% |
| Baseline tokens | ~9,200 | ~18,500 | +101% |
| Mnemosyne tool calls | 1 | 1 | +0% (constant) |
| Mnemosyne time | ~0.2s | ~1.2s | Slight increase (more chunks) |
| Mnemosyne tokens | 2,409 | 4,131 | +71% (proportional to answer complexity) |
| Token savings | 74% | 78% | Better on complex queries |
| Time savings | 99% | 98% | Consistent |

**The more complex the query, the more Mnemosyne saves.** Baseline tool calls and time scale linearly with query complexity. Mnemosyne stays at 1 tool call regardless.

### Cost projection

For a 10-query complex coding session:
- **Baseline:** ~185,000 context tokens (10 x 18,500), 13 tool calls per query
- **Mnemosyne:** ~41,310 context tokens (10 x 4,131), 1 tool call per query
- **Savings:** ~143,690 tokens per session (~78%)

---

## Verdict

On a complex, domain-specific query against a large production codebase:
- **78% fewer tokens** (4,131 vs 18,500)
- **98% faster** (1.2s vs 72s)
- **92% fewer tool calls** (1 vs 13)
- **Equivalent answer depth** — both approaches surfaced the detection system, ML pipeline, negation logic, and architecture

The savings increase with query complexity because baseline costs scale with navigation difficulty while Mnemosyne costs scale only with answer size.
