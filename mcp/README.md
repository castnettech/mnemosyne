# mnemosyne-mcp

MCP server for [Mnemosyne](https://github.com/castnettech/mnemosyne) -- a 6-signal hybrid retrieval engine for code, documents, and database schemas. Reduces LLM context waste by 73%.

For the full reference, see [MCP.md](../MCP.md) in the repository root.

## Install

```bash
pip install mnemosyne-mcp
```

## Register with Claude Code

```bash
claude mcp add mnemosyne -- mnemosyne-mcp
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "mnemosyne-mcp",
      "args": []
    }
  }
}
```

## Tools

### `search`

Federated search across code and document partitions. Code results use 6-signal hybrid retrieval (BM25, TF-IDF, symbol matching, usage frequency, predictive prefetch, optional dense embeddings) fused via Reciprocal Rank Fusion. Document results use BM25 + TF-IDF with isolated vocabulary. Returns labeled sections so the LLM can perform cross-type ranking.

**Parameters:**
- `query` (string, required) -- natural language or keyword query
- `budget` (integer, default 8000) -- maximum token budget
- `project_root` (string, optional) -- path to project root

### `search_docs`

Search the document partition only (PDFs, DOCX, CSVs, logs, and other non-code files). Uses BM25 and TF-IDF with an isolated vocabulary tuned for prose retrieval.

**Parameters:**
- `query` (string, required) -- natural language query
- `budget` (integer, default 8000) -- maximum token budget
- `project_root` (string, optional) -- path to project root

### `index`

Index or re-index a codebase. Incremental by default (only processes changed files). Indexes both code and document partitions.

**Parameters:**
- `project_root` (string, optional) -- path to project root
- `full` (boolean, default false) -- force full re-index

### `stats`

Show index statistics: file count, chunk count, tokens, language breakdown, chunk types.

**Parameters:**
- `project_root` (string, optional) -- path to project root

### `schema_ingest`

Ingest database schema into the index. Accepts DDL files (.sql), JSON/YAML schema snapshots, or SQLite database paths for live introspection.

**Parameters:**
- `source_path` (string, required) -- path to schema source
- `environment` (string, optional) -- tag for the schema source (e.g., "production", "staging")
- `project_root` (string, optional) -- path to project root

### `schema_stats`

Report indexed schema sources and statistics.

**Parameters:**
- `project_root` (string, optional) -- path to project root

## How it works

Mnemosyne indexes your codebase and documents into local SQLite, scoring every chunk with retrieval signals fused through Reciprocal Rank Fusion. Code gets 6-signal hybrid search; documents get BM25 + TF-IDF with isolated vocabulary. AST-aware compression strips boilerplate while preserving function signatures, control flow, and documentation.

Zero runtime dependencies beyond Python 3.11+. No API keys. No cloud services. Everything runs locally.

## License

AGPL-3.0 -- commercial licensing available from [Cast Net Technology](https://castnettechnology.com).
