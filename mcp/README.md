# mnemosyne-mcp

MCP server for [Mnemosyne](https://github.com/castnettech/mnemosyne-engine) -- a 6-signal hybrid code retrieval engine that reduces LLM context waste by 73%.

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

### `mnemosyne.search`

Search a codebase using 6-signal hybrid retrieval (BM25 + TF-IDF + symbol matching + usage frequency + predictive prefetch + optional dense embeddings) fused via Reciprocal Rank Fusion. Returns the most relevant code chunks within a configurable token budget, with AST-aware compression.

**Parameters:**
- `query` (string, required) -- natural language or keyword query
- `budget` (integer, default 8000) -- maximum token budget
- `project_root` (string, optional) -- path to project root

### `mnemosyne.index`

Index or re-index a codebase. Incremental by default (only processes changed files).

**Parameters:**
- `project_root` (string, optional) -- path to project root
- `full` (boolean, default false) -- force full re-index

### `mnemosyne.stats`

Show index statistics: file count, chunk count, tokens, language breakdown, chunk types.

**Parameters:**
- `project_root` (string, optional) -- path to project root

## How it works

Mnemosyne indexes your codebase into local SQLite, scoring every chunk with six retrieval signals fused through Reciprocal Rank Fusion. AST-aware compression then strips boilerplate while preserving function signatures, control flow, and documentation -- delivering exactly within your token budget.

Zero runtime dependencies beyond Python 3.11+. No API keys. No cloud services. Everything runs locally.

## License

AGPL-3.0 -- commercial licensing available from [Cast Net Technology](https://castnettechnology.com).
