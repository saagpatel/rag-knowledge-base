# MCP Server Setup for Claude Code

The RAG Knowledge Base includes an MCP (Model Context Protocol) server that gives Claude Code direct access to your knowledge base for search, Q&A, ingestion, and management.

## Prerequisites

- Qdrant running (`docker compose up -d`)
- Ollama running with models pulled
- SQLite database initialized (`make init-db`)

## Installation

### Option 1: From project directory (development)

```bash
# In the rag-knowledge-base directory
uv run rag-kb-mcp
```

### Option 2: Install as a tool

```bash
uv tool install /path/to/rag-knowledge-base
rag-kb-mcp
```

## Configure Claude Code

Add to your Claude Code `settings.json` (or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rag-knowledge-base", "rag-kb-mcp"],
      "env": {}
    }
  }
}
```

Or with a custom config path:

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/rag-knowledge-base", "rag-kb-mcp", "--config", "/path/to/config.yaml"],
      "env": {}
    }
  }
}
```

## CLI Options

```bash
rag-kb-mcp --help
rag-kb-mcp --config /path/to/config.yaml    # Custom config path
rag-kb-mcp --transport stdio                  # Default (for Claude Code)
rag-kb-mcp --transport sse                    # Network transport
```

## Available Tools (12)

### Search & Q&A

| Tool | Description | Key Parameters |
|------|-------------|---------------|
| `search` | Semantic search | `query`, `collection`, `mode`, `top_k`, `rerank` |
| `ask` | AI Q&A with sources | `query`, `collection`, `top_k`, `model` |

### Ingestion

| Tool | Description | Key Parameters |
|------|-------------|---------------|
| `ingest` | Ingest file or directory | `path`, `collection`, `chunk_size`, `patterns` |

### Collections

| Tool | Description | Key Parameters |
|------|-------------|---------------|
| `list_collections` | List all collections | — |
| `create_collection` | Create new collection | `name` |
| `delete_collection` | Delete collection | `name` |

### Documents

| Tool | Description | Key Parameters |
|------|-------------|---------------|
| `list_documents` | List documents | `collection`, `limit` |
| `get_document` | Document details | `doc_id` |
| `delete_document` | Delete document | `doc_id` |

### System

| Tool | Description | Key Parameters |
|------|-------------|---------------|
| `health` | Service health check | — |
| `stats` | Query statistics | `days` |
| `query_history` | Recent queries | `limit`, `collection` |

## Usage Examples in Claude Code

Once configured, Claude Code can use the tools directly:

```
> Search my docs for authentication patterns
(Claude uses the `search` tool with query="authentication patterns")

> Ingest the /Users/me/project/docs directory into the "project" collection
(Claude uses the `ingest` tool)

> What does our API documentation say about error handling?
(Claude uses the `ask` tool)
```

## Troubleshooting

**"Ollama unavailable"** — Ensure Ollama is running: `ollama serve`

**"Qdrant unavailable"** — Ensure Qdrant is running: `docker compose up -d`

**"Collection 'X' does not exist"** — Create it first: use the `create_collection` tool or `rag collections create X`

**MCP server not connecting** — Check that the `command` and `args` paths in settings.json are correct. The `--directory` flag must point to the project root where `pyproject.toml` lives.
