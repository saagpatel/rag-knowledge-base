# RAG Knowledge Base

A 100% local, production-grade Retrieval-Augmented Generation system. Ingest documents, generate embeddings, and query your knowledge base with semantic search and AI-powered Q&A — all running on your machine with zero cloud dependencies.

## Features

- **Multi-format ingestion** — Markdown, PDF, code (Python, JS, TS, Java, etc.), HTML, JSON, YAML, CSV, plain text
- **Hybrid search** — Dense vector search + BM25 sparse search with Reciprocal Rank Fusion
- **AI-powered Q&A** — Ask questions and get answers with source citations
- **Optional reranking** — BGE-reranker-v2-m3 for improved result relevance
- **4 interfaces** — REST API, CLI, React web dashboard, MCP server for Claude Code
- **Collection isolation** — Organize documents into separate searchable collections
- **Background ingestion** — Async directory ingest with job tracking
- **Query analytics** — Latency percentiles, query history, interface breakdowns

## Architecture

```
CLI / REST API / Web UI / MCP Server
              |
         Core Engine
              |
  +--------+----------+-----------+
  |Ingest  | Retrieve | Generate  |
  |loaders | dense    | prompts   |
  |chunkers| sparse   | Ollama    |
  |embedder| hybrid   | streaming |
  +---+----+----+-----+-----+----+
      |         |            |
  Qdrant    SQLite        Ollama
  (vectors) (metadata)   (LLM + embeddings)
```

## Prerequisites

- **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
- **Docker** (for Qdrant vector database)
- **Ollama** with models pulled:
  ```bash
  ollama pull nomic-embed-text   # embeddings (required)
  ollama pull mistral:7b          # generation (required for Q&A)
  ```
- **Node.js 20+** (only for web UI development)

## Quick Start

### Development

```bash
# 1. Clone and install
git clone <repo-url> && cd rag-knowledge-base
uv sync

# 2. Start Qdrant
docker compose up -d

# 3. Initialize database
make init-db

# 4. Start the API server
make dev

# 5. (Optional) Start the web dashboard
make dev-web
```

The API is available at `http://127.0.0.1:8000/api/docs` (Swagger UI).
The web dashboard runs at `http://127.0.0.1:5173`.

### Production (Docker)

```bash
# Build and start the full stack (API + Qdrant + Nginx)
make prod-up

# Access at http://127.0.0.1:80
```

### Ingest your first documents

```bash
# Via CLI
uv run rag ingest /path/to/docs -c my-collection

# Via API
curl -X POST http://127.0.0.1:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/docs", "collection": "my-collection"}'
```

### Search and ask

```bash
# Search
uv run rag search "How does authentication work?" -c my-collection

# Ask a question
uv run rag ask "Explain the authentication flow" -c my-collection

# Interactive search
uv run rag search -i -c my-collection
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data flow, component responsibilities |
| [Configuration](docs/CONFIGURATION.md) | Full config.yaml reference + environment variables |
| [API Reference](docs/API.md) | All 16 REST endpoints with curl examples |
| [CLI Reference](docs/CLI.md) | All commands, options, and examples |
| [MCP Setup](docs/MCP-SETUP.md) | Claude Code integration guide |
| [Deployment](docs/DEPLOYMENT.md) | Production Docker setup + backup/restore |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| API | FastAPI 0.115+ |
| Vector DB | Qdrant 1.13+ (Docker ARM64) |
| Metadata DB | SQLite via aiosqlite |
| Embeddings | Ollama + nomic-embed-text (768-dim) |
| Generation | Ollama + mistral:7b |
| Reranker | BAAI/bge-reranker-v2-m3 (optional) |
| Web UI | React 19 + Vite + Tailwind CSS |
| CLI | Click 8.1+ |
| MCP | FastMCP 2.0+ |

## Development

```bash
make test              # Run all tests
make test-integration  # Run integration tests (requires Qdrant + Ollama)
make lint              # Ruff linter
make format            # Black + isort
make type-check        # mypy strict mode
make check             # lint + type-check + test (not format)
```

## License

MIT
