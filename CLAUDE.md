# RAG Knowledge Base — Production Foundation

## Project Overview
A 100% local, production-grade Retrieval-Augmented Generation system running on M4 Pro MacBook (48GB RAM). Ingests documents (Markdown, PDF, code, HTML, JSON, YAML, CSV, plain text), chunks them per-format, generates embeddings via Ollama (nomic-embed-text), stores vectors in Qdrant (Docker ARM64), and serves semantic search + LLM-powered Q&A through 4 interfaces: CLI, REST API (FastAPI), React web dashboard, and MCP server for Claude Code. Hybrid retrieval combines dense vector search with BM25 sparse search via Reciprocal Rank Fusion. Zero cloud dependency — all data stays on the machine.

## Tech Stack
- **Language:** Python 3.12
- **API Framework:** FastAPI 0.115+
- **Vector DB:** Qdrant 1.13+ (Docker ARM64 image)
- **Metadata DB:** SQLite 3.45+ via aiosqlite
- **Embeddings:** Ollama + nomic-embed-text (768-dim, 8192 token context)
- **Generation:** Ollama + mistral:7b (configurable)
- **Reranker:** sentence-transformers + BAAI/bge-reranker-v2-m3 (optional, on-demand)
- **PDF Parsing:** PyMuPDF (fitz) 1.24+
- **Code Parsing:** tree-sitter 0.24+
- **CLI:** Click 8.1+
- **Web UI:** React 19 + Vite + Tailwind CSS
- **MCP Server:** FastMCP 2.0+
- **Process Management:** Docker Compose v2
- **Testing:** pytest + pytest-asyncio

## Architecture
```
User Interfaces (CLI, REST API, Web UI, MCP Server)
        ↓
    Core Engine (Python)
        ↓
┌───────────┬──────────────┬──────────────┐
│ Ingestion │  Retrieval   │  Generation  │
│ Pipeline  │   Engine     │   Engine     │
│ (loaders, │ (dense,      │ (prompts,    │
│ chunkers, │  sparse,     │  Ollama,     │
│ embedder) │  hybrid,     │  streaming)  │
│           │  reranker)   │              │
└─────┬─────┴──────┬───────┴──────┬───────┘
      │            │              │
┌─────▼────────────▼──────────────▼───────┐
│ Qdrant (vectors) │ SQLite (meta) │ Disk │
└─────────────────────────────────────────┘
             ↕
         Ollama (local LLM + embeddings)
```

## Development Conventions
- **Style:** Black formatter, isort imports, ruff linter. Type hints on all public functions.
- **File naming:** snake_case for Python modules. Kebab-case for config files.
- **Project structure:** `src/rag_kb/` package with subpackages: `ingestion/`, `retrieval/`, `generation/`, `api/`, `cli/`, `mcp/`, `web/`, `core/`, `models/`.
- **Git commits:** Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- **Testing:** Every module has a corresponding test file. Integration tests in `tests/integration/`. Run `pytest` before committing. Minimum: all chunkers have unit tests, all API endpoints have integration tests.
- **Config:** All settings in `config.yaml` at project root. Environment variable overrides use `RAG_` prefix (e.g., `RAG_OLLAMA_HOST`).

## Current Phase
**All phases complete** (as of commit 5a036aa, 2026-05-30)

- [x] Phase 0: Foundation — scaffolding, Docker Compose, Ollama/Qdrant clients, loaders, chunkers, embedding pipeline, BM25
- [x] Phase 1: Core Engine + CLI — retrieval engine, reranker, generation engine, CLI tool, benchmarking
- [x] Phase 2: REST API + MCP Server — 16 REST endpoints, 12 MCP tools, auth, rate limiting, streaming
- [x] Phase 3: Web UI + Polish — 7-page React dashboard, Docker production stack, full documentation
- [x] Phase 4: Production hardening — performance, analytics, query history, metrics endpoint
- [x] Phase 5: E2E tests, Docker deployment finalisation, MCP packaging

## Key Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Privacy model | 100% local, zero cloud | User requirement. No data leaves the machine. |
| Embedding model | nomic-embed-text via Ollama | 274M params, fast on Apple Silicon, 768-dim, 8192 tokens. Config-switchable. |
| Vector DB | Qdrant (Docker ARM64) | Production-grade, Rust-native, hybrid search, advanced filtering. No migration needed. |
| Chunk size | 512 tokens, 50 overlap | Industry standard for balanced recall/precision. Configurable per collection. |
| Search strategy | Hybrid (dense + BM25 sparse) with RRF | Dense alone misses exact terms. BM25 covers lexical precision. 60/40 fusion weight. |
| Reranker | BGE-reranker-v2-m3, optional, on-demand | Loads only when needed. ~50ms latency. Not required for basic search. |
| Generation model | Ollama mistral:7b (configurable) | Good quality/speed tradeoff. User swappable to any Ollama model via config. |
| Backend | FastAPI (Python 3.12) | Async, auto-OpenAPI docs, WebSocket streaming. |
| Web UI | React 19 + Vite + Tailwind | Fast dev, modern stack. |
| MCP | FastMCP wrapping REST API | Thin wrapper pattern — MCP is a frontend to the API. |
| Metadata | SQLite via aiosqlite | Zero-config, local, fast. |
| Orchestration | Docker Compose | Single `docker-compose up` starts everything. |

## Do NOT
- **Do not use cloud APIs** — no OpenAI, no Anthropic API, no cloud embeddings. Everything runs via Ollama locally.
- **Do not scaffold the entire project in one Claude Code session** — break into phases. Each session = 1 sub-phase (2-3 hours).
- **Do not store credentials in plaintext** — use OS keychain via `keyring` library for any API keys.
- **Do not use Alembic for migrations** — SQLite + simple version-based migration scripts are sufficient for this project's scope.
- **Do not bind FastAPI to 0.0.0.0** — bind to 127.0.0.1 only. This is a local-only system.
- **Do not create a monolithic chunker** — each file type gets its own chunker class registered in the chunker registry.
- **Do not cache entire document contents in SQLite** — SQLite stores metadata only. Raw docs and chunks stay on disk and in Qdrant.

<!-- portfolio-context:start -->
# Portfolio Context

## What This Project Is

rag-knowledge-base is an active local project in the /Users/d/Projects portfolio.

## Current State

**All phases complete** (as of commit 5a036aa, 2026-05-30)

- [x] Phase 0: Foundation — scaffolding, Docker Compose, Ollama/Qdrant clients, loaders, chunkers, embedding pipeline, BM25
- [x] Phase 1: Core Engine + CLI — retrieval engine, reranker, generation engine, CLI tool, benchmarking
- [x] Phase 2: REST API + MCP Server — 16 REST endpoints, 12 MCP tools, auth, rate limiting, streaming
- [x] Phase 3: Web UI + Polish — 7-page React dashboard, Docker production stack, full documentation
- [x] Phase 4: Production hardening — performance, analytics, query history, metrics endpoint
- [x] Phase 5: E2E tests, Docker deployment finalisation, MCP packaging

## Stack

- **Language:** Python 3.12
- **API Framework:** FastAPI 0.115+
- **Vector DB:** Qdrant 1.13+ (Docker ARM64 image)
- **Metadata DB:** SQLite 3.45+ via aiosqlite
- **Embeddings:** Ollama + nomic-embed-text (768-dim, 8192 token context)
- **Generation:** Ollama + mistral:7b (configurable)
- **Reranker:** sentence-transformers + BAAI/bge-reranker-v2-m3 (optional, on-demand)
- **PDF Parsing:** PyMuPDF (fitz) 1.24+
- **Code Parsing:** tree-sitter 0.24+
- **CLI:** Click 8.1+
- **Web UI:** React 19 + Vite + Tailwind CSS
- **MCP Server:** FastMCP 2.0+
- **Process Management:** Docker Compose v2
- **Testing:** pytest + pytest-asyncio

## How To Run

- Review the README and top-level scripts before the next session; this repo does not yet expose one canonical run command inside the new context block.

## Known Risks

- **Do not use cloud APIs** — no OpenAI, no Anthropic API, no cloud embeddings. Everything runs via Ollama locally.
- **Do not scaffold the entire project in one Claude Code session** — break into phases. Each session = 1 sub-phase (2-3 hours).
- **Do not store credentials in plaintext** — use OS keychain via `keyring` library for any API keys.
- **Do not use Alembic for migrations** — SQLite + simple version-based migration scripts are sufficient for this project's scope.
- **Do not bind FastAPI to 0.0.0.0** — bind to 127.0.0.1 only. This is a local-only system.
- **Do not create a monolithic chunker** — each file type gets its own chunker class registered in the chunker registry.
- **Do not cache entire document contents in SQLite** — SQLite stores metadata only. Raw docs and chunks stay on disk and in Qdrant.

## Next Recommended Move

Use this context plus the README and supporting docs to resume the next active task, then promote the repo beyond minimum-viable by capturing a dedicated handoff, roadmap, or discovery artifact.

<!-- portfolio-context:end -->
