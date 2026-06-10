# RAG Knowledge Base — Implementation Roadmap

## Session Strategy
Each Claude Code session should tackle ONE sub-phase. Target 2-3 hours per session. Do not attempt multiple sub-phases in a single session. Always start a session by reading CLAUDE.md, then the relevant section of this roadmap.

---

## Phase 0: Foundation (Weeks 1-2)

### Session 1: Scaffolding + Infrastructure
**Scope:** Project structure, Docker Compose, config system, SQLite schema
**Tasks:**
1. Create project directory structure:
   ```
   rag-knowledge-base/
   ├── src/rag_kb/
   │   ├── __init__.py
   │   ├── core/           # Config, logging, constants
   │   ├── models/          # Pydantic models, SQLite models
   │   ├── ingestion/       # Loaders, chunkers, embedder
   │   ├── retrieval/       # Search engine, reranker
   │   ├── generation/      # Prompt builder, Ollama client
   │   ├── api/             # FastAPI routes
   │   ├── cli/             # Click commands
   │   ├── mcp/             # MCP server
   │   └── web/             # React build output mount
   ├── web/                  # React source (separate)
   ├── tests/
   ├── config.yaml
   ├── docker-compose.yml
   ├── docker-compose.prod.yml
   ├── pyproject.toml
   ├── Makefile
   └── README.md
   ```
2. `pyproject.toml` with all dependencies (pinned versions)
3. Docker Compose: Qdrant ARM64 with persistent volume, health check
4. Config system: `config.yaml` loader with env var overrides, Pydantic settings model
5. SQLite schema: all 4 tables (documents, collections, queries, ingestion_jobs) + migration runner
6. Makefile: `make setup`, `make dev`, `make test`, `make docker-up`, `make docker-down`

**Deliverables:** Running Qdrant via Docker, SQLite initialized, config loading works
**Verification:** `make docker-up` starts Qdrant, `make setup` installs deps, `python -c "from rag_kb.core.config import get_config; print(get_config())"` prints config

### Session 2: Ollama + Qdrant Client Wrappers
**Scope:** Async client wrappers for Ollama and Qdrant with full error handling
**Tasks:**
1. Ollama client (`src/rag_kb/core/ollama_client.py`):
   - `async embed(text: str) -> list[float]` — single text embedding
   - `async embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]` — batch with progress
   - `async generate(prompt: str, model: str, stream: bool) -> str | AsyncGenerator` — text generation
   - `async chat(messages: list[dict], model: str, stream: bool)` — chat completion
   - `async health() -> bool` — check Ollama is running + model available
   - Retry logic: 3 retries with exponential backoff on connection errors
   - Connection pooling via httpx.AsyncClient
2. Qdrant client (`src/rag_kb/core/qdrant_client.py`):
   - `create_collection(name, dense_size, sparse)` — with HNSW config (m=16, ef=200)
   - `delete_collection(name)`
   - `list_collections() -> list[CollectionInfo]`
   - `upsert_points(collection, points: list[Point])` — batch upsert (≤100 per call)
   - `search_dense(collection, vector, top_k, filters) -> list[SearchResult]`
   - `search_sparse(collection, sparse_vector, top_k, filters) -> list[SearchResult]`
   - `search_hybrid(collection, dense_vector, sparse_vector, top_k, dense_weight, sparse_weight, filters) -> list[SearchResult]` — RRF fusion
   - `delete_points(collection, filter)`
   - `get_collection_info(name) -> CollectionInfo`
   - Uses `qdrant-client` Python SDK (async mode)
3. Unit tests for both clients (mock Ollama/Qdrant for fast tests)
4. Integration test: embed text → store in Qdrant → search → verify match

**Deliverables:** Working async clients for both services with retry and error handling
**Verification:** Integration test passes: embed → store → search roundtrip
**Context files:** CLAUDE.md, `src/rag_kb/core/config.py`, `config.yaml`

### Session 3: Document Loaders
**Scope:** Multi-format document loading with metadata extraction
**Tasks:**
1. Base loader interface: `DocumentLoader` abstract class with `load(path) -> list[RawDocument]`
2. Implement loaders:
   - `MarkdownLoader` — extracts frontmatter metadata, preserves header hierarchy
   - `PlainTextLoader` — simple text extraction with encoding detection
   - `CodeLoader` — tree-sitter AST parsing for Python, JavaScript, TypeScript, Rust. Extracts functions, classes, methods with docstrings. Falls back to regex for unsupported languages.
   - `PDFLoader` — PyMuPDF extraction with page numbers, handles multi-column layouts
   - `HTMLLoader` — BeautifulSoup, strips nav/footer/scripts, extracts main content
   - `StructuredDataLoader` — JSON, YAML, CSV with schema-aware parsing
3. Loader registry: auto-selects loader by file extension
4. Unit tests: one test file per format, verify clean extraction + correct metadata

**Deliverables:** All loaders working, registered, tested
**Verification:** Load 1 file of each type, print extracted text + metadata, verify correctness
**Context files:** CLAUDE.md, `src/rag_kb/core/`, `src/rag_kb/models/`

### Session 4: Chunking + Embedding + Storage Pipeline
**Scope:** Per-format chunking, embedding generation, BM25 sparse vectors, Qdrant storage
**Tasks:**
1. Chunking registry: pluggable per-format chunkers
   - `MarkdownChunker` — splits on headers (H1/H2/H3), respects code blocks, merges small sections
   - `CodeChunker` — splits on function/class boundaries via tree-sitter AST nodes
   - `PDFChunker` — splits on page boundaries, merges paragraphs within pages
   - `DefaultChunker` — RecursiveCharacterTextSplitter(512 tokens, 50 overlap) for everything else
   - Each chunk includes: content, chunk_index, total_chunks, heading (if applicable), source metadata
2. Embedding pipeline:
   - Batch embed all chunks via Ollama client (configurable batch size)
   - Progress bar during embedding (tqdm)
3. BM25 sparse vector generation:
   - Build vocabulary from corpus (top 30K terms by TF-IDF)
   - Generate sparse vectors per chunk (term frequencies weighted by IDF)
   - Store as Qdrant sparse vectors in the `bm25` named vector
4. Ingestion orchestrator: `ingest_file(path, collection)` and `ingest_directory(path, collection, recursive)`
   - Registers documents in SQLite
   - Runs loader → chunker → embedder → Qdrant upsert pipeline
   - Tracks progress in ingestion_jobs table
   - Skips files with unchanged hash (stale detection)
5. Ingest the 50-doc test corpus, verify search works

**Deliverables:** Complete ingestion pipeline, 50 docs in Qdrant, search functional
**Verification:** `search_hybrid("how to reset password")` returns relevant chunks from ingested docs
**Context files:** CLAUDE.md, `src/rag_kb/core/`, `src/rag_kb/ingestion/`, `src/rag_kb/models/`

---

## Phase 1: Core Engine + CLI (Weeks 3-4)

### Session 5: Search Engine + Reranker
**Scope:** Unified search interface, reranker integration, metadata filtering
**Tasks:**
1. Search engine (`src/rag_kb/retrieval/search_engine.py`):
   - `async search(query, collection, mode, top_k, filters) -> list[SearchResult]`
   - Modes: dense, sparse, hybrid (default)
   - Metadata filters: file_type, filename, date range, custom tags
   - RRF fusion with configurable weights (default 0.6 dense / 0.4 sparse)
2. Reranker (`src/rag_kb/retrieval/reranker.py`):
   - Lazy-loads BGE-reranker-v2-m3 on first use
   - `rerank(query, results, top_k) -> list[SearchResult]` — rescores top 50 → returns top k
   - Auto-unloads after configurable idle timeout to free memory
3. Result formatting: score normalization, snippet extraction, highlight generation
4. Tests: search quality tests against test corpus (assert expected docs in top 10)

**Deliverables:** Search engine with hybrid + reranker working
**Verification:** Benchmark script shows recall@10 ≥ 0.80
**Context files:** CLAUDE.md, `src/rag_kb/core/`, `src/rag_kb/retrieval/`, `src/rag_kb/models/`

### Session 6: Generation Engine
**Scope:** Prompt building, Ollama generation with streaming, source citation
**Tasks:**
1. Prompt builder (`src/rag_kb/generation/prompt_builder.py`):
   - System prompt template with instructions for source citation
   - Context formatter: retrieved chunks → numbered context blocks with source metadata
   - Handles context window limits: truncate chunks if total exceeds model context
2. Generation engine (`src/rag_kb/generation/generator.py`):
   - `async answer(question, context_chunks, model, stream) -> Answer`
   - Streaming mode returns AsyncGenerator of tokens
   - Non-streaming mode returns complete answer
   - Source extraction: parses answer for citation references, maps to original chunks
3. Answer model: text, sources (with document name, chunk index, relevance score), latency_ms, model used
4. Tests: Q&A against test corpus, verify answers cite correct sources

**Deliverables:** Q&A pipeline with streaming and citations
**Verification:** Ask 5 questions about ingested docs, verify accurate answers with sources
**Context files:** CLAUDE.md, `src/rag_kb/retrieval/`, `src/rag_kb/generation/`, `src/rag_kb/models/`

### Session 7: CLI Tool
**Scope:** Complete Click CLI with all commands
**Tasks:**
1. CLI structure (`src/rag_kb/cli/`):
   - `rag ingest <path> [--collection NAME] [--recursive]` — ingest file or directory
   - `rag search <query> [--collection NAME] [--mode hybrid|dense|sparse] [--top-k 10]` — search
   - `rag ask <question> [--collection NAME] [--model mistral:7b]` — Q&A with streaming output
   - `rag collections list` — list all collections with stats
   - `rag collections create <name> [--description TEXT]` — create collection
   - `rag collections delete <name> [--confirm]` — delete collection
   - `rag documents list [--collection NAME]` — list documents
   - `rag documents delete <id> [--confirm]` — delete document
   - `rag status` — system status (Qdrant, Ollama, collections, disk usage)
   - `rag benchmark [--collection NAME]` — run quality benchmark
2. Rich output: colored terminal output, progress bars, formatted tables
3. Error handling: clear messages for common issues (Qdrant down, Ollama not running, model not pulled)
4. `--help` on every command with examples

**Deliverables:** Fully functional CLI tool
**Verification:** All commands work with real data, --help is complete
**Context files:** CLAUDE.md, `src/rag_kb/core/`, `src/rag_kb/ingestion/`, `src/rag_kb/retrieval/`, `src/rag_kb/generation/`

### Session 8: Config System + Stale Detection + Benchmarking
**Scope:** Full configuration, change detection, quality measurement
**Tasks:**
1. Config refinement: validate all settings on startup, warn on invalid values, print active config
2. Stale detection: on re-ingest, compare file hash. Skip unchanged, re-embed changed, mark deleted as stale
3. Benchmarking script:
   - Curated test set: 50 queries with expected document matches (JSON file)
   - Measures: recall@5, recall@10, MRR, avg latency, p95 latency
   - Outputs markdown report
   - Runs via `rag benchmark`
4. Logging: structured JSON logging, configurable level, log to file + stdout

**Deliverables:** Production-ready config, stale detection, benchmark baseline
**Verification:** Benchmark report generated, recall@10 ≥ 0.80
**Context files:** CLAUDE.md, `config.yaml`, `src/rag_kb/core/`, `src/rag_kb/ingestion/`

---

## Phase 2: REST API + MCP Server (Weeks 5-6)

### Session 9: FastAPI REST API
**Scope:** All REST endpoints, request validation, error handling
**Tasks:**
1. API structure (`src/rag_kb/api/`):
   - `routes/collections.py` — CRUD for collections
   - `routes/documents.py` — list, get, delete documents
   - `routes/ingest.py` — file upload, directory ingest, job status
   - `routes/search.py` — search + Q&A endpoints
   - `routes/admin.py` — stats, health, query history
   - `middleware.py` — CORS, optional auth, request logging
   - `main.py` — FastAPI app assembly, lifespan events
2. Pydantic v2 request/response models for all endpoints
3. OpenAPI docs at /docs with examples
4. Background task: async ingestion via FastAPI BackgroundTasks
5. Error handling middleware: consistent error response format

**Deliverables:** Full REST API, all endpoints working
**Verification:** Hit every endpoint via /docs UI, all return correct responses
**Context files:** CLAUDE.md, `src/rag_kb/core/`, `src/rag_kb/models/`

### Session 10: Streaming + Query Analytics
**Scope:** SSE streaming for Q&A, query logging, analytics endpoint
**Tasks:**
1. SSE streaming endpoint: `POST /api/v1/qa/stream` returns Server-Sent Events with tokens
2. Query logging middleware: every search/qa request logged to SQLite with latency, interface, results
3. Analytics endpoint: `GET /api/v1/stats` returns aggregated stats (queries/day, avg latency, top collections)
4. Query history endpoint: `GET /api/v1/queries?limit=50&collection=X` with filtering
5. Tests: httpx AsyncClient tests for all endpoints including streaming

**Deliverables:** Streaming Q&A, analytics pipeline
**Verification:** Stream a Q&A response via curl, check query appears in history
**Context files:** CLAUDE.md, `src/rag_kb/api/`, `src/rag_kb/models/`

### Session 11: MCP Server
**Scope:** FastMCP server wrapping the REST API for Claude Code integration
**Tasks:**
1. MCP server (`src/rag_kb/mcp/server.py`):
   - Tool: `search(query, collection, top_k)` — semantic search
   - Tool: `ask(question, collection, model)` — Q&A with sources
   - Tool: `ingest(file_path, collection)` — ingest a file
   - Tool: `list_collections()` — list available collections
   - Tool: `list_documents(collection)` — list documents in collection
2. Tool descriptions optimized for LLM understanding (so Claude knows when to use each tool)
3. SSE transport for Claude Code connection
4. Error handling: graceful failures, clear error messages
5. Launch script: `python -m rag_kb.mcp` starts MCP server
6. Test: connect Claude Code, verify tools are discoverable, run search + ask

**Deliverables:** Working MCP server, Claude Code integration tested
**Verification:** Claude Code connects, `search` and `ask` tools work
**Context files:** CLAUDE.md, `src/rag_kb/api/`, `src/rag_kb/mcp/`

### Session 12: API Auth + Health Monitoring
**Scope:** Optional API key auth, comprehensive health endpoint
**Tasks:**
1. Optional API key middleware (off by default, configurable in config.yaml)
2. Health endpoint checks: Qdrant connection, Ollama availability, disk space, memory, model availability
3. Startup validation: verify Qdrant reachable, verify Ollama running + model pulled, verify SQLite writable
4. Graceful degradation: if Qdrant down, return 503 with specific error; if Ollama down, search still works (no Q&A)
5. Integration tests for auth + health scenarios

**Deliverables:** Auth system, robust health monitoring
**Verification:** Kill Qdrant → /health reports degraded. Enable auth → unauthenticated requests rejected.
**Context files:** CLAUDE.md, `src/rag_kb/api/`, `config.yaml`

---

## Phase 3: Web UI + Polish (Weeks 7-8)

### Session 13: Web UI — Search + Q&A Views
**Scope:** React app scaffolding, search view, Q&A view with streaming
**Tasks:**
1. React + Vite + Tailwind scaffolding in `web/` directory
2. API client: typed fetch wrapper for all REST endpoints
3. Search view: query input, collection dropdown, mode selector, results list with highlighted snippets, score indicators, source attribution
4. Q&A view: question input, streaming answer display (typewriter), source cards, answer history (session-only)
5. Layout: sidebar navigation, responsive design, dark mode default

**Deliverables:** Search and Q&A views functional
**Verification:** Search returns results, Q&A streams answer in real-time
**Context files:** CLAUDE.md, `web/`, API route files for reference

### Session 14: Web UI — Collections + Documents Views
**Scope:** Collection management, document browser, chunk preview
**Tasks:**
1. Collections view: list with stats (doc count, chunks, size), create dialog, delete with confirmation, settings editor (chunk size, weights, model)
2. Documents view: sortable table (name, type, chunks, status, date), click to expand → chunk preview list, delete with confirmation, re-ingest button
3. Ingestion panel: drag-and-drop file upload, directory path input, progress bar for active jobs, job history table
4. Toast notifications for actions (ingested, deleted, errors)

**Deliverables:** Full collection and document management UI
**Verification:** Create collection → ingest files → browse documents → view chunks → delete
**Context files:** CLAUDE.md, `web/`, `src/rag_kb/api/routes/`

### Session 15: Web UI — Admin Dashboard + Polish
**Scope:** Admin view, system health, analytics charts, final polish
**Tasks:**
1. Admin dashboard: system health cards (Qdrant/Ollama status, disk usage, memory), query analytics (queries/day chart, avg latency chart, top collections pie chart), ingestion job monitor
2. Charts: use Recharts library for all data visualization
3. Polish: loading states, empty states, error boundaries, 404 page, favicon
4. Build configuration: Vite builds to `src/rag_kb/web/static/`, FastAPI serves static files

**Deliverables:** Complete admin dashboard, polished UI
**Verification:** Dashboard shows real-time stats, charts render, all edge cases handled
**Context files:** CLAUDE.md, `web/`, `src/rag_kb/api/`

### Session 16: Docker Compose Production + Documentation
**Scope:** Production deployment config, README, final integration testing
**Tasks:**
1. `docker-compose.prod.yml`: Qdrant with persistent volume + resource limits, API server container, Nginx reverse proxy serving React build + API proxy, log rotation, auto-restart policies
2. Makefile additions: `make prod-up`, `make prod-down`, `make prod-logs`, `make build-web`
3. README.md: quick start (5-minute setup), architecture overview, configuration reference, API reference link, MCP setup for Claude Code, troubleshooting FAQ
4. End-to-end integration test: fresh clone → setup → ingest → search via CLI → Q&A via API → verify via Web UI → connect MCP
5. Update CLAUDE.md with completed phases

**Deliverables:** Production-ready deployment, comprehensive documentation
**Verification:** Fresh clone → follow README → system running in <15 minutes
**Context files:** CLAUDE.md, all project files

---

## Context Management

- **Always include** in every session: `CLAUDE.md`
- **Phase 0 sessions:** Add `config.yaml`, `src/rag_kb/core/`, `src/rag_kb/models/`
- **Phase 1 sessions:** Add `src/rag_kb/ingestion/`, `src/rag_kb/retrieval/`, `src/rag_kb/generation/`
- **Phase 2 sessions:** Add `src/rag_kb/api/`, `src/rag_kb/mcp/`
- **Phase 3 sessions:** Add `web/src/`, `src/rag_kb/api/routes/`
- **Maximum context files per session:** 5-7 (keep it focused)
- **After each session:** Update CLAUDE.md "Current Phase" checkboxes
