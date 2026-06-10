# RAG Knowledge Base — Discovery Summary

## Problem Statement
Knowledge is scattered across files, notes, code repos, Confluence exports, and research articles with no unified search layer. Finding specific information means manually `grep`-ing, opening files, and relying on memory. Every future project in the portfolio (Slack Ticket Deflector, Incident Review Workbench, Personal Finance Dashboard, etc.) would benefit from a semantic search + Q&A layer, but none exists. Time wasted: 15-30 min/day searching for things that should be instantly queryable. Compound waste: every downstream project re-implements its own search/retrieval logic.

## Target User
Solo developer (senior IT support engineer), multiple times daily, across 4 interfaces:
- **CLI:** Quick lookups from terminal during Claude Code sessions
- **REST API:** Other projects consume search/Q&A programmatically
- **Web UI:** Visual exploration, ingestion management, analytics
- **MCP Server:** Claude Code queries the knowledge base autonomously during development sessions

## Success Metrics
- Query latency: < 2 seconds end-to-end (embed → retrieve → generate) for 50K+ chunks
- Retrieval quality: Recall@10 ≥ 0.80 on 50-query manually curated test set
- Ingestion throughput: ≥ 5 docs/second (Markdown), ≥ 1 doc/second (PDF)
- System uptime: Qdrant + API auto-start, crash recovery without data loss
- Interface completeness: All 4 interfaces support ingest, search, Q&A, and collection management

## Scope Boundaries

**In scope:**
- Multi-format document ingestion (Markdown, PDF, code files in Python/JS/TS/Rust, HTML, plain text, JSON, YAML, CSV)
- Hybrid search (dense + BM25 sparse) with Reciprocal Rank Fusion
- Optional reranking (BGE-reranker-v2-m3)
- LLM-powered Q&A with streaming and source citations
- 4 interfaces: CLI (Click), REST API (FastAPI), Web UI (React), MCP server (FastMCP)
- Collection management (create, configure, delete)
- Document management (ingest, list, delete, stale detection, re-ingestion)
- Ingestion job tracking and progress reporting
- Query analytics and history
- Admin dashboard (system health, stats)
- Docker Compose orchestration
- Configuration via YAML with env var overrides
- Benchmarking script for quality measurement

**Out of scope:**
- Cloud API integration (no OpenAI, no Anthropic API)
- Multi-user authentication or RBAC
- Web scraping / URL ingestion
- Real-time file system watching (daemon mode)
- Knowledge graph construction
- Fine-tuning embedding or generation models
- Mobile interface
- Multi-machine distributed deployment

**Deferred to future phases (post v1.0):**
- Web scraping connector (Phase 5+)
- Confluence API live connector (Phase 5+)
- Knowledge graph overlay with Neo4j (Phase 6+)
- Agentic multi-hop retrieval (Phase 6+)
- Fine-tuned domain embedding model (Phase 7+)

## Technical Constraints
- 100% local execution — zero data leaves the machine
- M4 Pro MacBook, 48GB RAM — must fit Ollama models + Qdrant + FastAPI + React dev server simultaneously
- Docker for Qdrant only (ARM64 native image) — Python app runs natively for faster dev iteration
- Ollama must be pre-installed and running (`ollama serve`)
- Python 3.12+ required
- Node.js 20+ required (for React/Vite)

## Key Integrations
| Service | API | Auth Method | Rate Limits | Purpose |
|---------|-----|-------------|-------------|---------|
| Ollama | `http://localhost:11434/api/embeddings` | None (local) | None | Generate embeddings with nomic-embed-text |
| Ollama | `http://localhost:11434/api/generate` | None (local) | None | LLM generation with streaming |
| Ollama | `http://localhost:11434/api/chat` | None (local) | None | Chat completion for Q&A |
| Qdrant | `http://localhost:6333` (REST) / `:6334` (gRPC) | Optional API key | None (local) | Vector storage, search, filtering |
| PyMuPDF | Local library | N/A | N/A | PDF text extraction with layout awareness |
| tree-sitter | Local library | N/A | N/A | AST-based code parsing for function-level chunking |
| sentence-transformers | Local library | N/A | N/A | BGE reranker model (on-demand loading) |
