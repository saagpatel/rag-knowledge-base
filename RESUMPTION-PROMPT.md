# Resumption Prompt

Paste this into Claude Code to begin or resume the project:

---

You are building a production-grade, 100% local RAG (Retrieval-Augmented Generation) knowledge base system. This is the foundational intelligence layer for an entire portfolio of future projects. Everything runs locally — zero cloud dependency.

## Project Context
A multi-interface RAG system on M4 Pro MacBook (48GB RAM) that ingests documents (Markdown, PDF, code, HTML, JSON, YAML, CSV, plain text), chunks them per-format, embeds via Ollama (nomic-embed-text, 768-dim), stores in Qdrant (Docker ARM64), and serves semantic search + LLM-powered Q&A through CLI, REST API (FastAPI), React web dashboard, and MCP server for Claude Code integration. Hybrid retrieval uses dense + BM25 sparse search with Reciprocal Rank Fusion.

## Current State
- **Last completed phase:** None (fresh start)
- **Current phase:** Phase 0 — Foundation
- **Next task:** Session 1 — Project scaffolding, Docker Compose, config system, SQLite schema

## What's Already Built
Nothing yet. Fresh project.

## What's NOT Built Yet
- Phase 0: Scaffolding, clients (Ollama + Qdrant), document loaders, chunkers, embedding pipeline, BM25 sparse vectors
- Phase 1: Search engine, reranker, generation engine, CLI tool, config system, benchmarking
- Phase 2: REST API (FastAPI), streaming, MCP server, auth, health monitoring
- Phase 3: React web UI (search, Q&A, collections, documents, admin), Docker production config, docs

## Immediate Next Steps
1. Create project directory structure under `~/Projects/claude-code/production/rag-knowledge-base/`
2. Set up `pyproject.toml` with all dependencies (fastapi, qdrant-client, click, httpx, aiosqlite, pymupdf, tree-sitter, sentence-transformers, tqdm, rich, pydantic>=2.0)
3. Create `docker-compose.yml` with Qdrant ARM64 image + persistent volume
4. Create `config.yaml` with all settings (Ollama host, Qdrant host, models, chunk sizes, search weights)
5. Create SQLite schema (documents, collections, queries, ingestion_jobs tables)
6. Create Makefile with setup/dev/test/docker targets

## Key Files to Read First
- `CLAUDE.md` — Project root config (tech stack, conventions, decisions, anti-patterns)
- `IMPLEMENTATION-ROADMAP.md` — Session-by-session execution plan
- `config.yaml` — All runtime settings

## Decisions Already Made (Do Not Revisit)
- **Privacy:** 100% local. No cloud APIs. No OpenAI. No Anthropic API. Ollama only.
- **Embedding model:** nomic-embed-text via Ollama (768-dim, 8192 tokens). Config-switchable.
- **Vector DB:** Qdrant in Docker (ARM64 native). HNSW with m=16, ef_construction=200.
- **Search:** Hybrid (dense + BM25 sparse) with Reciprocal Rank Fusion. Default 60/40 weights.
- **Reranker:** BGE-reranker-v2-m3 via sentence-transformers. Optional, loads on-demand.
- **Generation:** Ollama mistral:7b (configurable). Streaming support.
- **Chunk size:** 512 tokens, 50 overlap. Per-format pluggable chunkers.
- **Backend:** FastAPI (Python 3.12), async throughout.
- **Web UI:** React 19 + Vite + Tailwind CSS.
- **MCP:** FastMCP wrapping REST API. Thin wrapper pattern.
- **Metadata:** SQLite via aiosqlite. 4 tables: documents, collections, queries, ingestion_jobs.
- **Orchestration:** Docker Compose. Qdrant in container, Python app runs natively.
- **Binding:** FastAPI binds to 127.0.0.1 ONLY. Local-only system.
- **Credentials:** OS keychain via `keyring` for any API keys. Never plaintext.
- **No Alembic:** Simple version-based SQLite migrations.
- **No monolithic chunker:** Each file type gets its own chunker class in a registry.
