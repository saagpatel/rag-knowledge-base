# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
│  CLI (Click)  │  REST API (FastAPI)  │  Web (React 19)  │
│               │                      │  MCP (FastMCP)   │
└───────┬───────┴──────────┬───────────┴────────┬─────────┘
        │                  │                    │
        ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                     Core Engine                          │
│                                                          │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │  Ingestion   │  │  Retrieval    │  │  Generation   │  │
│  │  Pipeline    │  │  Engine       │  │  Engine       │  │
│  │              │  │               │  │               │  │
│  │  Loaders     │  │  Dense search │  │  Prompt build │  │
│  │  Chunkers    │  │  BM25 sparse  │  │  Ollama call  │  │
│  │  Embedder    │  │  Hybrid + RRF │  │  Streaming    │  │
│  │  BM25 store  │  │  Reranker     │  │               │  │
│  └──────┬───────┘  └──────┬────────┘  └──────┬────────┘  │
└─────────┼─────────────────┼───────────────────┼──────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Qdrant     │  │   SQLite     │  │   Ollama         │
│   (vectors)  │  │   (metadata) │  │   (LLM + embed)  │
│   Port 6333  │  │   data/*.db  │  │   Port 11434     │
└──────────────┘  └──────────────┘  └──────────────────┘
```

## Data Flow

### Ingestion Pipeline

```
Document → Loader → Raw Text → Chunker → Chunks
                                           │
                               ┌───────────┴───────────┐
                               ▼                       ▼
                         Ollama embed           BM25 tokenize
                         (768-dim vector)       (sparse vector)
                               │                       │
                               └───────────┬───────────┘
                                           ▼
                                    Qdrant upsert
                                    (dense + sparse)
                                           │
                                           ▼
                                    SQLite insert
                                    (document metadata)
```

1. **Loader** detects file type and extracts raw text (7 loaders: markdown, plaintext, code, PDF, HTML, structured data)
2. **Chunker** splits text into overlapping chunks respecting format boundaries (7 chunkers: one per format)
3. **Embedder** generates 768-dimensional dense vectors via Ollama nomic-embed-text
4. **BM25 Store** generates sparse vectors for lexical search
5. **Qdrant** stores both vector types per chunk with payload metadata
6. **SQLite** records document metadata (hash, chunk count, status, timestamps)

### Retrieval Engine

```
Query → Ollama embed → Dense search ──┐
    │                                  │ RRF Fusion
    └──→ BM25 tokenize → Sparse search┘   (60/40)
                                       │
                                       ▼
                                   Merged results
                                       │
                              (optional reranker)
                                       │
                                       ▼
                                  Final results
```

- **Dense mode**: Vector similarity only (cosine distance)
- **Sparse mode**: BM25 only (lexical matching)
- **Hybrid mode** (default): Both with Reciprocal Rank Fusion (dense weight 0.6, sparse weight 0.4)
- **Reranker**: Optional BGE-reranker-v2-m3 cross-encoder for improved relevance

### Generation Engine

```
Query + Retrieved chunks → Prompt template → Ollama generate → Answer
                                                   │
                                            (optional stream)
```

## Component Responsibilities

| Component | Location | Responsibility |
|-----------|----------|---------------|
| Loaders | `src/rag_kb/ingestion/loaders/` | Extract text from each file format |
| Chunkers | `src/rag_kb/ingestion/chunkers/` | Split text into chunks respecting format boundaries |
| Orchestrator | `src/rag_kb/ingestion/orchestrator.py` | Coordinate file/directory ingestion |
| Pipeline | `src/rag_kb/ingestion/pipeline.py` | Single-file ingestion flow |
| BM25 Store | `src/rag_kb/ingestion/bm25_store.py` | Generate and store sparse vectors |
| Retrieval Engine | `src/rag_kb/retrieval/engine.py` | Search execution and result fusion |
| Query Log | `src/rag_kb/retrieval/query_log.py` | Record queries to SQLite for analytics |
| Generation Engine | `src/rag_kb/generation/engine.py` | Build prompts and call Ollama for answers |
| Ollama Client | `src/rag_kb/core/ollama_client.py` | HTTP client for Ollama (embed + generate) |
| Qdrant Manager | `src/rag_kb/core/qdrant_client.py` | Vector DB operations (CRUD, search, hybrid) |
| Database | `src/rag_kb/core/database.py` | SQLite schema, migrations, connection |
| Config | `src/rag_kb/core/config.py` | YAML + env var configuration loading |
| API | `src/rag_kb/api/` | FastAPI routes, middleware, auth, rate limiting |
| CLI | `src/rag_kb/cli/` | Click commands with Rich formatting |
| MCP | `src/rag_kb/mcp/` | FastMCP server wrapping core engine |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Privacy model | 100% local | No data leaves the machine |
| Embedding model | nomic-embed-text | 274M params, fast on Apple Silicon, 768-dim, 8192 token context |
| Vector DB | Qdrant | Rust-native, hybrid search, advanced filtering |
| Chunk size | 512 tokens, 50 overlap | Balanced recall/precision, configurable per collection |
| Search strategy | Hybrid with RRF | Dense alone misses exact terms; BM25 covers lexical precision |
| Metadata DB | SQLite | Zero-config, local, fast for document tracking and analytics |
| Generation model | mistral:7b | Good quality/speed tradeoff, swappable via config |
