# REST API Reference

Base URL: `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/api/docs` (Swagger UI)

## Response Envelope

All responses use a standard envelope:

```json
// Success
{
  "success": true,
  "data": { ... },
  "meta": {
    "request_id": "uuid",
    "timestamp": "2026-02-22T12:00:00Z"
  }
}

// Error
{
  "success": false,
  "error": {
    "code": "MACHINE_READABLE",
    "message": "Human-readable description",
    "statusCode": 404,
    "details": null
  },
  "meta": { ... }
}
```

## Authentication

Set `server.api_key` in config.yaml or `RAG_SERVER__API_KEY` env var. When set, all requests require the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-key" http://127.0.0.1:8000/api/health
```

When `api_key` is empty (default), no authentication is required.

---

## Health

### GET /api/health

Check service health.

```bash
curl http://127.0.0.1:8000/api/health
```

Response:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "ollama": { "status": "ok", "detail": null },
    "qdrant": { "status": "ok", "detail": null },
    "sqlite": { "status": "ok", "detail": null },
    "uptime_seconds": 1234.56,
    "version": "0.1.0"
  }
}
```

---

## Ingest

### POST /api/ingest

Ingest a file or directory. Files return `200` synchronously. Directories return `200` with a `job_id` for async tracking.

```bash
# Single file
curl -X POST http://127.0.0.1:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/file.md", "collection": "docs"}'

# Directory
curl -X POST http://127.0.0.1:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/docs/", "collection": "docs"}'
```

Request body:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | required | Absolute path to file or directory |
| `collection` | string | `"default"` | Target collection |
| `chunk_size` | int | null | Override chunk size (uses collection/config default) |
| `chunk_overlap` | int | null | Override chunk overlap |
| `patterns` | string[] | null | Glob patterns for directory ingest (e.g., `["*.md", "*.py"]`) |
| `force` | bool | false | Re-ingest even if file hash matches |

File response (200):
```json
{
  "data": {
    "total_files": 1,
    "processed": 1,
    "failed": 0,
    "skipped": 0,
    "results": [
      { "file_path": "/path/to/file.md", "status": "completed", "chunk_count": 5 }
    ]
  }
}
```

Directory response (200):
```json
{
  "data": {
    "job_id": "abc123",
    "status": "running",
    "total_files": 15,
    "started_at": "2026-02-22T12:00:00"
  }
}
```

---

## Search

### POST /api/search

Semantic search across a collection.

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does RAG work?", "collection": "docs", "mode": "hybrid"}'
```

Request body:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Search query text |
| `collection` | string | `"default"` | Collection to search |
| `mode` | string | `"hybrid"` | Search mode: `dense`, `sparse`, `hybrid` |
| `top_k` | int | 10 | Number of results |
| `rerank` | bool | false | Enable reranking |
| `filters` | object | null | Metadata filters (e.g., `{"file_type": "markdown"}`) |

Response (200):
```json
{
  "data": {
    "results": [
      {
        "id": "point-id",
        "score": 0.87,
        "content": "RAG combines retrieval with generation...",
        "file_path": "/docs/guide.md",
        "file_type": "markdown",
        "chunk_index": 2,
        "total_chunks": 8,
        "reranked": false
      }
    ],
    "total": 5,
    "query": "How does RAG work?",
    "collection": "docs",
    "mode": "hybrid",
    "latency_ms": 45.23
  }
}
```

---

## Ask (Q&A)

### POST /api/ask

Ask a question and get an AI-generated answer with source citations.

```bash
# Non-streaming
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the retrieval pipeline", "collection": "docs"}'

# Streaming (SSE)
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the retrieval pipeline", "collection": "docs", "stream": true}'
```

Request body:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Question to answer |
| `collection` | string | `"default"` | Collection to search for context |
| `mode` | string | `"hybrid"` | Search mode for context retrieval |
| `top_k` | int | 5 | Number of context chunks |
| `model` | string | null | Override generation model |
| `stream` | bool | false | Enable SSE streaming |

Non-streaming response (200):
```json
{
  "data": {
    "answer": "The retrieval pipeline works by...",
    "sources": [
      { "file_path": "/docs/guide.md", "score": 0.87, "chunk_index": 2, "total_chunks": 8, "file_type": "markdown" }
    ],
    "query": "Explain the retrieval pipeline",
    "model": "mistral:7b",
    "latency_ms": 2340.5,
    "context_chunks_used": 5
  }
}
```

Streaming response: `Content-Type: text/event-stream`
```
data: The
data:  retrieval
data:  pipeline
data:  works...
data: [DONE]
```

---

## Collections

### GET /api/collections

List all collections.

```bash
curl http://127.0.0.1:8000/api/collections
```

### GET /api/collections/{name}

Get collection details.

```bash
curl http://127.0.0.1:8000/api/collections/docs
```

Response:
```json
{
  "data": {
    "name": "rag_docs",
    "points_count": 150,
    "vectors_count": 150,
    "status": "green",
    "description": "Project documentation",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "nomic-embed-text"
  }
}
```

### POST /api/collections (201)

Create a new collection.

```bash
curl -X POST http://127.0.0.1:8000/api/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "description": "Project docs", "chunk_size": 256}'
```

### PUT /api/collections/{name}

Update collection settings.

```bash
curl -X PUT http://127.0.0.1:8000/api/collections/docs \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description", "chunk_size": 1024}'
```

### DELETE /api/collections/{name}

Delete a collection and all its vectors.

```bash
curl -X DELETE http://127.0.0.1:8000/api/collections/docs
```

---

## Documents

### GET /api/documents

List ingested documents with pagination and filtering.

```bash
curl "http://127.0.0.1:8000/api/documents?collection=docs&limit=20&offset=0"
```

Query parameters:
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `collection` | string | null | Filter by collection name |
| `status` | string | null | Filter by status |
| `limit` | int | 50 | Page size (1-100) |
| `offset` | int | 0 | Page offset |

### GET /api/documents/{doc_id}

Get document details.

### DELETE /api/documents/{doc_id}

Delete a document and its vectors from Qdrant.

---

## Analytics

### GET /api/stats

Aggregate query statistics.

```bash
curl "http://127.0.0.1:8000/api/stats?days=30"
```

Response includes: `total_queries`, `avg_latency_ms`, latency percentiles (p50/p95/p99), `queries_by_interface`, `queries_by_type`, `top_collections`.

### GET /api/queries

Query history with pagination and filtering.

```bash
curl "http://127.0.0.1:8000/api/queries?limit=20&interface=api&query_type=search"
```

Query parameters: `limit`, `offset`, `collection`, `interface` (api/cli/mcp), `query_type` (search/qa).

### GET /api/metrics

Performance metrics: latency percentiles, cache hit rate, active jobs.

```bash
curl http://127.0.0.1:8000/api/metrics
```

---

## Jobs

### GET /api/jobs/{job_id}

Check status of a background ingestion job.

```bash
curl http://127.0.0.1:8000/api/jobs/abc123
```

Response:
```json
{
  "data": {
    "job_id": "abc123",
    "status": "completed",
    "total_files": 15,
    "processed_files": 14,
    "failed_files": 1,
    "started_at": "2026-02-22T12:00:00",
    "completed_at": "2026-02-22T12:01:30"
  }
}
```

Status values: `running`, `completed`, `failed`.

---

## Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `COLLECTION_NOT_FOUND` | 404 | Collection does not exist |
| `DOCUMENT_NOT_FOUND` | 404 | Document does not exist |
| `FILE_NOT_FOUND` | 422 | Path does not exist on disk |
| `VALIDATION_ERROR` | 422 | Invalid request body |
| `UNAUTHORIZED` | 401 | Missing or invalid API key |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

## Rate Limiting

Default: 60 requests/minute with burst of 10. Configurable via `server.rate_limit_rpm` and `server.rate_limit_burst`. Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1708617600
```
