# Deployment Guide

## Development Setup

### Prerequisites

1. **Python 3.12+** with [uv](https://docs.astral.sh/uv/)
2. **Docker** and Docker Compose v2
3. **Ollama** — install from [ollama.ai](https://ollama.ai/)
4. **Node.js 20+** (only for web UI development)

### Steps

```bash
# 1. Install Python dependencies
uv sync

# 2. Pull required Ollama models
ollama pull nomic-embed-text
ollama pull mistral:7b

# 3. Start Qdrant vector database
docker compose up -d

# 4. Initialize SQLite database
make init-db

# 5. Start the API server (with hot reload)
make dev

# 6. (Optional) Start the React dev server
make dev-web

# 7. (Optional) Start the MCP server
make dev-mcp
```

### Verify

```bash
make health
# Should return: {"status": "healthy", ...}
```

---

## Production Docker Deployment

The production stack runs three containers: API server (Python + built React UI), Qdrant, and Nginx reverse proxy.

### Prerequisites

- Docker and Docker Compose v2
- Ollama running on the host (not containerized — needs GPU access)

### Build and Start

```bash
# Build images and start all services
make prod-up

# The app is available at http://127.0.0.1:80
```

### Architecture

```
Client → Nginx (:80) → API (:8000) → Qdrant (:6333)
                                   → SQLite (volume)
                                   → Ollama (host:11434)
```

- **Nginx** handles gzip, static asset caching, security headers, and proxies `/api/` to the API container
- **API** runs uvicorn with 2 workers, serves both the REST API and the built React UI
- **Qdrant** runs with telemetry disabled and resource limits (4GB RAM, 2 CPU)

### Container Details

| Service | Image | Port | Resources |
|---------|-------|------|-----------|
| nginx | nginx:1.27-alpine | 127.0.0.1:80 | minimal |
| api | Custom (Dockerfile) | internal:8000 | 2GB RAM, 2 CPU |
| qdrant | qdrant/qdrant:v1.13.2 | internal:6333 | 4GB RAM, 2 CPU |

### Volumes

| Volume | Purpose |
|--------|---------|
| `rag-kb-qdrant-data` | Qdrant vector storage |
| `rag-kb-api-data` | SQLite database |
| `rag-kb-api-logs` | Application logs |

### Environment Variables

The API container connects to Qdrant by service name (`qdrant`) inside the Docker network. Ollama runs on the host, so use `host.docker.internal` or the host IP:

```bash
# If Ollama is on the host machine
RAG_OLLAMA__HOST=http://host.docker.internal:11434
```

### Commands

```bash
make prod-build   # Build images only
make prod-up      # Build + start all services
make prod-down    # Stop all services
make prod-logs    # Stream logs from all services
```

---

## Data Backup and Restore

### Qdrant Snapshots

```bash
# Create snapshot of all collections
curl -X POST http://127.0.0.1:6333/snapshots

# List snapshots
curl http://127.0.0.1:6333/snapshots

# Download snapshot
curl http://127.0.0.1:6333/snapshots/<snapshot-name> -o backup.snapshot
```

### SQLite Backup

```bash
# Copy the database file
cp data/rag_kb.db data/rag_kb.db.backup

# Or for Docker volumes
docker cp rag-kb-api:/app/data/rag_kb.db ./backup_rag_kb.db
```

### Full Restore

1. Stop services: `make prod-down`
2. Restore Qdrant snapshot via Qdrant API
3. Restore SQLite file to the data volume
4. Start services: `make prod-up`

---

## Monitoring

### Health Check

```bash
curl http://127.0.0.1:8000/api/health
```

Returns individual status for Ollama, Qdrant, and SQLite. Overall status is `healthy` when all services are up, `degraded` otherwise.

### Logs

```bash
# Development
make logs-api

# Production
make prod-logs

# API logs only (JSON format)
docker logs rag-kb-api -f
```

### Metrics

```bash
curl http://127.0.0.1:8000/api/metrics
```

Returns: query count, latency percentiles, embedding cache hit rate, active ingestion jobs.
