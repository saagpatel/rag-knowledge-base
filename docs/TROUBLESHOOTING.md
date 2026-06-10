# Troubleshooting

## Qdrant Issues

### "Qdrant not reachable" / Connection refused

**Cause:** Qdrant container is not running.

```bash
# Check container status
docker compose ps

# Start Qdrant
docker compose up -d

# Verify
curl http://127.0.0.1:6333/healthz
```

### Qdrant high memory usage

Qdrant stores vectors in memory for fast retrieval. For large collections:

```bash
# Check collection size
curl http://127.0.0.1:6333/collections

# Increase Docker memory limit in docker-compose.prod.yml
# deploy.resources.limits.memory: "8G"
```

---

## Ollama Issues

### "Ollama unavailable" / Model not found

**Cause:** Ollama is not running or the required model is not pulled.

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull nomic-embed-text
ollama pull mistral:7b

# Verify
curl http://127.0.0.1:11434/api/tags
```

### Embedding generation is slow

On first use, Ollama loads the model into memory. Subsequent calls are fast. If consistently slow:

- Check available RAM: nomic-embed-text needs ~1GB
- Check GPU offloading: `ollama ps` shows loaded models
- Reduce batch size by ingesting smaller directories

### "Model not found" after Ollama update

Ollama updates may remove models. Re-pull:

```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

---

## Search Issues

### Empty search results

1. **Collection is empty:** Check if documents were ingested successfully
   ```bash
   curl http://127.0.0.1:8000/api/collections/your-collection
   # points_count should be > 0
   ```

2. **Wrong collection name:** Collection names are prefixed with `rag_` internally. The API handles this automatically, but verify you're using the unprefixed name.

3. **Query too specific:** Try broader queries or switch to `hybrid` mode:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/search \
     -d '{"query": "your query", "collection": "docs", "mode": "hybrid"}'
   ```

### Low-quality results

- **Enable reranking:** Add `"rerank": true` to the search request (requires `sentence-transformers` dependency)
- **Switch to hybrid mode:** Combines dense and sparse search for better coverage
- **Adjust chunk size:** Smaller chunks (256 tokens) for precise retrieval, larger chunks (1024) for broader context

---

## Ingestion Issues

### "File type not supported"

Supported extensions: `.md`, `.txt`, `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.rb`, `.php`, `.pdf`, `.html`, `.htm`, `.json`, `.yaml`, `.yml`, `.csv`

### Ingestion skipped (duplicate detection)

Files are hashed on ingest. If the file content hasn't changed, it's skipped. To force re-ingestion:

```bash
# CLI
rag ingest /path/to/file.md -c docs --force

# API
curl -X POST http://127.0.0.1:8000/api/ingest \
  -d '{"path": "/path/to/file.md", "collection": "docs", "force": true}'
```

### Background job stuck

If a directory ingest job doesn't complete:

```bash
# Check job status
curl http://127.0.0.1:8000/api/jobs/JOB_ID

# Check API logs for errors
make logs-api
```

---

## Reranker Issues

### "Reranker not available"

The reranker is an optional dependency. Install it:

```bash
uv sync --group reranker
```

This downloads the `sentence-transformers` library and the BGE-reranker model (~1GB).

### Reranker slow on first use

The model loads lazily on first rerank request (~5-10 seconds). Subsequent requests are fast (~50ms).

---

## API Issues

### 401 Unauthorized

API key authentication is enabled. Include the key:

```bash
curl -H "X-API-Key: your-key" http://127.0.0.1:8000/api/health
```

Or disable auth by setting `server.api_key` to empty string in config.yaml.

### 429 Rate Limited

Default: 60 requests/minute. Increase in config:

```yaml
server:
  rate_limit_rpm: 120
  rate_limit_burst: 20
```

### 422 Validation Error

Check the `error.details` field in the response for specific field validation failures.

---

## Database Issues

### SQLite locked

Multiple processes accessing the same SQLite file can cause locking. Ensure only one API server instance runs at a time.

### Reset database

```bash
make reset-db
# This deletes the database and re-runs migrations
```

Note: This does NOT delete vectors from Qdrant. Delete collections separately if needed.

---

## Docker Production Issues

### API can't reach Qdrant

Inside Docker, services communicate by service name. The API uses `RAG_QDRANT__HOST=qdrant` automatically in `docker-compose.prod.yml`.

### API can't reach Ollama

Ollama runs on the host, not in Docker. Set:

```bash
RAG_OLLAMA__HOST=http://host.docker.internal:11434
```

On Linux, you may need `--add-host=host.docker.internal:host-gateway` in the compose file.

### Nginx returns 502 Bad Gateway

The API container hasn't started yet. Check:

```bash
docker logs rag-kb-api
```

Common cause: Qdrant health check hasn't passed yet. The API waits for Qdrant to be healthy before starting.
