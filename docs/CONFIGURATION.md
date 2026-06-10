# Configuration Reference

All settings live in `config.yaml` at the project root. Every field can be overridden with environment variables using the `RAG_SECTION__KEY` pattern.

## config.yaml

```yaml
ollama:
  host: "http://127.0.0.1:11434"     # Ollama server URL
  embedding_model: "nomic-embed-text"  # Embedding model name
  generation_model: "mistral:7b"       # LLM for Q&A generation
  timeout: 120                         # Request timeout (seconds)
  max_retries: 3                       # Retry count for failed requests

qdrant:
  host: "127.0.0.1"           # Qdrant server host
  port: 6333                   # Qdrant HTTP port
  grpc_port: 6334              # Qdrant gRPC port
  collection_prefix: "rag_"   # Prefix for all collection names
  hnsw_m: 16                   # HNSW index M parameter
  hnsw_ef_construct: 200       # HNSW index ef_construct parameter

sqlite:
  path: "data/rag_kb.db"      # SQLite database file path

chunking:
  default_size: 512            # Default chunk size in tokens
  default_overlap: 50          # Default overlap between chunks
  min_chunk_size: 50           # Minimum chunk size
  max_chunk_size: 2048         # Maximum chunk size

search:
  default_top_k: 10            # Default number of results
  max_top_k: 100               # Maximum allowed top_k
  dense_weight: 0.6            # Weight for dense search in hybrid mode
  sparse_weight: 0.4           # Weight for sparse (BM25) search
  default_mode: "hybrid"       # Default search mode: dense, sparse, hybrid

generation:
  max_context_chunks: 10       # Maximum chunks passed to LLM
  temperature: 0.1             # LLM temperature (lower = more focused)
  max_tokens: 2048             # Maximum response tokens

logging:
  level: "INFO"                # Log level: DEBUG, INFO, WARNING, ERROR
  format: "json"               # Log format: json or text
  file: "logs/rag_kb.log"     # Log file path

server:
  host: "127.0.0.1"           # API bind address
  port: 8000                   # API port
  strict_startup: false        # Fail if services unavailable at startup
  api_key: ""                  # API key (empty = no auth required)
  rate_limit_rpm: 60           # Requests per minute limit
  rate_limit_burst: 10         # Burst allowance above RPM

cache:
  embedding_cache_size: 1000   # LRU cache size for embeddings
  enabled: true                # Enable/disable embedding cache
```

## Environment Variable Overrides

Pattern: `RAG_<SECTION>__<KEY>=<value>`

Double underscore (`__`) separates section from key. Case-insensitive.

### Examples

```bash
# Point to a different Ollama host
export RAG_OLLAMA__HOST=http://192.168.1.100:11434

# Use a different embedding model
export RAG_OLLAMA__EMBEDDING_MODEL=mxbai-embed-large

# Change Qdrant connection (e.g., inside Docker)
export RAG_QDRANT__HOST=qdrant
export RAG_QDRANT__PORT=6333

# Set SQLite path
export RAG_SQLITE__PATH=/app/data/rag_kb.db

# Enable API key authentication
export RAG_SERVER__API_KEY=my-secret-key

# Increase rate limits
export RAG_SERVER__RATE_LIMIT_RPM=120

# Fail fast if services are down
export RAG_SERVER__STRICT_STARTUP=true

# Change log level
export RAG_LOGGING__LEVEL=DEBUG
```

## Config Resolution Order

1. Default values (hardcoded in Pydantic models)
2. `config.yaml` file values
3. Environment variable overrides (highest priority)

For the MCP server, use `--config` to specify an alternate config path:
```bash
rag-kb-mcp --config /path/to/custom-config.yaml
```

Or set `RAG_CONFIG_PATH` environment variable.
