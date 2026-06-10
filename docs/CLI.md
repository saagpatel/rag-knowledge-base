# CLI Reference

The CLI is available via `uv run rag` (or just `rag` if installed globally).

## Global Options

```bash
rag --version    # Show version
rag --help       # Show help
```

---

## Commands

### rag ingest

Ingest a file or directory into the knowledge base.

```bash
rag ingest PATH [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--collection` | `-c` | `default` | Target collection |
| `--chunk-size` | | 512 | Chunk size in tokens |
| `--chunk-overlap` | | 50 | Overlap between chunks |
| `--patterns` | `-p` | all supported | Glob patterns for directories (repeatable) |

```bash
# Ingest a single file
rag ingest /path/to/document.md -c my-collection

# Ingest a directory
rag ingest /path/to/docs/ -c my-collection

# Ingest only Markdown and Python files
rag ingest /path/to/project/ -c code -p "*.md" -p "*.py"

# Custom chunk size
rag ingest /path/to/long-docs/ -c papers --chunk-size 1024 --chunk-overlap 100
```

Shows a progress bar for directories and a summary of processed/failed/skipped files.

---

### rag search

Search the knowledge base for relevant documents.

```bash
rag search QUERY [OPTIONS]
rag search --interactive [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--collection` | `-c` | `default` | Collection to search |
| `--mode` | `-m` | `hybrid` | Search mode: `dense`, `sparse`, `hybrid` |
| `--top-k` | `-k` | 10 | Number of results |
| `--rerank` | | false | Enable reranking |
| `--interactive` | `-i` | false | Enter interactive search loop |

```bash
# Basic search
rag search "How does authentication work?" -c docs

# Dense-only search with reranking
rag search "auth flow" -c docs -m dense --rerank

# Interactive mode (type queries, 'quit' to exit)
rag search -i -c docs
```

---

### rag ask

Ask a question and get an AI-generated answer with source citations.

```bash
rag ask QUERY [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--collection` | `-c` | `default` | Collection for context |
| `--mode` | `-m` | `hybrid` | Search mode |
| `--top-k` | `-k` | 5 | Number of context chunks |
| `--model` | | config default | Override generation model |
| `--no-stream` | | false | Disable streaming output |

```bash
# Ask with streaming (default)
rag ask "Explain the authentication flow" -c docs

# Ask without streaming
rag ask "What is RAG?" -c docs --no-stream

# Use a different model
rag ask "Summarize the API" -c docs --model llama3:8b
```

By default, tokens stream to the terminal as they are generated.

---

### rag collections

Manage collections.

```bash
rag collections list                    # List all collections
rag collections info NAME               # Show collection details
rag collections create NAME             # Create a new collection
rag collections delete NAME             # Delete (with confirmation)
rag collections delete NAME -y          # Delete without confirmation
```

---

### rag documents

Manage ingested documents.

```bash
rag documents list                      # List all documents
rag documents list -c my-collection     # Filter by collection
rag documents list -n 20                # Limit results
rag documents info DOC_ID               # Show document details
rag documents delete DOC_ID             # Delete (with confirmation)
rag documents delete DOC_ID -y          # Delete without confirmation
```

---

### rag benchmark

Run a retrieval benchmark against a test set.

```bash
rag benchmark TESTSET_PATH [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--collection` | `-c` | `default` | Collection to benchmark |
| `--mode` | `-m` | `hybrid` | Search mode |
| `--top-k` | `-k` | 10 | Number of results per query |
| `--output` | `-o` | null | Save report to file |

The test set is a JSON file with query/expected-result pairs.

---

### rag status

Check service health and configuration.

```bash
rag status
```

Shows: Ollama status, Qdrant status, configured models, database path.

---

## Supported File Types

| Extension | Loader | Chunker |
|-----------|--------|---------|
| `.md` | MarkdownLoader | MarkdownChunker (preserves headers) |
| `.txt` | PlaintextLoader | DefaultChunker (token-based) |
| `.py`, `.js`, `.ts`, `.java`, etc. | CodeLoader (tree-sitter) | CodeChunker (syntax-aware) |
| `.pdf` | PdfLoader (PyMuPDF) | PdfChunker (page boundaries) |
| `.html` | HtmlLoader (BeautifulSoup) | HtmlChunker (semantic tags) |
| `.json`, `.yaml`, `.csv` | StructuredLoader | StructuredChunker (row/key-based) |
