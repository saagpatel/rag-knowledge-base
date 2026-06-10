"""Custom exception hierarchy for the RAG knowledge base."""

from __future__ import annotations


class RAGError(Exception):
    """Base exception for all RAG knowledge base errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# --- Ollama errors ---


class OllamaError(RAGError):
    """Base exception for Ollama-related errors."""


class OllamaConnectionError(OllamaError):
    """Cannot reach the Ollama server."""


class OllamaModelNotFoundError(OllamaError):
    """Requested model is not available in Ollama."""


class OllamaTimeoutError(OllamaError):
    """Ollama request timed out after retries."""


# --- Qdrant errors ---


class QdrantError(RAGError):
    """Base exception for Qdrant-related errors."""


class QdrantConnectionError(QdrantError):
    """Cannot reach the Qdrant server."""


class QdrantCollectionError(QdrantError):
    """A collection operation failed."""


# --- Ingestion errors ---


class IngestionError(RAGError):
    """Base exception for ingestion pipeline errors."""


class LoaderError(IngestionError):
    """File loading failed (unsupported format, parse error, file not found)."""


class ChunkerError(IngestionError):
    """Chunking failed."""


class EmbeddingError(IngestionError):
    """Embedding generation failed."""


class PipelineError(IngestionError):
    """Pipeline orchestration error."""


# --- Retrieval errors ---


class RetrievalError(RAGError):
    """Base exception for retrieval engine errors."""


class BM25StoreError(RetrievalError):
    """BM25 vocabulary file not found or corrupt."""


class CollectionNotFoundError(RetrievalError):
    """Requested collection does not exist in Qdrant."""


class DocumentNotFoundError(RetrievalError):
    """Requested document does not exist."""


# --- Generation errors ---


class GenerationError(RAGError):
    """Base exception for generation engine errors."""


class PromptTooLargeError(GenerationError):
    """Context would exceed model limits, or query is empty."""
