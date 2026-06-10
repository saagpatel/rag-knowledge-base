"""Embedding pipeline — convert chunks to Qdrant-ready PointStruct objects."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from qdrant_client.models import PointStruct

from rag_kb.ingestion.bm25 import BM25Vectorizer
from rag_kb.models.document import Chunk

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient

# Fixed namespace for deterministic UUID5 generation
_NAMESPACE = uuid.UUID("a3e2f8c1-7b4d-4e9a-b5c6-1d2e3f4a5b6c")

# Core payload fields that chunk metadata must not overwrite
_CORE_FIELDS = frozenset({
    "content", "file_path", "file_type", "file_hash",
    "chunk_index", "total_chunks", "token_count",
})


def generate_point_id(file_hash: str, chunk_index: int) -> str:
    """Deterministic point ID from file hash + chunk index."""
    return str(uuid.uuid5(_NAMESPACE, f"{file_hash}:{chunk_index}"))


def build_payload(chunk: Chunk) -> dict[str, object]:
    """Build Qdrant payload from a chunk, merging metadata without overwriting core fields."""
    payload: dict[str, object] = {
        "content": chunk.content,
        "file_path": chunk.file_path,
        "file_type": chunk.file_type,
        "file_hash": chunk.file_hash,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "token_count": chunk.token_count,
    }
    for key, value in chunk.metadata.items():
        if key not in _CORE_FIELDS:
            payload[key] = value
    return payload


def build_points(
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[object],
) -> list[PointStruct]:
    """Assemble PointStruct objects from chunks and their vectors.

    Raises ValueError if input lengths don't match.
    """
    if len(chunks) != len(dense_vectors) or len(chunks) != len(sparse_vectors):
        raise ValueError(
            f"Length mismatch: {len(chunks)} chunks, "
            f"{len(dense_vectors)} dense vectors, "
            f"{len(sparse_vectors)} sparse vectors"
        )

    points: list[PointStruct] = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors):
        point = PointStruct(
            id=generate_point_id(chunk.file_hash, chunk.chunk_index),
            vector={"dense": dense, "sparse": sparse},
            payload=build_payload(chunk),
        )
        points.append(point)
    return points


async def embed_chunks(
    chunks: list[Chunk],
    ollama: OllamaClient,
    bm25: BM25Vectorizer | None = None,
) -> list[PointStruct]:
    """Generate dense + sparse vectors for chunks and return PointStruct objects.

    If ``bm25`` is None, builds a BM25Vectorizer from the chunk texts.
    """
    if not chunks:
        return []

    texts = [c.content for c in chunks]

    # Dense embeddings via Ollama
    dense_vectors = await ollama.embed_batch(texts, show_progress=False)

    # Sparse vectors via BM25
    if bm25 is None:
        bm25 = BM25Vectorizer.from_texts(texts)
    sparse_vectors = bm25.vectorize_batch(texts)

    return build_points(chunks, dense_vectors, sparse_vectors)
