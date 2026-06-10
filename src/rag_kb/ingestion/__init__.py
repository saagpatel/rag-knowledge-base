"""Ingestion pipeline — load documents and chunk them."""

from __future__ import annotations

from pathlib import Path

from rag_kb.models.document import Chunk, RawDocument

from .bm25 import BM25Vectorizer
from .bm25_store import bm25_exists, load_bm25, save_bm25
from .chunkers import get_chunker
from .loaders import get_loader
from .orchestrator import BatchIngestionResult, IngestionResult, ingest_directory, ingest_file
from .pipeline import build_payload, build_points, embed_chunks, generate_point_id


def load_document(path: str | Path) -> RawDocument:
    """Load a document using the appropriate loader."""
    path = Path(path)
    ext = path.suffix.lower()
    loader = get_loader(ext)
    return loader.load(path)


def chunk_document(
    doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
) -> list[Chunk]:
    """Chunk a document using the appropriate chunker."""
    chunker = get_chunker(doc.file_type)
    return chunker.chunk(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_and_chunk(
    path: str | Path, chunk_size: int = 512, chunk_overlap: int = 50
) -> tuple[RawDocument, list[Chunk]]:
    """Load and chunk a document in one call."""
    doc = load_document(path)
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return doc, chunks


__all__ = [
    "load_document",
    "chunk_document",
    "load_and_chunk",
    "RawDocument",
    "Chunk",
    "BM25Vectorizer",
    "BatchIngestionResult",
    "IngestionResult",
    "ingest_directory",
    "ingest_file",
    "build_payload",
    "build_points",
    "embed_chunks",
    "generate_point_id",
    "save_bm25",
    "load_bm25",
    "bm25_exists",
]
