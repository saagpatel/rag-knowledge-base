"""Base chunker class and chunker registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_kb.models.document import Chunk, RawDocument


class DocumentChunker(ABC):
    """Abstract base for document chunkers."""

    supported_file_types: list[str] = []

    @abstractmethod
    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        """Split a document into chunks."""

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _make_chunk(
        self,
        content: str,
        chunk_index: int,
        doc: RawDocument,
        extra_metadata: dict | None = None,
    ) -> Chunk:
        metadata = dict(doc.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        return Chunk(
            content=content,
            chunk_index=chunk_index,
            metadata=metadata,
            file_path=doc.file_path,
            file_type=doc.file_type,
            file_hash=doc.file_hash,
            token_count=self._estimate_tokens(content),
        )

    def _finalize_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total
        return chunks


class ChunkerRegistry:
    """Maps file types to chunker classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[DocumentChunker]] = {}
        self._default: type[DocumentChunker] | None = None

    def register(self, chunker_class: type[DocumentChunker]) -> None:
        for ft in chunker_class.supported_file_types:
            self._registry[ft] = chunker_class

    def set_default(self, chunker_class: type[DocumentChunker]) -> None:
        self._default = chunker_class

    def get_chunker(self, file_type: str) -> DocumentChunker:
        ft = file_type.lstrip(".")
        cls = self._registry.get(ft, self._default)
        if cls is None:
            from rag_kb.core.errors import ChunkerError

            raise ChunkerError(f"No chunker for file type: {ft}")
        return cls()


chunker_registry = ChunkerRegistry()
