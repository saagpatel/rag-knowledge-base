"""Default chunker — recursive character-based splitting."""

from __future__ import annotations

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker


class DefaultChunker(DocumentChunker):
    """Recursive character splitting with overlap."""

    supported_file_types = ["txt", "text", "log", "rst"]

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        pieces = self._recursive_split(doc.content, chunk_size)
        merged = self._merge_small(pieces, chunk_size)
        overlapped = self._apply_overlap(merged, chunk_overlap, chunk_size)

        chunks = [
            self._make_chunk(text, i, doc) for i, text in enumerate(overlapped) if text.strip()
        ]
        return self._finalize_chunks(chunks)

    def _recursive_split(self, text: str, chunk_size: int) -> list[str]:
        """Split text recursively on decreasing separators."""
        if self._estimate_tokens(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " "]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                result: list[str] = []
                for part in parts:
                    piece = part if sep == " " else part
                    if self._estimate_tokens(piece) > chunk_size:
                        result.extend(self._recursive_split(piece, chunk_size))
                    else:
                        result.append(piece)
                return result

        # Last resort: hard split by character count
        max_chars = chunk_size * 4
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    def _merge_small(self, pieces: list[str], chunk_size: int) -> list[str]:
        """Merge consecutive small pieces until chunk_size is reached."""
        if not pieces:
            return []

        merged: list[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            combined = current + "\n\n" + piece
            if self._estimate_tokens(combined) <= chunk_size:
                current = combined
            else:
                merged.append(current)
                current = piece

        merged.append(current)
        return merged

    def _apply_overlap(
        self, pieces: list[str], overlap: int, chunk_size: int
    ) -> list[str]:
        """Add overlap from end of previous chunk to start of next."""
        if len(pieces) <= 1 or overlap <= 0:
            return pieces

        overlap_chars = overlap * 4
        result = [pieces[0]]
        for i in range(1, len(pieces)):
            prev_tail = pieces[i - 1][-overlap_chars:]
            result.append(prev_tail + "\n" + pieces[i])
        return result
