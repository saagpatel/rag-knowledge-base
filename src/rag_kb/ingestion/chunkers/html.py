"""HTML chunker — block-element splitting."""

from __future__ import annotations

import re

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker


class HTMLChunker(DocumentChunker):
    """Split extracted HTML text on double newlines or headers."""

    supported_file_types = ["html", "htm"]

    _HEADER_RE = re.compile(r"^(#{1,3}|[A-Z][A-Za-z ]+)\s*$", re.MULTILINE)

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        # Since HTMLLoader already extracted clean text, split on double newlines
        sections = [s.strip() for s in doc.content.split("\n\n") if s.strip()]

        if not sections:
            return self._finalize_chunks([])

        # Merge small consecutive sections
        merged: list[str] = []
        current = sections[0]
        for section in sections[1:]:
            combined = current + "\n\n" + section
            if self._estimate_tokens(combined) <= chunk_size:
                current = combined
            else:
                merged.append(current)
                current = section
        merged.append(current)

        # If still too large, sub-split
        final_pieces: list[str] = []
        for piece in merged:
            if self._estimate_tokens(piece) > chunk_size:
                final_pieces.extend(self._sub_split(piece, chunk_size, chunk_overlap))
            else:
                final_pieces.append(piece)

        chunks = [
            self._make_chunk(text, i, doc) for i, text in enumerate(final_pieces) if text.strip()
        ]
        return self._finalize_chunks(chunks)

    def _sub_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        from .default import DefaultChunker

        temp_doc = RawDocument(
            content=text, metadata={}, file_path="", file_type="txt", file_hash="", file_size=0
        )
        return [c.content for c in DefaultChunker().chunk(temp_doc, chunk_size, chunk_overlap)]
