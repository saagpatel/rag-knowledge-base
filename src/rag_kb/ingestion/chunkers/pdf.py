"""PDF chunker — page-aware splitting."""

from __future__ import annotations

import re

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker


class PDFChunker(DocumentChunker):
    """Split PDF content on page markers."""

    supported_file_types = ["pdf"]

    _PAGE_RE = re.compile(r"--- Page (\d+) ---")

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        pages = self._split_pages(doc.content)

        if not pages:
            return self._finalize_chunks(
                [self._make_chunk(doc.content, 0, doc, {"page_number": 1})]
                if doc.content.strip()
                else []
            )

        # Merge very short pages with the next
        merged: list[tuple[list[int], str]] = []
        min_tokens = 50
        for page_num, text in pages:
            if merged and self._estimate_tokens(merged[-1][1]) < min_tokens:
                prev_pages, prev_text = merged[-1]
                prev_pages.append(page_num)
                merged[-1] = (prev_pages, prev_text + "\n\n" + text)
            else:
                merged.append(([page_num], text))

        chunks: list[Chunk] = []
        for page_nums, text in merged:
            if not text.strip():
                continue

            if len(page_nums) == 1:
                meta = {"page_number": page_nums[0]}
            else:
                meta = {"page_numbers": page_nums}

            if self._estimate_tokens(text) > chunk_size:
                sub_pieces = self._sub_split(text, chunk_size, chunk_overlap)
                for piece in sub_pieces:
                    chunks.append(self._make_chunk(piece, len(chunks), doc, meta))
            else:
                chunks.append(self._make_chunk(text, len(chunks), doc, meta))

        return self._finalize_chunks(chunks)

    def _split_pages(self, content: str) -> list[tuple[int, str]]:
        """Split content on page markers, return (page_num, text) pairs."""
        parts = self._PAGE_RE.split(content)
        pages: list[tuple[int, str]] = []

        # parts alternates: [pre_text, page_num, page_text, page_num, page_text, ...]
        i = 1
        while i < len(parts) - 1:
            page_num = int(parts[i])
            page_text = parts[i + 1].strip()
            pages.append((page_num, page_text))
            i += 2

        return pages

    def _sub_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        from .default import DefaultChunker

        temp_doc = RawDocument(
            content=text, metadata={}, file_path="", file_type="txt", file_hash="", file_size=0
        )
        return [c.content for c in DefaultChunker().chunk(temp_doc, chunk_size, chunk_overlap)]
