"""Markdown chunker — header-based splitting."""

from __future__ import annotations

import re

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker


class MarkdownChunker(DocumentChunker):
    """Split markdown on headers, preserving code blocks."""

    supported_file_types = ["md", "mdx"]

    _HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        sections = self._split_on_headers(doc.content)
        min_tokens = 50

        # Merge small sections with the next one
        merged: list[tuple[str, str]] = []
        for heading, text in sections:
            if merged and self._estimate_tokens(merged[-1][1]) < min_tokens:
                prev_heading, prev_text = merged[-1]
                merged[-1] = (prev_heading, prev_text + "\n\n" + text)
            else:
                merged.append((heading, text))

        chunks: list[Chunk] = []
        for heading, text in merged:
            if not text.strip():
                continue

            if self._estimate_tokens(text) > chunk_size:
                sub_pieces = self._sub_split(text, chunk_size, chunk_overlap)
                for piece in sub_pieces:
                    chunks.append(
                        self._make_chunk(
                            piece,
                            len(chunks),
                            doc,
                            {"heading": heading} if heading else None,
                        )
                    )
            else:
                chunks.append(
                    self._make_chunk(
                        text,
                        len(chunks),
                        doc,
                        {"heading": heading} if heading else None,
                    )
                )

        return self._finalize_chunks(chunks)

    def _split_on_headers(self, content: str) -> list[tuple[str, str]]:
        """Split content into (heading_path, text) sections."""
        # Protect code blocks by replacing them temporarily
        code_blocks: list[str] = []
        code_re = re.compile(r"```[\s\S]*?```", re.MULTILINE)

        def replace_code(m: re.Match) -> str:  # type: ignore[type-arg]
            code_blocks.append(m.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        protected = code_re.sub(replace_code, content)

        sections: list[tuple[str, str]] = []
        heading_stack: list[str] = []
        last_pos = 0

        for match in self._HEADER_RE.finditer(protected):
            # Capture text before this header
            if last_pos < match.start():
                pre_text = protected[last_pos : match.start()].strip()
                if pre_text:
                    heading_path = " > ".join(heading_stack) if heading_stack else ""
                    sections.append((heading_path, pre_text))

            level = len(match.group(1))
            title = match.group(2).strip()

            # Update heading stack
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)

            last_pos = match.end()

        # Capture remaining text
        remaining = protected[last_pos:].strip()
        if remaining:
            heading_path = " > ".join(heading_stack) if heading_stack else ""
            sections.append((heading_path, remaining))

        # Restore code blocks
        restored: list[tuple[str, str]] = []
        for heading, text in sections:
            for i, block in enumerate(code_blocks):
                text = text.replace(f"__CODE_BLOCK_{i}__", block)
            restored.append((heading, text))

        return restored

    def _sub_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Sub-split large sections using paragraph boundaries."""
        from .default import DefaultChunker

        temp_doc = RawDocument(
            content=text,
            metadata={},
            file_path="",
            file_type="txt",
            file_hash="",
            file_size=0,
        )
        sub_chunks = DefaultChunker().chunk(temp_doc, chunk_size, chunk_overlap)
        return [c.content for c in sub_chunks]
