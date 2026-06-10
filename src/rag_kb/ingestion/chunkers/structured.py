"""Structured data chunker — JSON/YAML key splitting, CSV row batching."""

from __future__ import annotations

import json

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker


class StructuredDataChunker(DocumentChunker):
    """Split structured data on top-level keys (JSON/YAML) or row batches (CSV)."""

    supported_file_types = ["json", "yaml", "yml", "csv"]

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        if doc.file_type == "csv":
            return self._chunk_csv(doc, chunk_size)
        return self._chunk_json_yaml(doc, chunk_size, chunk_overlap)

    def _chunk_json_yaml(
        self, doc: RawDocument, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        """Split on top-level keys."""
        try:
            data = json.loads(doc.content)
        except json.JSONDecodeError:
            # Fallback for content that isn't valid JSON
            from .default import DefaultChunker

            return DefaultChunker().chunk(doc, chunk_size, chunk_overlap)

        if isinstance(data, dict):
            chunks: list[Chunk] = []
            for key, value in data.items():
                text = json.dumps({key: value}, indent=2)
                if self._estimate_tokens(text) > chunk_size:
                    # Sub-split large values
                    from .default import DefaultChunker

                    temp_doc = RawDocument(
                        content=text,
                        metadata={},
                        file_path="",
                        file_type="txt",
                        file_hash="",
                        file_size=0,
                    )
                    sub = DefaultChunker().chunk(temp_doc, chunk_size, chunk_overlap)
                    for sc in sub:
                        sc.metadata["key"] = key
                        sc.chunk_index = len(chunks)
                        chunks.append(sc)
                else:
                    chunks.append(
                        self._make_chunk(text, len(chunks), doc, {"key": key})
                    )
            return self._finalize_chunks(chunks)

        # For arrays, treat as single chunk or split by items
        chunks = []
        batch: list = []
        batch_text = ""
        for i, item in enumerate(data):
            item_text = json.dumps(item, indent=2)
            test = batch_text + "\n" + item_text if batch_text else item_text
            if self._estimate_tokens(test) > chunk_size and batch:
                start = i - len(batch) + 1
                meta = {"row_range": f"{start}-{i}"}
                chunks.append(self._make_chunk(batch_text, len(chunks), doc, meta))
                batch = [item]
                batch_text = item_text
            else:
                batch.append(item)
                batch_text = test

        if batch:
            start = len(data) - len(batch) + 1
            meta = {"row_range": f"{start}-{len(data)}"}
            chunks.append(self._make_chunk(batch_text, len(chunks), doc, meta))

        return self._finalize_chunks(chunks)

    def _chunk_csv(self, doc: RawDocument, chunk_size: int) -> list[Chunk]:
        """Group CSV rows into batches approaching chunk_size."""
        lines = doc.content.strip().split("\n")
        if not lines:
            return self._finalize_chunks([])

        chunks: list[Chunk] = []
        batch_lines: list[str] = []
        batch_start = 1

        for line in lines:
            test_text = "\n".join(batch_lines + [line])
            if self._estimate_tokens(test_text) > chunk_size and batch_lines:
                batch_end = batch_start + len(batch_lines) - 1
                chunks.append(
                    self._make_chunk(
                        "\n".join(batch_lines),
                        len(chunks),
                        doc,
                        {"row_range": f"{batch_start}-{batch_end}"},
                    )
                )
                batch_lines = [line]
                batch_start = batch_end + 1
            else:
                batch_lines.append(line)

        if batch_lines:
            batch_end = batch_start + len(batch_lines) - 1
            chunks.append(
                self._make_chunk(
                    "\n".join(batch_lines),
                    len(chunks),
                    doc,
                    {"row_range": f"{batch_start}-{batch_end}"},
                )
            )

        return self._finalize_chunks(chunks)
