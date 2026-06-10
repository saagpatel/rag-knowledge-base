"""PDF loader using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

from rag_kb.core.errors import LoaderError
from rag_kb.models.document import RawDocument

from .base import DocumentLoader


class PDFLoader(DocumentLoader):
    supported_extensions = [".pdf"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        try:
            import pymupdf
        except ImportError as e:
            raise LoaderError("pymupdf is required for PDF loading", cause=e)

        try:
            doc = pymupdf.open(str(path))
        except Exception as e:
            raise LoaderError(f"Failed to open PDF: {path}", cause=e)

        pages: list[str] = []
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                pages.append(f"--- Page {i} ---\n\n{text}")

        metadata: dict = {
            "page_count": len(doc),
        }
        pdf_meta = doc.metadata
        if pdf_meta:
            if pdf_meta.get("title"):
                metadata["title"] = pdf_meta["title"]
            if pdf_meta.get("author"):
                metadata["author"] = pdf_meta["author"]

        doc.close()

        return RawDocument(
            content="\n\n".join(pages),
            metadata=metadata,
            file_path=str(path.resolve()),
            file_type="pdf",
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
