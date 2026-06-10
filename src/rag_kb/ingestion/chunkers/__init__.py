"""Document chunkers — registry-based format-aware chunking."""

from .base import ChunkerRegistry, DocumentChunker, chunker_registry
from .code import CodeChunker
from .default import DefaultChunker
from .html import HTMLChunker
from .markdown import MarkdownChunker
from .pdf import PDFChunker
from .structured import StructuredDataChunker

# Set default
chunker_registry.set_default(DefaultChunker)

# Register specific chunkers
for _cls in [MarkdownChunker, CodeChunker, PDFChunker, HTMLChunker, StructuredDataChunker]:
    chunker_registry.register(_cls)


def get_chunker(file_type: str) -> DocumentChunker:
    """Get the appropriate chunker for a file type."""
    return chunker_registry.get_chunker(file_type)


__all__ = [
    "DocumentChunker",
    "ChunkerRegistry",
    "chunker_registry",
    "get_chunker",
    "DefaultChunker",
    "MarkdownChunker",
    "CodeChunker",
    "PDFChunker",
    "HTMLChunker",
    "StructuredDataChunker",
]
