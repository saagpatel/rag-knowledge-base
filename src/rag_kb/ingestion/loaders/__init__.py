"""Document loaders — registry-based format detection."""

from .base import DocumentLoader, LoaderRegistry, loader_registry
from .code import CodeLoader
from .html import HTMLLoader
from .markdown import MarkdownLoader
from .pdf import PDFLoader
from .plaintext import PlainTextLoader
from .structured import CSVLoader, JSONLoader, YAMLLoader

# Register all loaders
for _cls in [
    MarkdownLoader,
    PlainTextLoader,
    CodeLoader,
    PDFLoader,
    HTMLLoader,
    JSONLoader,
    YAMLLoader,
    CSVLoader,
]:
    loader_registry.register(_cls)


def get_loader(extension: str) -> DocumentLoader:
    """Get the appropriate loader for a file extension."""
    return loader_registry.get_loader(extension)


__all__ = [
    "DocumentLoader",
    "LoaderRegistry",
    "loader_registry",
    "get_loader",
    "MarkdownLoader",
    "PlainTextLoader",
    "CodeLoader",
    "PDFLoader",
    "HTMLLoader",
    "JSONLoader",
    "YAMLLoader",
    "CSVLoader",
]
