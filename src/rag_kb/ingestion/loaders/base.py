"""Base loader class and loader registry."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from rag_kb.core.errors import LoaderError
from rag_kb.models.document import RawDocument


class DocumentLoader(ABC):
    """Abstract base for document loaders."""

    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, path: Path) -> RawDocument:
        """Load a document from the given file path."""

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise LoaderError(f"File not found: {path}")
        if not path.is_file():
            raise LoaderError(f"Not a file: {path}")

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def _file_size(self, path: Path) -> int:
        return path.stat().st_size


class LoaderRegistry:
    """Maps file extensions to loader classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[DocumentLoader]] = {}

    def register(self, loader_class: type[DocumentLoader]) -> None:
        for ext in loader_class.supported_extensions:
            self._registry[ext] = loader_class

    def get_loader(self, extension: str) -> DocumentLoader:
        ext = extension.lower() if extension.startswith(".") else f".{extension}"
        if ext not in self._registry:
            raise LoaderError(f"Unsupported file extension: {ext}")
        return self._registry[ext]()

    def supported_extensions(self) -> list[str]:
        return sorted(self._registry.keys())


loader_registry = LoaderRegistry()
