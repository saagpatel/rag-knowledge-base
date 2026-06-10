"""Plain text loader with encoding detection."""

from __future__ import annotations

from pathlib import Path

import chardet

from rag_kb.models.document import RawDocument

from .base import DocumentLoader


class PlainTextLoader(DocumentLoader):
    supported_extensions = [".txt", ".text", ".log", ".rst"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw_bytes = path.read_bytes()
        detection = chardet.detect(raw_bytes)
        encoding = detection.get("encoding") or "utf-8"

        try:
            content = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            content = raw_bytes.decode("utf-8", errors="replace")
            encoding = "utf-8"

        return RawDocument(
            content=content,
            metadata={"encoding": encoding},
            file_path=str(path.resolve()),
            file_type=path.suffix.lstrip(".").lower(),
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
