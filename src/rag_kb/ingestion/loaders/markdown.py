"""Markdown document loader with frontmatter support."""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_kb.models.document import RawDocument

from .base import DocumentLoader


class MarkdownLoader(DocumentLoader):
    supported_extensions = [".md", ".mdx"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw = path.read_text(encoding="utf-8")
        metadata: dict = {}
        content = raw

        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1])
                    if isinstance(fm, dict):
                        metadata = fm
                    content = parts[2].strip()
                except yaml.YAMLError:
                    content = raw

        return RawDocument(
            content=content,
            metadata=metadata,
            file_path=str(path.resolve()),
            file_type=path.suffix.lstrip(".").lower(),
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
