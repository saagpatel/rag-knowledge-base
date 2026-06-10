"""HTML loader using BeautifulSoup."""

from __future__ import annotations

from pathlib import Path

from rag_kb.models.document import RawDocument

from .base import DocumentLoader


class HTMLLoader(DocumentLoader):
    supported_extensions = [".html", ".htm"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw = path.read_text(encoding="utf-8")

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw, "html.parser")

        # Remove unwanted tags
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract metadata
        metadata: dict = {}
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag["content"]

        # Prefer <main> or <article>, fallback to <body>
        content_el = soup.find("main") or soup.find("article") or soup.find("body")
        if content_el:
            content = content_el.get_text(separator="\n", strip=True)
        else:
            content = soup.get_text(separator="\n", strip=True)

        return RawDocument(
            content=content,
            metadata=metadata,
            file_path=str(path.resolve()),
            file_type=path.suffix.lstrip(".").lower(),
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
