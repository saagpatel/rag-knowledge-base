"""Tests for document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_kb.core.errors import LoaderError
from rag_kb.ingestion.loaders import (
    MarkdownLoader,
    get_loader,
    loader_registry,
)
from rag_kb.models.document import RawDocument

FIXTURES = Path(__file__).parent / "fixtures"


class TestMarkdownLoader:
    def test_load_with_frontmatter(self):
        doc = get_loader(".md").load(FIXTURES / "sample.md")
        assert isinstance(doc, RawDocument)
        assert doc.metadata.get("title") == "Sample Document"
        assert doc.metadata.get("tags") == ["test", "markdown"]
        assert "---" not in doc.content.split("\n")[0]
        assert "Introduction" in doc.content

    def test_load_no_frontmatter(self, tmp_dir: Path):
        md = tmp_dir / "plain.md"
        md.write_text("# Just a heading\nSome text.")
        doc = get_loader(".md").load(md)
        assert doc.metadata == {}
        assert "Just a heading" in doc.content


class TestPlainTextLoader:
    def test_load(self):
        doc = get_loader(".txt").load(FIXTURES / "sample.txt")
        assert isinstance(doc, RawDocument)
        assert "plain text sample" in doc.content
        assert "encoding" in doc.metadata


class TestCodeLoader:
    def test_load_python(self):
        doc = get_loader(".py").load(FIXTURES / "sample.py")
        assert doc.metadata["language"] == "python"
        assert "greet" in doc.metadata["functions"]
        assert "farewell" in doc.metadata["functions"]
        assert "Calculator" in doc.metadata["classes"]

    def test_load_unsupported_extension(self, tmp_dir: Path):
        xyz = tmp_dir / "test.xyz"
        xyz.write_text("some content")
        with pytest.raises(LoaderError):
            get_loader(".xyz").load(xyz)


class TestPDFLoader:
    def test_load(self, tmp_dir: Path):
        import pymupdf

        pdf_path = tmp_dir / "test.pdf"
        doc = pymupdf.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Page one content here")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Page two content here")
        doc.save(str(pdf_path))
        doc.close()

        result = get_loader(".pdf").load(pdf_path)
        assert "Page one content" in result.content
        assert "Page two content" in result.content
        assert result.metadata["page_count"] == 2


class TestHTMLLoader:
    def test_load(self):
        doc = get_loader(".html").load(FIXTURES / "sample.html")
        assert doc.metadata.get("title") == "Sample Page"
        assert doc.metadata.get("description") == "A sample HTML page for testing"
        assert "Main Heading" in doc.content
        assert "main content" in doc.content
        # Scripts and nav should be stripped
        assert "console.log" not in doc.content
        assert "Home" not in doc.content


class TestJSONLoader:
    def test_load(self):
        doc = get_loader(".json").load(FIXTURES / "sample.json")
        assert doc.metadata["key_count"] == 3
        assert doc.metadata["data_type"] == "object"
        assert "RAG Knowledge Base" in doc.content


class TestYAMLLoader:
    def test_load(self):
        doc = get_loader(".yaml").load(FIXTURES / "sample.yaml")
        assert doc.metadata["key_count"] == 3
        assert doc.metadata["data_type"] == "object"
        assert "RAG Knowledge Base" in doc.content


class TestCSVLoader:
    def test_load(self):
        doc = get_loader(".csv").load(FIXTURES / "sample.csv")
        assert doc.metadata["row_count"] == 5
        assert "name" in doc.metadata["column_names"]
        assert "age" in doc.metadata["column_names"]
        assert "Row 1:" in doc.content
        assert "Alice" in doc.content


class TestLoaderRegistry:
    def test_get_loader_md(self):
        loader = get_loader(".md")
        assert isinstance(loader, MarkdownLoader)

    def test_unsupported_extension(self):
        with pytest.raises(LoaderError, match="Unsupported"):
            get_loader(".xyz")

    def test_supported_extensions_list(self):
        exts = loader_registry.supported_extensions()
        assert ".md" in exts
        assert ".py" in exts
        assert ".pdf" in exts
        assert ".csv" in exts


class TestFileNotFound:
    def test_nonexistent_path(self):
        with pytest.raises(LoaderError, match="File not found"):
            get_loader(".md").load(Path("/nonexistent/file.md"))


class TestFileHash:
    def test_hash_is_64_char_hex(self):
        doc = get_loader(".txt").load(FIXTURES / "sample.txt")
        assert len(doc.file_hash) == 64
        assert all(c in "0123456789abcdef" for c in doc.file_hash)
