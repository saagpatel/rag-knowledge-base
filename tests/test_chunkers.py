"""Tests for document chunkers."""

from __future__ import annotations

from pathlib import Path

from rag_kb.ingestion import load_and_chunk
from rag_kb.ingestion.chunkers import (
    CodeChunker,
    DefaultChunker,
    MarkdownChunker,
    StructuredDataChunker,
    get_chunker,
)
from rag_kb.models.document import Chunk, RawDocument

FIXTURES = Path(__file__).parent / "fixtures"


def _make_doc(content: str, file_type: str = "txt") -> RawDocument:
    return RawDocument(
        content=content,
        metadata={},
        file_path="/test/file." + file_type,
        file_type=file_type,
        file_hash="a" * 64,
        file_size=len(content),
    )


class TestDefaultChunker:
    def test_basic_split(self):
        text = ("This is a paragraph. " * 200 + "\n\n") * 5
        doc = _make_doc(text)
        chunks = DefaultChunker().chunk(doc, chunk_size=128)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count > 0

    def test_small_text_single_chunk(self):
        doc = _make_doc("Short text.")
        chunks = DefaultChunker().chunk(doc, chunk_size=512)
        assert len(chunks) == 1

    def test_overlap_present(self):
        text = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(10)])
        doc = _make_doc(text)
        chunks = DefaultChunker().chunk(doc, chunk_size=128, chunk_overlap=20)
        if len(chunks) >= 2:
            # Second chunk should share some text from end of first chunk
            first_tail = chunks[0].content[-80:]
            assert any(word in chunks[1].content[:200] for word in first_tail.split()[:3])


class TestMarkdownChunker:
    def test_splits_on_headers(self):
        section = "Some detailed content. " * 60  # ~60 tokens per section
        content = f"# Title\n{section}\n## Section A\n{section}\n## Section B\n{section}"
        doc = _make_doc(content, "md")
        chunks = MarkdownChunker().chunk(doc, chunk_size=100)
        assert len(chunks) >= 2
        # Check heading metadata
        headings = [c.metadata.get("heading", "") for c in chunks]
        assert any("Section A" in h or "Title" in h for h in headings)

    def test_preserves_code_blocks(self):
        content = "# Code Section\n\n```python\ndef foo():\n    pass\n```\n\nMore text."
        doc = _make_doc(content, "md")
        chunks = MarkdownChunker().chunk(doc, chunk_size=512)
        # Code block should be intact in one chunk
        code_chunks = [c for c in chunks if "def foo():" in c.content]
        assert len(code_chunks) == 1
        assert "```python" in code_chunks[0].content

    def test_merges_small_sections(self):
        content = "# A\nTiny.\n## B\nAlso tiny.\n## C\nStill tiny."
        doc = _make_doc(content, "md")
        chunks = MarkdownChunker().chunk(doc, chunk_size=512)
        # Very small sections should get merged
        assert len(chunks) <= 2


class TestCodeChunker:
    def test_python_functions(self):
        code = FIXTURES / "sample.py"
        doc = _make_doc(code.read_text(), "py")
        doc.metadata["language"] = "python"
        chunks = CodeChunker().chunk(doc, chunk_size=512)
        assert len(chunks) >= 2
        # Check function_name metadata
        func_names = [c.metadata.get("function_name") for c in chunks]
        assert "greet" in func_names or "farewell" in func_names

    def test_fallback_unknown_language(self):
        doc = _make_doc("some random code content\n" * 10, "xyz")
        chunks = CodeChunker().chunk(doc, chunk_size=512)
        assert len(chunks) >= 1


class TestPDFChunker:
    def test_page_split(self):
        page_text = "Some page content. " * 60  # ~60 tokens per page
        content = f"--- Page 1 ---\n\n{page_text}\n\n--- Page 2 ---\n\n{page_text}"
        doc = _make_doc(content, "pdf")
        from rag_kb.ingestion.chunkers.pdf import PDFChunker

        chunks = PDFChunker().chunk(doc, chunk_size=100)
        assert len(chunks) >= 2
        page_nums = [c.metadata.get("page_number") for c in chunks]
        assert 1 in page_nums


class TestHTMLChunker:
    def test_html_chunking(self):
        doc = _make_doc(
            "Main Heading\n\nFirst paragraph content.\n\nSecond paragraph content.",
            "html",
        )
        from rag_kb.ingestion.chunkers.html import HTMLChunker

        chunks = HTMLChunker().chunk(doc, chunk_size=512)
        assert len(chunks) >= 1


class TestStructuredDataChunker:
    def test_json_top_level_keys(self):
        import json

        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        doc = _make_doc(json.dumps(data, indent=2), "json")
        chunks = StructuredDataChunker().chunk(doc, chunk_size=512)
        assert len(chunks) <= 3
        keys = [c.metadata.get("key") for c in chunks]
        assert "key1" in keys

    def test_csv_row_batches(self):
        csv_content = "\n".join(
            [f"Row {i}: name=Alice, age=30, city=NYC" for i in range(1, 6)]
        )
        doc = _make_doc(csv_content, "csv")
        chunks = StructuredDataChunker().chunk(doc, chunk_size=512)
        assert len(chunks) >= 1
        assert "row_range" in chunks[0].metadata


class TestChunkerRegistry:
    def test_get_chunker_md(self):
        chunker = get_chunker("md")
        assert isinstance(chunker, MarkdownChunker)

    def test_fallback_to_default(self):
        chunker = get_chunker("unknown")
        assert isinstance(chunker, DefaultChunker)


class TestChunkFields:
    def test_token_count(self):
        doc = _make_doc("Hello world. " * 20)
        chunks = DefaultChunker().chunk(doc, chunk_size=512)
        for c in chunks:
            expected = len(c.content) // 4
            assert c.token_count == expected

    def test_total_chunks_set(self):
        text = "\n\n".join([f"Paragraph {i}. " * 50 for i in range(10)])
        doc = _make_doc(text)
        chunks = DefaultChunker().chunk(doc, chunk_size=128)
        total = len(chunks)
        for c in chunks:
            assert c.total_chunks == total


class TestLoadAndChunkIntegration:
    def test_markdown_end_to_end(self):
        doc, chunks = load_and_chunk(FIXTURES / "sample.md")
        assert isinstance(doc, RawDocument)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert len(chunks) >= 1
        assert doc.file_type == "md"
