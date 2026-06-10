"""Code chunker — AST-boundary splitting."""

from __future__ import annotations

from rag_kb.models.document import Chunk, RawDocument

from .base import DocumentChunker

# Map file extensions to tree-sitter language names
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "h": "c",
    "hpp": "cpp",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin",
    "scala": "scala",
    "lua": "lua",
    "sh": "bash",
    "bash": "bash",
}

# AST node types that represent top-level definitions
DEFINITION_TYPES = {
    "function_definition",
    "function_declaration",
    "class_definition",
    "class_declaration",
    "method_definition",
    "method_declaration",
    "impl_item",
    "struct_item",
    "enum_item",
}


class CodeChunker(DocumentChunker):
    """Split code on function/class boundaries using tree-sitter."""

    supported_file_types = list(EXTENSION_TO_LANGUAGE.keys())

    def chunk(
        self, doc: RawDocument, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[Chunk]:
        language = EXTENSION_TO_LANGUAGE.get(doc.file_type, "")
        if not language:
            return self._fallback_chunk(doc, chunk_size, chunk_overlap)

        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language)
            source = doc.content.encode("utf-8")
            tree = parser.parse(source)
        except Exception:
            return self._fallback_chunk(doc, chunk_size, chunk_overlap)

        segments = self._extract_segments(tree.root_node, source)
        if not segments:
            return self._fallback_chunk(doc, chunk_size, chunk_overlap)

        chunks: list[Chunk] = []
        for name, text in segments:
            extra = {}
            if name:
                if "class" in name.lower() or name[0].isupper():
                    extra["class_name"] = name
                else:
                    extra["function_name"] = name
            chunks.append(self._make_chunk(text, len(chunks), doc, extra or None))

        return self._finalize_chunks(chunks)

    def _extract_segments(
        self, root, source: bytes  # type: ignore[no-untyped-def]
    ) -> list[tuple[str, str]]:
        """Extract named segments from top-level AST nodes."""
        segments: list[tuple[str, str]] = []
        between_start = 0

        for child in root.children:
            if child.type in DEFINITION_TYPES:
                # Capture any code between definitions
                if child.start_byte > between_start:
                    between_text = source[between_start : child.start_byte].decode(
                        "utf-8"
                    ).strip()
                    if between_text:
                        segments.append(("", between_text))

                name_node = child.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else ""
                text = source[child.start_byte : child.end_byte].decode("utf-8")
                segments.append((name, text))
                between_start = child.end_byte

        # Capture trailing code
        if between_start < len(source):
            trailing = source[between_start:].decode("utf-8").strip()
            if trailing:
                segments.append(("", trailing))

        return segments

    def _fallback_chunk(
        self, doc: RawDocument, chunk_size: int, chunk_overlap: int
    ) -> list[Chunk]:
        from .default import DefaultChunker

        return DefaultChunker().chunk(doc, chunk_size, chunk_overlap)
