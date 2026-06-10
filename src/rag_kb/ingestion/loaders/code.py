"""Code loader with tree-sitter AST extraction."""

from __future__ import annotations

from pathlib import Path

from rag_kb.models.document import RawDocument

from .base import DocumentLoader

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}


def _extract_definitions(tree, source_bytes: bytes) -> tuple[list[str], list[str]]:
    """Walk AST to find function and class names."""
    functions: list[str] = []
    classes: list[str] = []

    def walk(node):  # type: ignore[no-untyped-def]
        ntype = node.type
        if ntype in (
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "arrow_function",
        ):
            name_node = node.child_by_field_name("name")
            if name_node:
                functions.append(name_node.text.decode("utf-8"))
        elif ntype in ("class_definition", "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                classes.append(name_node.text.decode("utf-8"))
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return functions, classes


class CodeLoader(DocumentLoader):
    supported_extensions = list(EXTENSION_TO_LANGUAGE.keys())

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        content = path.read_text(encoding="utf-8")
        ext = path.suffix.lower()
        language = EXTENSION_TO_LANGUAGE.get(ext, "")
        functions: list[str] = []
        classes: list[str] = []

        if language:
            try:
                from tree_sitter_language_pack import get_parser

                parser = get_parser(language)
                tree = parser.parse(content.encode("utf-8"))
                functions, classes = _extract_definitions(tree, content.encode("utf-8"))
            except Exception:
                pass

        return RawDocument(
            content=content,
            metadata={
                "language": language or ext.lstrip("."),
                "functions": functions,
                "classes": classes,
            },
            file_path=str(path.resolve()),
            file_type=ext.lstrip("."),
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
