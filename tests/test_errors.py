"""Tests for error mapping in api/errors.py."""

from rag_kb.api.errors import _resolve_rag_error
from rag_kb.core.errors import (
    DocumentNotFoundError,
    OllamaConnectionError,
    RAGError,
)


def test_ollama_connection_maps_to_503():
    status, code = _resolve_rag_error(OllamaConnectionError("cannot connect"))
    assert status == 503
    assert code == "OLLAMA_UNAVAILABLE"


def test_document_not_found_maps_to_404():
    status, code = _resolve_rag_error(DocumentNotFoundError("doc-1"))
    assert status == 404
    assert code == "DOCUMENT_NOT_FOUND"


def test_base_rag_error_maps_to_500():
    status, code = _resolve_rag_error(RAGError("generic"))
    assert status == 500
    assert code == "INTERNAL_ERROR"
