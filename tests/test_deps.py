"""Tests for FastAPI dependency functions in api/deps.py."""

from unittest.mock import MagicMock

from rag_kb.api.deps import get_db, get_ollama, get_qdrant, get_start_time


def _make_request(**state_attrs):
    """Create a mock Request with app.state attributes."""
    request = MagicMock()
    for key, value in state_attrs.items():
        setattr(request.app.state, key, value)
    return request


def test_get_db_returns_state_db():
    sentinel = object()
    request = _make_request(db=sentinel)
    assert get_db(request) is sentinel


def test_get_ollama_returns_state_ollama():
    sentinel = object()
    request = _make_request(ollama=sentinel)
    assert get_ollama(request) is sentinel


def test_get_qdrant_returns_state_qdrant():
    sentinel = object()
    request = _make_request(qdrant=sentinel)
    assert get_qdrant(request) is sentinel


def test_get_start_time_returns_state_start_time():
    request = _make_request(start_time=12345.0)
    assert get_start_time(request) == 12345.0
