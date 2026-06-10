"""Tests for the JSONFormatter, sensitive redaction, and request context filter."""

from __future__ import annotations

import json
import logging

from rag_kb.core.logging import JSONFormatter, RequestContextFilter


def _make_record(message: str = "test message", **extras) -> logging.LogRecord:
    """Create a log record with optional extra fields."""
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extras.items():
        setattr(record, key, value)
    return record


class TestJSONFormatter:
    def test_basic_format(self):
        """Basic log record produces valid JSON with required fields."""
        fmt = JSONFormatter()
        record = _make_record("hello world")
        output = fmt.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "hello world"
        assert "timestamp" in data

    def test_with_extras(self):
        """Extra fields (method, path, status_code, etc.) are included."""
        fmt = JSONFormatter()
        record = _make_record(
            "GET /api/health 200",
            method="GET",
            path="/api/health",
            status_code=200,
            latency_ms=42.5,
            request_id="abc-123",
        )
        output = fmt.format(record)
        data = json.loads(output)

        assert data["method"] == "GET"
        assert data["path"] == "/api/health"
        assert data["status_code"] == 200
        assert data["latency_ms"] == 42.5
        assert data["request_id"] == "abc-123"

    def test_with_exception(self):
        """Exception info is included when present."""
        fmt = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="something failed",
            args=(),
            exc_info=exc_info,
        )
        output = fmt.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "test error" in data["exception"]


class TestSensitiveRedaction:
    def test_password_redacted(self):
        """Password field is redacted in JSON output."""
        fmt = JSONFormatter()
        record = _make_record("login attempt", password="s3cret")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["password"] == "[REDACTED]"

    def test_api_key_redacted(self):
        """api_key field is redacted in JSON output."""
        fmt = JSONFormatter()
        record = _make_record("config loaded", api_key="sk-abc123")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["api_key"] == "[REDACTED]"

    def test_authorization_redacted(self):
        """authorization field is redacted in JSON output."""
        fmt = JSONFormatter()
        record = _make_record("request received", authorization="Bearer tok123")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["authorization"] == "[REDACTED]"

    def test_non_sensitive_not_redacted(self):
        """Non-sensitive fields are preserved as-is."""
        fmt = JSONFormatter()
        record = _make_record("request", method="GET", path="/api/health")
        output = fmt.format(record)
        data = json.loads(output)
        assert data["method"] == "GET"
        assert data["path"] == "/api/health"


class TestRequestContextFilter:
    def test_adds_request_id_from_contextvars(self):
        """Filter injects request_id from contextvars into record."""
        from rag_kb.api.middleware import request_id_var

        token = request_id_var.set("test-req-id-123")
        try:
            f = RequestContextFilter()
            record = _make_record("test")
            result = f.filter(record)
            assert result is True
            assert record.request_id == "test-req-id-123"
        finally:
            request_id_var.reset(token)

    def test_default_empty_string_when_no_context(self):
        """Filter sets empty string when no request context is active."""
        f = RequestContextFilter()
        record = _make_record("test")
        result = f.filter(record)
        assert result is True
        assert record.request_id == ""
