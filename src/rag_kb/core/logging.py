"""Logging setup — structured JSON output via stdlib logging."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_kb.core.config import LoggingConfig

_SENSITIVE_KEYS = {"password", "token", "secret", "authorization", "api_key"}

_EXTRA_KEYS = ("method", "path", "status_code", "latency_ms", "request_id")


class RequestContextFilter(logging.Filter):
    """Inject request_id from contextvars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from rag_kb.api.middleware import request_id_var

            record.request_id = request_id_var.get("")
        except (ImportError, LookupError):
            if not hasattr(record, "request_id"):
                record.request_id = ""
        return True


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines, including known extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (*_EXTRA_KEYS, *_SENSITIVE_KEYS):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Redact sensitive fields
        for key in list(log_entry.keys()):
            if key in _SENSITIVE_KEYS:
                log_entry[key] = "[REDACTED]"
        return json.dumps(log_entry)


def setup_logging(config: LoggingConfig) -> None:
    """Configure Python logging with structured JSON output."""
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    formatter: logging.Formatter
    if config.format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    ctx_filter = RequestContextFilter()

    # File handler
    file_handler = logging.FileHandler(config.file)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ctx_filter)
    root.addHandler(file_handler)

    # Console handler (plain text for readability)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    console_handler.addFilter(ctx_filter)
    root.addHandler(console_handler)
