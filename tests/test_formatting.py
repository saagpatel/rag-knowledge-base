"""Tests for CLI formatting helpers in cli/formatting.py."""

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from rag_kb.cli import formatting


def test_print_error_outputs_to_stderr():
    buf = StringIO()
    fake_console = Console(file=buf, force_terminal=False)
    orig = formatting.error_console
    formatting.error_console = fake_console
    try:
        formatting.print_error("boom")
    finally:
        formatting.error_console = orig
    output = buf.getvalue()
    assert "Error:" in output
    assert "boom" in output


def test_print_success_has_checkmark():
    buf = StringIO()
    fake_console = Console(file=buf, force_terminal=False)
    orig = formatting.console
    formatting.console = fake_console
    try:
        formatting.print_success("done")
    finally:
        formatting.console = orig
    output = buf.getvalue()
    assert "2713" in output  # checkmark rendered as \u2713 or ✓
    assert "done" in output


def test_print_search_results_empty():
    buf = StringIO()
    fake_console = Console(file=buf, force_terminal=False)
    orig = formatting.console
    formatting.console = fake_console
    try:
        mock_response = MagicMock()
        mock_response.results = []
        formatting.print_search_results(mock_response)
    finally:
        formatting.console = orig
    output = buf.getvalue()
    assert "0 results" in output


def test_print_collections_table_empty():
    buf = StringIO()
    fake_console = Console(file=buf, force_terminal=False)
    orig = formatting.console
    formatting.console = fake_console
    try:
        formatting.print_collections_table([])
    finally:
        formatting.console = orig
    output = buf.getvalue()
    assert "No collections" in output


def test_print_batch_summary_counts():
    from rag_kb.models.schema import DocumentStatus

    buf = StringIO()
    fake_console = Console(file=buf, force_terminal=False)
    orig = formatting.console
    formatting.console = fake_console

    mock_result = MagicMock()
    mock_result.total_files = 3
    mock_result.processed = 2
    mock_result.failed = 1
    mock_result.skipped = 0

    r1 = MagicMock()
    r1.status = DocumentStatus.COMPLETED
    r1.chunk_count = 5
    r1.file_path = "a.md"

    r2 = MagicMock()
    r2.status = DocumentStatus.COMPLETED
    r2.chunk_count = 0
    r2.file_path = "b.md"

    r3 = MagicMock()
    r3.status = DocumentStatus.FAILED
    r3.chunk_count = 0
    r3.file_path = "c.md"

    mock_result.results = [r1, r2, r3]

    try:
        formatting.print_batch_summary(mock_result)
    finally:
        formatting.console = orig

    output = buf.getvalue()
    assert "3" in output  # total
    assert "2 processed" in output
    assert "1 failed" in output
