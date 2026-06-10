"""Loaders for structured data: JSON, YAML, CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from rag_kb.core.errors import LoaderError
from rag_kb.models.document import RawDocument

from .base import DocumentLoader


class JSONLoader(DocumentLoader):
    supported_extensions = [".json"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LoaderError(f"Invalid JSON: {path}", cause=e)

        content = json.dumps(data, indent=2)
        data_type = "array" if isinstance(data, list) else "object"
        key_count = len(data) if isinstance(data, (dict, list)) else 0

        return RawDocument(
            content=content,
            metadata={"key_count": key_count, "data_type": data_type},
            file_path=str(path.resolve()),
            file_type="json",
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )


class YAMLLoader(DocumentLoader):
    supported_extensions = [".yaml", ".yml"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise LoaderError(f"Invalid YAML: {path}", cause=e)

        content = json.dumps(data, indent=2, default=str)
        data_type = "array" if isinstance(data, list) else "object"
        key_count = len(data) if isinstance(data, (dict, list)) else 0

        return RawDocument(
            content=content,
            metadata={"key_count": key_count, "data_type": data_type},
            file_path=str(path.resolve()),
            file_type=path.suffix.lstrip(".").lower(),
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )


class CSVLoader(DocumentLoader):
    supported_extensions = [".csv"]

    def load(self, path: Path) -> RawDocument:
        self._validate_path(path)
        raw = path.read_text(encoding="utf-8")
        reader = csv.DictReader(raw.splitlines())
        column_names = reader.fieldnames or []

        rows: list[str] = []
        for i, row in enumerate(reader, 1):
            pairs = ", ".join(f"{k}={v}" for k, v in row.items())
            rows.append(f"Row {i}: {pairs}")

        return RawDocument(
            content="\n".join(rows),
            metadata={
                "column_names": list(column_names),
                "row_count": len(rows),
            },
            file_path=str(path.resolve()),
            file_type="csv",
            file_hash=self._compute_hash(path),
            file_size=self._file_size(path),
        )
