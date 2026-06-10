"""BM25 vocabulary persistence — save/load per collection."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rag_kb.core.errors import BM25StoreError

from .bm25 import BM25Vectorizer

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_name(name: str) -> str:
    """Sanitize collection name for safe filesystem use."""
    name = name.replace("..", "").replace("/", "_").replace("\\", "_")
    return _SAFE_NAME_RE.sub("_", name)


def save_bm25(
    vectorizer: BM25Vectorizer,
    collection_name: str,
    data_dir: Path | str = Path("data/bm25"),
) -> Path:
    """Serialize BM25 vocabulary to a JSON file.

    Returns the path to the written file.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_name(collection_name)
    file_path = data_dir / f"{safe_name}.json"

    data = {
        "vocab": vectorizer.vocab,
        "idf": vectorizer.idf,
        "avg_doc_len": vectorizer.avg_doc_len,
        "k1": vectorizer.k1,
        "b": vectorizer.b,
    }

    file_path.write_text(json.dumps(data))
    logger.debug("Saved BM25 vocabulary for %s to %s", collection_name, file_path)
    return file_path


def load_bm25(
    collection_name: str,
    data_dir: Path | str = Path("data/bm25"),
) -> BM25Vectorizer:
    """Load a BM25 vocabulary from disk.

    Raises ``BM25StoreError`` if the file is missing or corrupt.
    """
    data_dir = Path(data_dir)
    safe_name = _sanitize_name(collection_name)
    file_path = data_dir / f"{safe_name}.json"

    if not file_path.exists():
        raise BM25StoreError(
            f"BM25 vocabulary not found for collection '{collection_name}': {file_path}"
        )

    try:
        data = json.loads(file_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise BM25StoreError(
            f"Corrupt BM25 vocabulary for collection '{collection_name}': {exc}",
            cause=exc,
        ) from exc

    try:
        return BM25Vectorizer(
            vocab=data["vocab"],
            idf=data["idf"],
            avg_doc_len=data["avg_doc_len"],
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )
    except (KeyError, TypeError) as exc:
        raise BM25StoreError(
            f"Invalid BM25 data for collection '{collection_name}': {exc}",
            cause=exc,
        ) from exc


def bm25_exists(
    collection_name: str,
    data_dir: Path | str = Path("data/bm25"),
) -> bool:
    """Check if a BM25 vocabulary file exists for the collection."""
    data_dir = Path(data_dir)
    safe_name = _sanitize_name(collection_name)
    return (data_dir / f"{safe_name}.json").exists()
