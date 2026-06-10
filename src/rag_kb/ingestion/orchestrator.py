"""Ingestion orchestrator — end-to-end file → Qdrant + SQLite tracking."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cuid2 import cuid_wrapper

from rag_kb.core.errors import PipelineError
from rag_kb.models.schema import DocumentStatus, JobStatus

from .bm25 import BM25Vectorizer
from .bm25_store import save_bm25
from .chunkers import get_chunker
from .loaders import get_loader
from .pipeline import embed_chunks

if TYPE_CHECKING:
    import aiosqlite

    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)

_cuid = cuid_wrapper()

# File extensions supported by the loader system
_SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".sh", ".bash",
    ".css", ".html", ".htm", ".json", ".yaml", ".yml", ".csv", ".pdf",
}


@dataclass
class IngestionResult:
    """Result of ingesting a single file."""

    file_path: str
    status: DocumentStatus
    chunk_count: int = 0
    error_message: str | None = None


@dataclass
class BatchIngestionResult:
    """Result of ingesting a directory."""

    total_files: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[IngestionResult] = field(default_factory=list)


# --- Private helpers ---


async def _ensure_collection_exists(
    name: str,
    db: aiosqlite.Connection,
    qdrant: QdrantManager,
) -> str:
    """Ensure collection exists in both SQLite and Qdrant. Return collection ID."""
    async with db.execute(
        "SELECT id FROM collections WHERE name = ?", (name,)
    ) as cursor:
        row = await cursor.fetchone()

    if row:
        collection_id = row[0]  # type: ignore[index]
    else:
        collection_id = _cuid()
        await db.execute(
            "INSERT INTO collections (id, name) VALUES (?, ?)",
            (collection_id, name),
        )
        await db.commit()

    # Idempotent — no-op if already exists
    await qdrant.create_collection(name)
    return collection_id


async def _get_existing_doc_hash(
    db: aiosqlite.Connection,
    collection_id: str,
    file_path: str,
) -> str | None:
    """Return file_hash of an existing completed document, or None."""
    async with db.execute(
        "SELECT file_hash FROM documents WHERE collection_id = ? AND file_path = ? AND status = ?",
        (collection_id, file_path, DocumentStatus.COMPLETED),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None  # type: ignore[index]


async def _upsert_document_record(
    db: aiosqlite.Connection,
    collection_id: str,
    file_path: str,
    filename: str,
    file_type: str,
    file_hash: str,
    file_size: int,
) -> str:
    """Insert or update a document record, set status to PROCESSING. Return doc_id."""
    async with db.execute(
        "SELECT id FROM documents WHERE collection_id = ? AND file_path = ?",
        (collection_id, file_path),
    ) as cursor:
        row = await cursor.fetchone()

    now = datetime.now().isoformat()

    if row:
        doc_id = row[0]  # type: ignore[index]
        await db.execute(
            "UPDATE documents SET file_hash = ?, file_size = ?, status = ?, "
            "error_message = NULL, updated_at = ? WHERE id = ?",
            (file_hash, file_size, DocumentStatus.PROCESSING, now, doc_id),
        )
    else:
        doc_id = _cuid()
        await db.execute(
            "INSERT INTO documents (id, collection_id, filename, file_path, file_type, "
            "file_hash, file_size, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, collection_id, filename, file_path, file_type,
             file_hash, file_size, DocumentStatus.PROCESSING, now, now),
        )
    await db.commit()
    return doc_id


async def _mark_document_completed(
    db: aiosqlite.Connection,
    doc_id: str,
    chunk_count: int,
) -> None:
    """Mark a document as completed with its chunk count."""
    await db.execute(
        "UPDATE documents SET status = ?, chunk_count = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.COMPLETED, chunk_count, datetime.now().isoformat(), doc_id),
    )
    await db.commit()


async def _mark_document_failed(
    db: aiosqlite.Connection,
    doc_id: str,
    error_message: str,
) -> None:
    """Mark a document as failed with an error message."""
    await db.execute(
        "UPDATE documents SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.FAILED, error_message, datetime.now().isoformat(), doc_id),
    )
    await db.commit()


# --- Public API ---


async def ingest_file(
    path: str | Path,
    collection_name: str,
    db: aiosqlite.Connection,
    ollama: OllamaClient,
    qdrant: QdrantManager,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    bm25: BM25Vectorizer | None = None,
    force: bool = False,
) -> IngestionResult:
    """Ingest a single file: load → chunk → embed → upsert → track.

    Returns an ``IngestionResult`` even on failure (per-file errors don't raise).
    """
    path = Path(path).resolve()
    file_path_str = str(path)

    try:
        # 1. Ensure collection exists
        collection_id = await _ensure_collection_exists(collection_name, db, qdrant)

        # 2. Load and chunk
        ext = path.suffix.lower()
        loader = get_loader(ext)
        raw_doc = loader.load(path)
        chunker = get_chunker(raw_doc.file_type)
        chunks = chunker.chunk(raw_doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 3. Skip if unchanged (unless force)
        if not force:
            existing_hash = await _get_existing_doc_hash(db, collection_id, file_path_str)
            if existing_hash == raw_doc.file_hash:
                logger.debug("Skipping unchanged file: %s", path)
                return IngestionResult(
                    file_path=file_path_str,
                    status=DocumentStatus.COMPLETED,
                    chunk_count=0,
                )

        # 3b. If force, delete old points for this file
        if force:
            try:
                await qdrant.delete_points_by_filter(
                    collection_name, {"file_path": file_path_str}
                )
            except Exception:
                logger.debug("No old points to delete for %s (OK)", path)

        # 4. Record as PROCESSING
        doc_id = await _upsert_document_record(
            db,
            collection_id,
            file_path_str,
            path.name,
            raw_doc.file_type,
            raw_doc.file_hash,
            raw_doc.file_size,
        )

        # 5. Build BM25 and persist, then embed chunks (dense + sparse)
        texts = [c.content for c in chunks]
        bm25 = bm25 or BM25Vectorizer.from_texts(texts)
        save_bm25(bm25, collection_name)
        points = await embed_chunks(chunks, ollama, bm25)

        # 6. Upsert to Qdrant
        if points:
            await qdrant.upsert_points(collection_name, points)

        # 7. Mark completed
        await _mark_document_completed(db, doc_id, len(chunks))

        logger.info("Ingested %s: %d chunks", path.name, len(chunks))
        return IngestionResult(
            file_path=file_path_str,
            status=DocumentStatus.COMPLETED,
            chunk_count=len(chunks),
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to ingest %s: %s", path, error_msg)

        # Try to mark as failed in SQLite (best-effort)
        try:
            collection_id = await _ensure_collection_exists(collection_name, db, qdrant)
            async with db.execute(
                "SELECT id FROM documents WHERE collection_id = ? AND file_path = ?",
                (collection_id, file_path_str),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                await _mark_document_failed(db, row[0], error_msg)  # type: ignore[index]
        except Exception:
            pass

        return IngestionResult(
            file_path=file_path_str,
            status=DocumentStatus.FAILED,
            error_message=error_msg,
        )


async def ingest_directory(
    dir_path: str | Path,
    collection_name: str,
    db: aiosqlite.Connection,
    ollama: OllamaClient,
    qdrant: QdrantManager,
    patterns: list[str] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    force: bool = False,
    job_id: str | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> BatchIngestionResult:
    """Ingest all supported files in a directory.

    Args:
        dir_path: Directory to scan.
        collection_name: Target Qdrant collection.
        db: SQLite connection (with migrations applied).
        ollama: Ollama client for embeddings.
        qdrant: Qdrant manager for vector storage.
        patterns: Optional glob patterns (e.g. ``["*.md", "*.txt"]``).
            If None, all supported extensions are used.
        chunk_size: Chunk size in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        force: Re-ingest even if file hash is unchanged.
        job_id: Optional pre-created job ID (from async API ingest).
            If provided, the existing job record is reused instead of creating a new one.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise PipelineError(f"Not a directory: {dir_path}")

    # Collect files
    files: list[Path] = []
    if patterns:
        for pattern in patterns:
            files.extend(dir_path.rglob(pattern))
    else:
        for ext in _SUPPORTED_EXTENSIONS:
            files.extend(dir_path.rglob(f"*{ext}"))

    # Deduplicate and sort for deterministic ordering
    files = sorted(set(files))

    # Ensure collection + create job record (unless pre-created)
    collection_id = await _ensure_collection_exists(collection_name, db, qdrant)
    if job_id is None:
        job_id = _cuid()
        now = datetime.now().isoformat()
        await db.execute(
            "INSERT INTO ingestion_jobs "
            "(id, collection_id, status, total_files, started_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, collection_id, JobStatus.RUNNING, len(files), now, now),
        )
        await db.commit()
    else:
        # Update file count on pre-created job
        await db.execute(
            "UPDATE ingestion_jobs SET total_files = ? WHERE id = ?",
            (len(files), job_id),
        )
        await db.commit()

    result = BatchIngestionResult(total_files=len(files))

    # Process sequentially (memory-bounded)
    for file_path in files:
        file_result = await ingest_file(
            file_path, collection_name, db, ollama, qdrant,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            force=force,
        )
        result.results.append(file_result)

        if progress_callback:
            progress_callback(file_path.name, str(file_path))

        if file_result.status == DocumentStatus.FAILED:
            result.failed += 1
        elif file_result.chunk_count == 0 and file_result.status == DocumentStatus.COMPLETED:
            result.skipped += 1
        else:
            result.processed += 1

        # Update job progress
        processed_so_far = result.processed + result.failed + result.skipped
        await db.execute(
            "UPDATE ingestion_jobs SET processed_files = ?, failed_files = ? WHERE id = ?",
            (processed_so_far, result.failed, job_id),
        )
        await db.commit()

    # Finalize job
    final_status = JobStatus.COMPLETED if result.failed == 0 else JobStatus.FAILED
    await db.execute(
        "UPDATE ingestion_jobs SET status = ?, completed_at = ? WHERE id = ?",
        (final_status, datetime.now().isoformat(), job_id),
    )
    await db.commit()

    logger.info(
        "Directory ingest complete: %d processed, %d failed, %d skipped",
        result.processed, result.failed, result.skipped,
    )
    return result
