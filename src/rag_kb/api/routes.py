"""API route handlers — 16 endpoints on a single APIRouter."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from rag_kb.core.errors import CollectionNotFoundError, DocumentNotFoundError
from rag_kb.generation.engine import GenerationEngine
from rag_kb.generation.prompt import extract_sources
from rag_kb.ingestion.orchestrator import (
    BatchIngestionResult,
    IngestionResult,
    _SUPPORTED_EXTENSIONS,
    _ensure_collection_exists,
    ingest_directory,
    ingest_file,
)
from rag_kb.models.schema import Interface, QueryType, SearchMode
from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest
from rag_kb.retrieval.query_log import log_query

from .deps import get_db, get_ollama, get_qdrant, get_start_time
from .schemas import (
    AskData,
    AskRequest,
    CollectionInfo,
    CreateCollectionRequest,
    DocumentInfo,
    DocumentListData,
    FileIngestResult,
    HealthData,
    IngestData,
    IngestRequest,
    JobData,
    MetricsData,
    QueryListData,
    QueryRecord,
    SearchData,
    SearchRequest,
    SearchResultItem,
    ServiceCheck,
    SourceItem,
    StatsData,
    SuccessResponse,
    UpdateCollectionRequest,
)

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --- Health ---


@router.get("/health", tags=["Health"], summary="Check service health")
async def health(
    db: aiosqlite.Connection = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama),
    qdrant: QdrantManager = Depends(get_qdrant),
    start_time: float = Depends(get_start_time),
) -> SuccessResponse[HealthData]:
    ollama_ok = False
    ollama_detail = None
    try:
        ollama_ok = await ollama.health()
        if not ollama_ok:
            ollama_detail = "Model not found or unhealthy"
    except Exception as exc:
        ollama_detail = str(exc)

    qdrant_ok = False
    qdrant_detail = None
    try:
        await qdrant.list_collections()
        qdrant_ok = True
    except Exception as exc:
        qdrant_detail = str(exc)

    sqlite_ok = False
    sqlite_detail = None
    try:
        async with db.execute("SELECT 1") as cursor:
            await cursor.fetchone()
        sqlite_ok = True
    except Exception as exc:
        sqlite_detail = str(exc)

    all_ok = ollama_ok and qdrant_ok and sqlite_ok
    overall = "healthy" if all_ok else "degraded"
    uptime = time.monotonic() - start_time

    return SuccessResponse(
        data=HealthData(
            status=overall,
            ollama=ServiceCheck(
                status="ok" if ollama_ok else "error", detail=ollama_detail
            ),
            qdrant=ServiceCheck(
                status="ok" if qdrant_ok else "error", detail=qdrant_detail
            ),
            sqlite=ServiceCheck(
                status="ok" if sqlite_ok else "error", detail=sqlite_detail
            ),
            uptime_seconds=round(uptime, 2),
            version="0.1.0",
        )
    )


# --- Ingest ---


@router.post("/ingest", tags=["Ingest"], summary="Ingest files into knowledge base", response_model=None)
async def ingest(
    body: IngestRequest,
    request: Request,
    db: aiosqlite.Connection = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[IngestData] | SuccessResponse[JobData]:
    logger.info("Ingest request: path=%s collection=%s", body.path, body.collection)
    p = Path(body.path)
    if not p.exists():
        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "error": {
                    "code": "FILE_NOT_FOUND",
                    "message": f"Path does not exist: {body.path}",
                    "statusCode": 422,
                },
            },
        )

    # Resolve chunk_size / chunk_overlap from collection settings or config defaults
    from rag_kb.core.config import get_config

    config = get_config()
    chunk_size = body.chunk_size
    chunk_overlap = body.chunk_overlap

    if chunk_size is None or chunk_overlap is None:
        settings = await _get_collection_settings(db, body.collection)
        if chunk_size is None:
            chunk_size = settings.get("chunk_size") or config.chunking.default_size
        if chunk_overlap is None:
            chunk_overlap = settings.get("chunk_overlap") or config.chunking.default_overlap

    if p.is_file():
        result: IngestionResult = await ingest_file(
            p,
            body.collection,
            db,
            ollama,
            qdrant,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=body.force,
        )
        data = IngestData(
            total_files=1,
            processed=1 if result.status == "completed" and result.chunk_count > 0 else 0,
            failed=1 if result.status == "failed" else 0,
            skipped=1 if result.status == "completed" and result.chunk_count == 0 else 0,
            results=[
                FileIngestResult(
                    file_path=result.file_path,
                    status=result.status,
                    chunk_count=result.chunk_count,
                    error_message=result.error_message,
                )
            ],
        )
        return SuccessResponse(data=data)

    # --- Directory: async background ingest, return 202 ---
    from cuid2 import cuid_wrapper as _cw

    job_id = _cw()()

    # Pre-create job record
    collection_id = await _ensure_collection_exists(body.collection, db, qdrant)
    now = datetime.now().isoformat()

    # Count files
    files: list[Path] = []
    if body.patterns:
        for pattern in body.patterns:
            files.extend(p.rglob(pattern))
    else:
        for ext in _SUPPORTED_EXTENSIONS:
            files.extend(p.rglob(f"*{ext}"))
    files = sorted(set(files))

    await db.execute(
        "INSERT INTO ingestion_jobs "
        "(id, collection_id, status, total_files, started_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, collection_id, "running", len(files), now, now),
    )
    await db.commit()

    async def _bg_ingest() -> None:
        try:
            await ingest_directory(
                p,
                body.collection,
                db,
                ollama,
                qdrant,
                patterns=body.patterns,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force=body.force,
                job_id=job_id,
            )
        except Exception as exc:
            logger.error("Background ingest failed: %s", exc)
            try:
                await db.execute(
                    "UPDATE ingestion_jobs SET status = ?, completed_at = ? WHERE id = ?",
                    ("failed", datetime.now().isoformat(), job_id),
                )
                await db.commit()
            except Exception:
                pass

    task = asyncio.create_task(_bg_ingest())
    bg_tasks: set[asyncio.Task[Any]] | None = getattr(request.app.state, "background_tasks", None)
    if bg_tasks is not None:
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)

    return SuccessResponse(
        data=JobData(
            job_id=job_id,
            status="running",
            total_files=len(files),
            started_at=now,
        ),
    )


# --- Search ---


@router.post("/search", tags=["Search"], summary="Search the knowledge base")
async def search(
    body: SearchRequest,
    db: aiosqlite.Connection = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[SearchData]:
    logger.info("Search request: query=%r collection=%s", body.query, body.collection)
    engine = RetrievalEngine(ollama, qdrant)
    request = RetrievalRequest(
        query=body.query,
        collection=body.collection,
        mode=SearchMode(body.mode),
        top_k=body.top_k,
        rerank=body.rerank,
        filters=body.filters,
    )
    response = await engine.search(request)

    await log_query(db, response, query_type=QueryType.SEARCH, interface=Interface.API)

    return SuccessResponse(
        data=SearchData(
            results=[
                SearchResultItem(
                    id=r.id,
                    score=r.score,
                    content=r.content,
                    file_path=r.file_path,
                    file_type=r.file_type,
                    chunk_index=r.chunk_index,
                    total_chunks=r.total_chunks,
                    reranked=r.reranked,
                )
                for r in response.results
            ],
            total=response.total,
            query=response.query,
            collection=response.collection,
            mode=response.mode,
            latency_ms=round(response.latency_ms, 2),
        )
    )


# --- Ask ---


@router.post("/ask", tags=["Ask"], summary="Ask a question with AI-generated answer", response_model=None)
async def ask(
    body: AskRequest,
    db: aiosqlite.Connection = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[AskData] | StreamingResponse:
    logger.info("Ask request: query=%r collection=%s", body.query, body.collection)
    # Retrieve context
    retrieval = RetrievalEngine(ollama, qdrant)
    request = RetrievalRequest(
        query=body.query,
        collection=body.collection,
        mode=SearchMode(body.mode),
        top_k=body.top_k,
    )
    retrieval_response = await retrieval.search(request)

    await log_query(db, retrieval_response, query_type=QueryType.QA, interface=Interface.API)

    # Generate answer
    gen = GenerationEngine(ollama)

    if body.stream:
        stream_gen = await gen.answer(
            body.query, retrieval_response.results, model=body.model, stream=True
        )

        async def token_generator():  # type: ignore[no-untyped-def]
            async for token in stream_gen:  # type: ignore[union-attr]
                yield f"data: {token}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    result = await gen.answer(body.query, retrieval_response.results, model=body.model)
    sources_raw = extract_sources(retrieval_response.results)

    return SuccessResponse(
        data=AskData(
            answer=result.answer,  # type: ignore[union-attr]
            sources=[
                SourceItem(
                    file_path=str(s["file_path"]),
                    score=float(s["score"]),  # type: ignore[arg-type]
                    chunk_index=int(s["chunk_index"]),  # type: ignore[arg-type]
                    total_chunks=int(s["total_chunks"]),  # type: ignore[arg-type]
                    file_type=str(s["file_type"]),
                )
                for s in sources_raw
            ],
            query=body.query,
            model=result.model,  # type: ignore[union-attr]
            latency_ms=round(result.latency_ms, 2),  # type: ignore[union-attr]
            context_chunks_used=result.context_chunks_used,  # type: ignore[union-attr]
        )
    )


# --- Collections ---


async def _get_collection_settings(db: aiosqlite.Connection, name: str) -> dict[str, Any]:
    """Get collection settings from SQLite."""
    try:
        async with db.execute(
            "SELECT description, chunk_size, chunk_overlap, embedding_model "
            "FROM collections WHERE name = ?",
            (name,),
        ) as cursor:
            row = await cursor.fetchone()
    except Exception:
        return {}
    if not row or len(row) < 4:
        return {}
    return {
        "description": row[0] if row[0] else None,
        "chunk_size": row[1] if row[1] else None,
        "chunk_overlap": row[2] if row[2] else None,
        "embedding_model": row[3] if row[3] else None,
    }


@router.get("/collections", tags=["Collections"], summary="List all collections")
async def list_collections(
    db: aiosqlite.Connection = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[list[CollectionInfo]]:
    names = await qdrant.list_collections()
    collections = []
    for name in names:
        info = await qdrant.get_collection_info(name)
        settings = await _get_collection_settings(db, name)
        collections.append(
            CollectionInfo(
                name=info["name"],
                points_count=info["points_count"],
                vectors_count=info["vectors_count"],
                status=info["status"],
                **settings,
            )
        )
    return SuccessResponse(data=collections)


@router.get("/collections/{name}", tags=["Collections"], summary="Get collection details")
async def get_collection(
    name: str,
    db: aiosqlite.Connection = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[CollectionInfo]:
    if not await qdrant.collection_exists(name):
        raise CollectionNotFoundError(f"Collection '{name}' does not exist")

    info = await qdrant.get_collection_info(name)
    settings = await _get_collection_settings(db, name)
    return SuccessResponse(
        data=CollectionInfo(
            name=info["name"],
            points_count=info["points_count"],
            vectors_count=info["vectors_count"],
            status=info["status"],
            **settings,
        )
    )


@router.post("/collections", tags=["Collections"], summary="Create a collection", status_code=201)
async def create_collection(
    body: CreateCollectionRequest,
    db: aiosqlite.Connection = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[CollectionInfo]:
    await qdrant.create_collection(body.name)

    # Store optional settings in SQLite
    from cuid2 import cuid_wrapper

    _cuid = cuid_wrapper()
    async with db.execute("SELECT id FROM collections WHERE name = ?", (body.name,)) as cursor:
        row = await cursor.fetchone()
    if not row:
        collection_id = _cuid()
        await db.execute(
            "INSERT INTO collections (id, name, description, chunk_size, chunk_overlap, embedding_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (collection_id, body.name, body.description or "", body.chunk_size or 512,
             body.chunk_overlap or 50, body.embedding_model or "nomic-embed-text"),
        )
        await db.commit()
    else:
        # Update if settings provided
        updates = []
        params: list[Any] = []
        if body.description is not None:
            updates.append("description = ?")
            params.append(body.description)
        if body.chunk_size is not None:
            updates.append("chunk_size = ?")
            params.append(body.chunk_size)
        if body.chunk_overlap is not None:
            updates.append("chunk_overlap = ?")
            params.append(body.chunk_overlap)
        if body.embedding_model is not None:
            updates.append("embedding_model = ?")
            params.append(body.embedding_model)
        if updates:
            sql = f"UPDATE collections SET {', '.join(updates)} WHERE id = ?"
            params.append(row[0])
            await db.execute(sql, params)
            await db.commit()

    info = await qdrant.get_collection_info(body.name)
    settings = await _get_collection_settings(db, body.name)
    return SuccessResponse(
        data=CollectionInfo(
            name=info["name"],
            points_count=info["points_count"],
            vectors_count=info["vectors_count"],
            status=info["status"],
            **settings,
        )
    )


@router.put("/collections/{name}", tags=["Collections"], summary="Update collection settings")
async def update_collection(
    name: str,
    body: UpdateCollectionRequest,
    db: aiosqlite.Connection = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[CollectionInfo]:
    if not await qdrant.collection_exists(name):
        raise CollectionNotFoundError(f"Collection '{name}' does not exist")

    # Get or create SQLite collection record
    async with db.execute("SELECT id FROM collections WHERE name = ?", (name,)) as cursor:
        row = await cursor.fetchone()

    if row:
        collection_id = row[0]
    else:
        from cuid2 import cuid_wrapper

        _cuid = cuid_wrapper()
        collection_id = _cuid()
        await db.execute("INSERT INTO collections (id, name) VALUES (?, ?)", (collection_id, name))

    # Update fields
    updates = []
    params: list[Any] = []
    if body.description is not None:
        updates.append("description = ?")
        params.append(body.description)
    if body.chunk_size is not None:
        updates.append("chunk_size = ?")
        params.append(body.chunk_size)
    if body.chunk_overlap is not None:
        updates.append("chunk_overlap = ?")
        params.append(body.chunk_overlap)
    if body.embedding_model is not None:
        updates.append("embedding_model = ?")
        params.append(body.embedding_model)

    if updates:
        sql = f"UPDATE collections SET {', '.join(updates)} WHERE id = ?"
        params.append(collection_id)
        await db.execute(sql, params)
        await db.commit()

    # Return enriched info
    info = await qdrant.get_collection_info(name)
    settings = await _get_collection_settings(db, name)
    return SuccessResponse(
        data=CollectionInfo(
            name=info["name"],
            points_count=info["points_count"],
            vectors_count=info["vectors_count"],
            status=info["status"],
            **settings,
        )
    )


@router.delete("/collections/{name}", tags=["Collections"], summary="Delete a collection")
async def delete_collection(
    name: str,
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[dict[str, str]]:
    if not await qdrant.collection_exists(name):
        raise CollectionNotFoundError(f"Collection '{name}' does not exist")

    await qdrant.delete_collection(name)
    return SuccessResponse(data={"message": f"Collection '{name}' deleted"})


# --- Documents ---


@router.get("/documents", tags=["Documents"], summary="List documents")
async def list_documents(
    db: aiosqlite.Connection = Depends(get_db),
    collection: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> SuccessResponse[DocumentListData]:
    conditions: list[str] = []
    params: list[str | int] = []

    collection_id = None
    if collection:
        async with db.execute(
            "SELECT id FROM collections WHERE name = ?", (collection,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            raise CollectionNotFoundError(f"Collection '{collection}' does not exist")
        collection_id = row[0]  # type: ignore[index]
        conditions.append("d.collection_id = ?")
        params.append(collection_id)

    if status:
        conditions.append("d.status = ?")
        params.append(status)

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    async with db.execute(
        f"SELECT COUNT(*) FROM documents d{where}", params  # noqa: S608
    ) as cursor:
        count_row = await cursor.fetchone()
    total = count_row[0] if count_row else 0  # type: ignore[index]

    query_sql = (
        f"SELECT d.id, d.collection_id, d.filename, d.file_path, d.file_type, "  # noqa: S608
        f"d.file_hash, d.chunk_count, d.status, d.error_message, "
        f"d.created_at, d.updated_at "
        f"FROM documents d{where} ORDER BY d.created_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    documents = []
    async with db.execute(query_sql, params) as cursor:
        async for row in cursor:
            documents.append(
                DocumentInfo(
                    id=row[0],
                    collection_id=row[1],
                    filename=row[2],
                    file_path=row[3],
                    file_type=row[4],
                    file_hash=row[5],
                    chunk_count=row[6],
                    status=row[7],
                    error_message=row[8],
                    created_at=row[9],
                    updated_at=row[10],
                )
            )

    return SuccessResponse(
        data=DocumentListData(documents=documents, total=total, collection=collection)
    )


@router.get("/documents/{doc_id}", tags=["Documents"], summary="Get document details")
async def get_document(
    doc_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> SuccessResponse[DocumentInfo]:
    async with db.execute(
        "SELECT id, collection_id, filename, file_path, file_type, "
        "file_hash, chunk_count, status, error_message, "
        "created_at, updated_at FROM documents WHERE id = ?",
        (doc_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if not row:
        raise DocumentNotFoundError(f"Document '{doc_id}' does not exist")

    return SuccessResponse(
        data=DocumentInfo(
            id=row[0],
            collection_id=row[1],
            filename=row[2],
            file_path=row[3],
            file_type=row[4],
            file_hash=row[5],
            chunk_count=row[6],
            status=row[7],
            error_message=row[8],
            created_at=row[9],
            updated_at=row[10],
        )
    )


@router.delete("/documents/{doc_id}", tags=["Documents"], summary="Delete a document")
async def delete_document(
    doc_id: str,
    db: aiosqlite.Connection = Depends(get_db),
    qdrant: QdrantManager = Depends(get_qdrant),
) -> SuccessResponse[dict[str, str]]:
    # Fetch document
    async with db.execute(
        "SELECT d.id, d.collection_id, d.file_path, c.name "
        "FROM documents d JOIN collections c ON d.collection_id = c.id "
        "WHERE d.id = ?",
        (doc_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if not row:
        raise DocumentNotFoundError(f"Document '{doc_id}' does not exist")

    file_path = row[2]
    collection_name = row[3]

    # Delete points from Qdrant
    await qdrant.delete_points_by_filter(collection_name, {"file_path": file_path})

    # Delete document record
    await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    await db.commit()

    return SuccessResponse(data={"message": f"Document '{doc_id}' deleted"})


# --- Analytics ---


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile from sorted data. Returns 0.0 for empty lists."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])


@router.get("/stats", tags=["Analytics"], summary="Get aggregate statistics")
async def stats(
    db: aiosqlite.Connection = Depends(get_db),
    days: int = Query(30, ge=1, le=365),
) -> SuccessResponse[StatsData]:
    # Total queries and avg latency
    async with db.execute(
        "SELECT COUNT(*), COALESCE(AVG(latency_ms), 0) FROM queries"
    ) as cursor:
        row = await cursor.fetchone()
    total_queries = row[0] if row else 0  # type: ignore[index]
    avg_latency = round(row[1], 2) if row else 0.0  # type: ignore[index]

    # Latency percentiles
    async with db.execute(
        "SELECT latency_ms FROM queries ORDER BY latency_ms"
    ) as cursor:
        latencies = [r[0] async for r in cursor]

    p50 = round(_percentile(latencies, 50), 2)
    p95 = round(_percentile(latencies, 95), 2)
    p99 = round(_percentile(latencies, 99), 2)

    # Queries by interface
    queries_by_interface: dict[str, int] = {}
    async with db.execute(
        "SELECT interface, COUNT(*) FROM queries GROUP BY interface"
    ) as cursor:
        async for row in cursor:
            queries_by_interface[row[0]] = row[1]

    # Queries by type
    queries_by_type: dict[str, int] = {}
    async with db.execute(
        "SELECT query_type, COUNT(*) FROM queries GROUP BY query_type"
    ) as cursor:
        async for row in cursor:
            queries_by_type[row[0]] = row[1]

    # Top collections
    top_collections: list[dict[str, str | int]] = []
    async with db.execute(
        "SELECT c.name, COUNT(q.id) FROM queries q "
        "LEFT JOIN collections c ON q.collection_id = c.id "
        "GROUP BY q.collection_id ORDER BY COUNT(q.id) DESC LIMIT 10"
    ) as cursor:
        async for row in cursor:
            top_collections.append({
                "name": row[0] or "unknown",
                "count": row[1],
            })

    return SuccessResponse(
        data=StatsData(
            total_queries=total_queries,
            avg_latency_ms=avg_latency,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            queries_by_interface=queries_by_interface,
            queries_by_type=queries_by_type,
            top_collections=top_collections,
            period_days=days,
        )
    )


@router.get("/queries", tags=["Analytics"], summary="Get query history")
async def query_history(
    db: aiosqlite.Connection = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    collection: str | None = Query(None),
    interface: str | None = Query(None),
    query_type: str | None = Query(None),
) -> SuccessResponse[QueryListData]:
    conditions: list[str] = []
    params: list[str | int] = []

    if collection:
        async with db.execute(
            "SELECT id FROM collections WHERE name = ?", (collection,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            conditions.append("q.collection_id = ?")
            params.append(row[0])  # type: ignore[index]

    if interface:
        conditions.append("q.interface = ?")
        params.append(interface)

    if query_type:
        conditions.append("q.query_type = ?")
        params.append(query_type)

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""

    async with db.execute(
        f"SELECT COUNT(*) FROM queries q{where}", params  # noqa: S608
    ) as cursor:
        count_row = await cursor.fetchone()
    total = count_row[0] if count_row else 0  # type: ignore[index]

    query_sql = (
        f"SELECT q.id, q.query_text, q.query_type, q.search_mode, "  # noqa: S608
        f"q.result_count, q.latency_ms, q.interface, q.created_at "
        f"FROM queries q{where} ORDER BY q.created_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    queries = []
    async with db.execute(query_sql, params) as cursor:
        async for row in cursor:
            queries.append(
                QueryRecord(
                    id=row[0],
                    query_text=row[1],
                    query_type=row[2],
                    search_mode=row[3],
                    result_count=row[4],
                    latency_ms=row[5],
                    interface=row[6],
                    created_at=row[7],
                )
            )

    return SuccessResponse(data=QueryListData(queries=queries, total=total))


# --- Jobs ---


@router.get("/jobs/{job_id}", tags=["Jobs"], summary="Get job status")
async def get_job(
    job_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> SuccessResponse[JobData]:
    async with db.execute(
        "SELECT id, status, total_files, processed_files, failed_files, "
        "started_at, completed_at FROM ingestion_jobs WHERE id = ?",
        (job_id,),
    ) as cursor:
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return SuccessResponse(
        data=JobData(
            job_id=row[0],
            status=row[1],
            total_files=row[2],
            processed_files=row[3] or 0,
            failed_files=row[4] or 0,
            started_at=row[5],
            completed_at=row[6],
        )
    )


# --- Metrics ---


@router.get("/metrics", tags=["Analytics"], summary="Get performance metrics")
async def metrics(
    request: Request,
    db: aiosqlite.Connection = Depends(get_db),
    ollama: OllamaClient = Depends(get_ollama),
) -> SuccessResponse[MetricsData]:
    # Query latencies
    async with db.execute(
        "SELECT latency_ms FROM queries ORDER BY latency_ms"
    ) as cursor:
        latencies = [r[0] async for r in cursor]

    total = len(latencies)
    p50 = round(_percentile(latencies, 50), 2)
    p95 = round(_percentile(latencies, 95), 2)
    p99 = round(_percentile(latencies, 99), 2)

    # Cache stats
    cache = getattr(ollama, "_cache", None)
    cache_hit_rate = cache.hit_rate if cache else 0.0
    cache_size = cache.size if cache else 0

    # Active jobs
    async with db.execute(
        "SELECT COUNT(*) FROM ingestion_jobs WHERE status = 'running'"
    ) as cursor:
        row = await cursor.fetchone()
    active_jobs = row[0] if row else 0

    return SuccessResponse(
        data=MetricsData(
            total_queries=total,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
            cache_hit_rate=round(cache_hit_rate, 4),
            cache_size=cache_size,
            active_jobs=active_jobs,
        )
    )
