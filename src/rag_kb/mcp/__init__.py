"""MCP server — FastMCP wrapper exposing 12 tools over stdio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.context import Context

from rag_kb.core.errors import (
    CollectionNotFoundError,
    OllamaConnectionError,
    QdrantConnectionError,
    QdrantError,
    RAGError,
)


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Init DB, Ollama client, and Qdrant manager; tear down on shutdown."""
    from rag_kb.core.database import init_db
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager

    db = await init_db()
    ollama = OllamaClient()
    qdrant = QdrantManager()

    yield {"db": db, "ollama": ollama, "qdrant": qdrant}

    await qdrant.close()
    await ollama.close()
    await db.close()


mcp = FastMCP("RAG Knowledge Base", lifespan=lifespan)


def _ctx_resources(ctx: Context) -> tuple[Any, Any, Any]:
    """Extract db, ollama, qdrant from lifespan context."""
    lc = ctx.lifespan_context
    return lc["db"], lc["ollama"], lc["qdrant"]


# --- Tools ---


@mcp.tool
async def search(
    query: str,
    ctx: Context,
    collection: str = "default",
    mode: str = "hybrid",
    top_k: int = 10,
    rerank: bool = False,
) -> dict[str, Any]:
    """Search the knowledge base for relevant documents.

    Args:
        query: The search query text.
        collection: Collection to search (default: "default").
        mode: Search mode — "dense", "sparse", or "hybrid" (default: "hybrid").
        top_k: Maximum number of results to return (default: 10).
        rerank: Whether to rerank results for better relevance (default: False).
    """
    db, ollama, qdrant = _ctx_resources(ctx)
    try:
        from rag_kb.models.schema import SearchMode
        from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest

        engine = RetrievalEngine(ollama, qdrant)
        request = RetrievalRequest(
            query=query,
            collection=collection,
            mode=SearchMode(mode),
            top_k=top_k,
            rerank=rerank,
        )
        response = await engine.search(request)

        return {
            "results": [
                {
                    "content": r.content,
                    "file_path": r.file_path,
                    "score": r.score,
                    "file_type": r.file_type,
                    "chunk_index": r.chunk_index,
                    "total_chunks": r.total_chunks,
                }
                for r in response.results
            ],
            "total": response.total,
            "latency_ms": round(response.latency_ms, 2),
        }
    except CollectionNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    except OllamaConnectionError as exc:
        raise ToolError("Ollama unavailable") from exc
    except QdrantConnectionError as exc:
        raise ToolError("Qdrant unavailable") from exc
    except RAGError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def ask(
    query: str,
    ctx: Context,
    collection: str = "default",
    top_k: int = 5,
    model: str | None = None,
) -> dict[str, Any]:
    """Ask a question and get an AI-generated answer with sources.

    Args:
        query: The question to answer.
        collection: Collection to search for context (default: "default").
        top_k: Number of context chunks to retrieve (default: 5).
        model: Override the generation model (default: configured model).
    """
    db, ollama, qdrant = _ctx_resources(ctx)
    try:
        from rag_kb.generation.engine import GenerationEngine
        from rag_kb.generation.prompt import extract_sources
        from rag_kb.models.schema import SearchMode
        from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest

        # Retrieve context
        retrieval = RetrievalEngine(ollama, qdrant)
        request = RetrievalRequest(
            query=query,
            collection=collection,
            mode=SearchMode.HYBRID,
            top_k=top_k,
        )
        response = await retrieval.search(request)

        # Generate answer (never streaming in MCP)
        gen = GenerationEngine(ollama)
        result = await gen.answer(query, response.results, model=model)
        sources = extract_sources(response.results)

        return {
            "answer": result.answer,
            "sources": [
                {"file_path": s["file_path"], "score": s["score"]}
                for s in sources
            ],
            "model": result.model,
            "latency_ms": round(result.latency_ms, 2),
        }
    except CollectionNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    except OllamaConnectionError as exc:
        raise ToolError("Ollama unavailable") from exc
    except RAGError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def ingest(
    path: str,
    ctx: Context,
    collection: str = "default",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Ingest a file or directory into the knowledge base.

    Args:
        path: Absolute path to a file or directory to ingest.
        collection: Target collection name (default: "default").
        chunk_size: Chunk size in tokens (default: 512).
        chunk_overlap: Overlap between chunks in tokens (default: 50).
        patterns: Glob patterns for directory ingest (e.g. ["*.md", "*.py"]).
    """
    db, ollama, qdrant = _ctx_resources(ctx)
    p = Path(path)
    if not p.exists():
        raise ToolError(f"Path does not exist: {path}")

    try:
        from rag_kb.ingestion.orchestrator import ingest_directory, ingest_file

        if p.is_file():
            result = await ingest_file(
                p, collection, db, ollama, qdrant,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            return {
                "total_files": 1,
                "processed": 1 if result.chunk_count > 0 else 0,
                "failed": 1 if result.status == "failed" else 0,
                "skipped": 1 if result.chunk_count == 0 and result.status != "failed" else 0,
                "results": [{
                    "file_path": result.file_path,
                    "status": str(result.status),
                    "chunk_count": result.chunk_count,
                }],
            }
        else:
            batch = await ingest_directory(
                p, collection, db, ollama, qdrant,
                patterns=patterns,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            )
            return {
                "total_files": batch.total_files,
                "processed": batch.processed,
                "failed": batch.failed,
                "skipped": batch.skipped,
                "results": [
                    {
                        "file_path": r.file_path,
                        "status": str(r.status),
                        "chunk_count": r.chunk_count,
                    }
                    for r in batch.results
                ],
            }
    except OllamaConnectionError as exc:
        raise ToolError("Ollama unavailable") from exc
    except RAGError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def list_collections(ctx: Context) -> list[dict[str, Any]]:
    """List all collections in the knowledge base."""
    _, _, qdrant = _ctx_resources(ctx)
    try:
        names = await qdrant.list_collections()
        result = []
        for name in names:
            info = await qdrant.get_collection_info(name)
            result.append(info)
        return result
    except QdrantConnectionError as exc:
        raise ToolError("Qdrant unavailable") from exc
    except QdrantError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def create_collection(name: str, ctx: Context) -> dict[str, Any]:
    """Create a new collection in the knowledge base.

    Args:
        name: Name for the new collection.
    """
    _, _, qdrant = _ctx_resources(ctx)
    try:
        await qdrant.create_collection(name)
        return await qdrant.get_collection_info(name)
    except QdrantConnectionError as exc:
        raise ToolError("Qdrant unavailable") from exc
    except QdrantError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def delete_collection(name: str, ctx: Context) -> dict[str, str]:
    """Delete a collection from the knowledge base.

    Args:
        name: Name of the collection to delete.
    """
    _, _, qdrant = _ctx_resources(ctx)
    try:
        if not await qdrant.collection_exists(name):
            raise ToolError(f"Collection '{name}' does not exist")
        await qdrant.delete_collection(name)
        return {"message": f"Collection '{name}' deleted"}
    except ToolError:
        raise
    except QdrantConnectionError as exc:
        raise ToolError("Qdrant unavailable") from exc
    except QdrantError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def list_documents(
    ctx: Context,
    collection: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List documents in the knowledge base.

    Args:
        collection: Filter by collection name (optional).
        limit: Maximum number of documents to return (default: 50).
    """
    db, _, _ = _ctx_resources(ctx)
    try:
        conditions: list[str] = []
        params: list[str | int] = []

        if collection:
            async with db.execute(
                "SELECT id FROM collections WHERE name = ?", (collection,)
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise ToolError(f"Collection '{collection}' does not exist")
            conditions.append("d.collection_id = ?")
            params.append(row[0])

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        docs = []
        async with db.execute(
            f"SELECT d.id, d.filename, d.file_path, d.file_type, d.chunk_count, "  # noqa: S608
            f"d.status, d.created_at FROM documents d{where} "
            f"ORDER BY d.created_at DESC LIMIT ?",
            params,
        ) as cursor:
            async for row in cursor:
                docs.append({
                    "id": row[0],
                    "filename": row[1],
                    "file_path": row[2],
                    "file_type": row[3],
                    "chunk_count": row[4],
                    "status": row[5],
                    "created_at": row[6],
                })
        return docs
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def get_document(doc_id: str, ctx: Context) -> dict[str, Any]:
    """Get details about a specific document.

    Args:
        doc_id: The document ID to look up.
    """
    db, _, _ = _ctx_resources(ctx)
    try:
        async with db.execute(
            "SELECT id, collection_id, filename, file_path, file_type, "
            "file_hash, chunk_count, status, error_message, created_at, updated_at "
            "FROM documents WHERE id = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise ToolError(f"Document '{doc_id}' not found")

        return {
            "id": row[0],
            "collection_id": row[1],
            "filename": row[2],
            "file_path": row[3],
            "file_type": row[4],
            "file_hash": row[5],
            "chunk_count": row[6],
            "status": row[7],
            "error_message": row[8],
            "created_at": row[9],
            "updated_at": row[10],
        }
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def delete_document(doc_id: str, ctx: Context) -> dict[str, str]:
    """Delete a document from the knowledge base.

    Args:
        doc_id: The document ID to delete.
    """
    db, _, qdrant = _ctx_resources(ctx)
    try:
        async with db.execute(
            "SELECT d.id, d.file_path, c.name "
            "FROM documents d JOIN collections c ON d.collection_id = c.id "
            "WHERE d.id = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if not row:
            raise ToolError(f"Document '{doc_id}' not found")

        file_path = row[1]
        collection_name = row[2]

        await qdrant.delete_points_by_filter(collection_name, {"file_path": file_path})
        await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await db.commit()

        return {"message": f"Document '{doc_id}' deleted"}
    except ToolError:
        raise
    except QdrantConnectionError as exc:
        raise ToolError("Qdrant unavailable") from exc
    except Exception as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def health(ctx: Context) -> dict[str, Any]:
    """Check the health of all services (Ollama, Qdrant, SQLite).

    Returns status for each service and overall system health.
    """
    db, ollama, qdrant = _ctx_resources(ctx)

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
    return {
        "status": "healthy" if all_ok else "degraded",
        "ollama": {"status": "ok" if ollama_ok else "error", "detail": ollama_detail},
        "qdrant": {"status": "ok" if qdrant_ok else "error", "detail": qdrant_detail},
        "sqlite": {"status": "ok" if sqlite_ok else "error", "detail": sqlite_detail},
    }


@mcp.tool
async def stats(ctx: Context, days: int = 30) -> dict[str, Any]:
    """Get aggregate statistics about the knowledge base.

    Args:
        days: Number of days to aggregate (default: 30).
    """
    db, _, _ = _ctx_resources(ctx)
    try:
        async with db.execute(
            "SELECT COUNT(*), COALESCE(AVG(latency_ms), 0) FROM queries"
        ) as cursor:
            row = await cursor.fetchone()
        total = row[0] if row else 0
        avg_latency = round(row[1], 2) if row else 0.0

        by_interface: dict[str, int] = {}
        async with db.execute(
            "SELECT interface, COUNT(*) FROM queries GROUP BY interface"
        ) as cursor:
            async for row in cursor:
                by_interface[row[0]] = row[1]

        by_type: dict[str, int] = {}
        async with db.execute(
            "SELECT query_type, COUNT(*) FROM queries GROUP BY query_type"
        ) as cursor:
            async for row in cursor:
                by_type[row[0]] = row[1]

        return {
            "total_queries": total,
            "avg_latency_ms": avg_latency,
            "queries_by_interface": by_interface,
            "queries_by_type": by_type,
            "period_days": days,
        }
    except Exception as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool
async def query_history(
    ctx: Context,
    limit: int = 20,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent query history.

    Args:
        limit: Maximum number of queries to return (default: 20).
        collection: Filter by collection name (optional).
    """
    db, _, _ = _ctx_resources(ctx)
    try:
        conditions: list[str] = []
        params: list[str | int] = []

        if collection:
            async with db.execute(
                "SELECT id FROM collections WHERE name = ?", (collection,)
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                conditions.append("q.collection_id = ?")
                params.append(row[0])

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        queries = []
        async with db.execute(
            f"SELECT q.id, q.query_text, q.query_type, q.search_mode, "  # noqa: S608
            f"q.result_count, q.latency_ms, q.interface, q.created_at "
            f"FROM queries q{where} ORDER BY q.created_at DESC LIMIT ?",
            params,
        ) as cursor:
            async for row in cursor:
                queries.append({
                    "id": row[0], "query_text": row[1], "query_type": row[2],
                    "search_mode": row[3], "result_count": row[4],
                    "latency_ms": row[5], "interface": row[6], "created_at": row[7],
                })
        return queries
    except Exception as exc:
        raise ToolError(str(exc)) from exc


def run() -> None:
    """Entry point for the rag-kb-mcp script.

    Supports CLI args:
        --config PATH   Path to config.yaml (default: config.yaml)
        --transport STR  Transport mode: stdio (default) or sse
    """
    import argparse

    parser = argparse.ArgumentParser(description="RAG Knowledge Base MCP Server")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default, for Claude Code) or sse (network)",
    )
    args = parser.parse_args()

    # Set config path before server starts
    import os

    os.environ.setdefault("RAG_CONFIG_PATH", args.config)

    mcp.run(transport=args.transport)
