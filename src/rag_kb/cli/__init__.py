"""CLI interface — Click group + commands for the RAG knowledge base."""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import click

from rag_kb.core.config import get_config
from rag_kb.core.errors import (
    CollectionNotFoundError,
    OllamaConnectionError,
    QdrantConnectionError,
    RAGError,
)
from rag_kb.models.schema import Interface, QueryType, SearchMode

from .formatting import (
    console,
    print_answer,
    print_batch_summary,
    print_collections_table,
    print_document_detail,
    print_documents_table,
    print_error,
    print_ingestion_result,
    print_search_results,
    print_status,
    print_success,
)


def _run(coro: Any) -> Any:
    """Run async coroutine from sync Click context."""
    return asyncio.run(coro)


@asynccontextmanager
async def _resources(
    need_db: bool = True,
    need_ollama: bool = True,
    need_qdrant: bool = True,
):
    """Open only the resources a command needs, clean up on exit."""
    from rag_kb.core.database import init_db
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager

    db = None
    ollama = None
    qdrant = None
    try:
        if need_db:
            db = await init_db()
        if need_ollama:
            ollama = OllamaClient()
        if need_qdrant:
            qdrant = QdrantManager()
        yield {"db": db, "ollama": ollama, "qdrant": qdrant}
    finally:
        if db is not None:
            await db.close()
        if ollama is not None:
            await ollama.close()
        if qdrant is not None:
            await qdrant.close()


def _error_hint(exc: RAGError) -> str | None:
    """Map known errors to user-friendly hints."""
    if isinstance(exc, OllamaConnectionError):
        return "Is Ollama running? Try: ollama serve"
    if isinstance(exc, QdrantConnectionError):
        return "Is Qdrant running? Try: docker compose up -d qdrant"
    if isinstance(exc, CollectionNotFoundError):
        return "Create it first: rag collections create <name>"
    return None


# --- Click group ---


@click.group()
@click.version_option(package_name="rag-kb")
def main():
    """RAG Knowledge Base — local semantic search & Q&A."""


# --- ingest ---


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-c", "--collection", default="default", help="Target collection name.")
@click.option("--chunk-size", default=512, type=int, help="Chunk size in tokens.")
@click.option("--chunk-overlap", default=50, type=int, help="Overlap between chunks.")
@click.option("-p", "--patterns", multiple=True, help="Glob patterns for directory ingest.")
def ingest(
    path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    patterns: tuple[str, ...],
) -> None:
    """Ingest a file or directory into the knowledge base."""
    _run(_ingest_async(path, collection, chunk_size, chunk_overlap, list(patterns) or None))


async def _ingest_async(
    path: str,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    patterns: list[str] | None,
) -> None:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    from rag_kb.ingestion.orchestrator import (
        _SUPPORTED_EXTENSIONS,
        ingest_directory,
        ingest_file,
    )

    try:
        async with _resources(need_db=True, need_ollama=True, need_qdrant=True) as res:
            target = Path(path)
            if target.is_dir():
                # Count files first for progress bar
                files: list[Path] = []
                if patterns:
                    for p in patterns:
                        files.extend(target.rglob(p))
                else:
                    for ext in _SUPPORTED_EXTENSIONS:
                        files.extend(target.rglob(f"*{ext}"))
                files = sorted(set(files))

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]Ingesting...[/bold]"),
                    TextColumn("{task.fields[filename]}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as progress:
                    task_id = progress.add_task("Ingesting", total=len(files), filename="")

                    def _on_progress(filename: str, _path: str) -> None:
                        progress.update(task_id, advance=1, filename=filename)

                    result = await ingest_directory(
                        target,
                        collection,
                        res["db"],
                        res["ollama"],
                        res["qdrant"],
                        patterns=patterns,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        progress_callback=_on_progress,
                    )
                print_batch_summary(result)
            else:
                result = await ingest_file(
                    target,
                    collection,
                    res["db"],
                    res["ollama"],
                    res["qdrant"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                print_ingestion_result(result)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


# --- search ---


@main.command()
@click.argument("query", required=False, default=None)
@click.option("-c", "--collection", default="default", help="Collection to search.")
@click.option(
    "-m", "--mode",
    type=click.Choice(["dense", "sparse", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="Search mode.",
)
@click.option("-k", "--top-k", default=10, type=int, help="Number of results.")
@click.option("--rerank", is_flag=True, help="Enable reranking.")
@click.option("-i", "--interactive", is_flag=True, help="Enter interactive search loop.")
def search(query: str | None, collection: str, mode: str, top_k: int, rerank: bool, interactive: bool) -> None:
    """Search the knowledge base."""
    if interactive:
        _run(_search_interactive(collection, mode, top_k, rerank))
    elif query:
        _run(_search_async(query, collection, mode, top_k, rerank))
    else:
        console.print("[red]Error:[/red] query is required (or use --interactive)")
        sys.exit(1)


async def _search_async(
    query: str, collection: str, mode: str, top_k: int, rerank: bool
) -> None:
    from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest
    from rag_kb.retrieval.query_log import log_query

    try:
        async with _resources(need_db=True, need_ollama=True, need_qdrant=True) as res:
            engine = RetrievalEngine(res["ollama"], res["qdrant"])
            request = RetrievalRequest(
                query=query,
                collection=collection,
                mode=SearchMode(mode),
                top_k=top_k,
                rerank=rerank,
            )
            response = await engine.search(request)
            await log_query(res["db"], response, QueryType.SEARCH, Interface.CLI)
            print_search_results(response)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


async def _search_interactive(collection: str, mode: str, top_k: int, rerank: bool) -> None:
    from rich.prompt import Prompt

    from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest
    from rag_kb.retrieval.query_log import log_query

    console.print("[bold]Interactive search mode[/bold] (type 'quit' to exit)")
    try:
        async with _resources(need_db=True, need_ollama=True, need_qdrant=True) as res:
            while True:
                try:
                    query = Prompt.ask("[cyan]Search[/cyan]")
                except (EOFError, KeyboardInterrupt):
                    break

                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query.strip():
                    continue

                try:
                    engine = RetrievalEngine(res["ollama"], res["qdrant"])
                    request = RetrievalRequest(
                        query=query,
                        collection=collection,
                        mode=SearchMode(mode),
                        top_k=top_k,
                        rerank=rerank,
                    )
                    response = await engine.search(request)
                    await log_query(res["db"], response, QueryType.SEARCH, Interface.CLI)
                    print_search_results(response)
                except RAGError as exc:
                    print_error(str(exc), hint=_error_hint(exc))
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    console.print("\n[dim]Goodbye.[/dim]")


# --- ask ---


@main.command()
@click.argument("query")
@click.option("-c", "--collection", default="default", help="Collection to search.")
@click.option(
    "-m", "--mode",
    type=click.Choice(["dense", "sparse", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="Search mode.",
)
@click.option("-k", "--top-k", default=5, type=int, help="Number of context chunks.")
@click.option("--model", default=None, help="Override generation model.")
@click.option("--no-stream", is_flag=True, help="Disable streaming output.")
def ask(
    query: str,
    collection: str,
    mode: str,
    top_k: int,
    model: str | None,
    no_stream: bool,
) -> None:
    """Ask a question and get an AI-generated answer."""
    _run(_ask_async(query, collection, mode, top_k, model, no_stream))


async def _ask_async(
    query: str,
    collection: str,
    mode: str,
    top_k: int,
    model: str | None,
    no_stream: bool,
) -> None:
    from rag_kb.generation.engine import GenerationEngine
    from rag_kb.generation.prompt import extract_sources
    from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest
    from rag_kb.retrieval.query_log import log_query

    try:
        async with _resources(need_db=True, need_ollama=True, need_qdrant=True) as res:
            # Retrieve
            retrieval = RetrievalEngine(res["ollama"], res["qdrant"])
            request = RetrievalRequest(
                query=query,
                collection=collection,
                mode=SearchMode(mode),
                top_k=top_k,
            )
            response = await retrieval.search(request)
            await log_query(res["db"], response, QueryType.QA, Interface.CLI)

            # Generate
            gen = GenerationEngine(res["ollama"])

            if no_stream:
                result = await gen.answer(query, response.results, model=model, stream=False)
                print_answer(
                    result.answer,
                    result.sources,
                    result.latency_ms,
                    result.model,
                )
            else:
                # Streaming: manually time and extract sources
                sources = extract_sources(response.results)
                start = time.perf_counter()
                token_gen = await gen.answer(query, response.results, model=model, stream=True)
                async for token in token_gen:
                    console.print(token, end="")
                console.print()  # newline after stream
                latency_ms = (time.perf_counter() - start) * 1000
                used_model = model or get_config().ollama.generation_model
                print_answer("", sources, latency_ms, used_model)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


# --- collections ---


@main.group()
def collections():
    """Manage collections."""


@collections.command("list")
def collections_list() -> None:
    """List all collections."""
    _run(_collections_list_async())


async def _collections_list_async() -> None:
    try:
        async with _resources(need_db=False, need_ollama=False, need_qdrant=True) as res:
            names = await res["qdrant"].list_collections()
            infos = []
            for name in names:
                info = await res["qdrant"].get_collection_info(name)
                infos.append(info)
            print_collections_table(infos)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


@collections.command("info")
@click.argument("name")
def collections_info(name: str) -> None:
    """Show details for a collection."""
    _run(_collections_info_async(name))


async def _collections_info_async(name: str) -> None:
    try:
        async with _resources(need_db=False, need_ollama=False, need_qdrant=True) as res:
            info = await res["qdrant"].get_collection_info(name)
            print_collections_table([info])
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


@collections.command("create")
@click.argument("name")
def collections_create(name: str) -> None:
    """Create a new collection."""
    _run(_collections_create_async(name))


async def _collections_create_async(name: str) -> None:
    try:
        async with _resources(need_db=True, need_ollama=False, need_qdrant=True) as res:
            await res["qdrant"].create_collection(name)
            print_success(f"Collection '{name}' created.")
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


@collections.command("delete")
@click.argument("name")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def collections_delete(name: str, yes: bool) -> None:
    """Delete a collection."""
    if not yes:
        click.confirm(f"Delete collection '{name}'? This cannot be undone", abort=True)
    _run(_collections_delete_async(name))


async def _collections_delete_async(name: str) -> None:
    try:
        async with _resources(need_db=False, need_ollama=False, need_qdrant=True) as res:
            await res["qdrant"].delete_collection(name)
            print_success(f"Collection '{name}' deleted.")
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


# --- documents ---


@main.group()
def documents():
    """Manage documents."""


@documents.command("list")
@click.option("-c", "--collection", default=None, help="Filter by collection.")
@click.option("-n", "--limit", default=50, type=int, help="Max documents to show.")
def documents_list(collection: str | None, limit: int) -> None:
    """List ingested documents."""
    _run(_documents_list_async(collection, limit))


async def _documents_list_async(collection: str | None, limit: int) -> None:
    try:
        async with _resources(need_db=True, need_ollama=False, need_qdrant=False) as res:
            conditions: list[str] = []
            params: list[str | int] = []

            if collection:
                async with res["db"].execute(
                    "SELECT id FROM collections WHERE name = ?", (collection,)
                ) as cursor:
                    row = await cursor.fetchone()
                if not row:
                    print_error(f"Collection '{collection}' not found")
                    return
                conditions.append("d.collection_id = ?")
                params.append(row[0])

            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            docs = []
            async with res["db"].execute(
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
            print_documents_table(docs)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


@documents.command("info")
@click.argument("doc_id")
def documents_info(doc_id: str) -> None:
    """Show details for a document."""
    _run(_documents_info_async(doc_id))


async def _documents_info_async(doc_id: str) -> None:
    try:
        async with _resources(need_db=True, need_ollama=False, need_qdrant=False) as res:
            async with res["db"].execute(
                "SELECT id, collection_id, filename, file_path, file_type, "
                "file_hash, chunk_count, status, error_message, created_at, updated_at "
                "FROM documents WHERE id = ?",
                (doc_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                print_error(f"Document '{doc_id}' not found")
                return

            doc = {
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
            print_document_detail(doc)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


@documents.command("delete")
@click.argument("doc_id")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def documents_delete(doc_id: str, yes: bool) -> None:
    """Delete a document."""
    if not yes:
        click.confirm(f"Delete document '{doc_id}'? This cannot be undone", abort=True)
    _run(_documents_delete_async(doc_id))


async def _documents_delete_async(doc_id: str) -> None:
    try:
        async with _resources(need_db=True, need_ollama=False, need_qdrant=True) as res:
            async with res["db"].execute(
                "SELECT d.id, d.file_path, c.name "
                "FROM documents d JOIN collections c ON d.collection_id = c.id "
                "WHERE d.id = ?",
                (doc_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                print_error(f"Document '{doc_id}' not found")
                return

            await res["qdrant"].delete_points_by_filter(row[2], {"file_path": row[1]})
            await res["db"].execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await res["db"].commit()
            print_success(f"Document '{doc_id}' deleted.")
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)


# --- benchmark ---


@main.command()
@click.argument("testset", type=click.Path(exists=True))
@click.option("-c", "--collection", default="default", help="Collection to benchmark.")
@click.option(
    "-m", "--mode",
    type=click.Choice(["dense", "sparse", "hybrid"], case_sensitive=False),
    default="hybrid",
    help="Search mode.",
)
@click.option("-k", "--top-k", default=10, type=int, help="Number of results to retrieve.")
@click.option("-o", "--output", default=None, type=click.Path(), help="Save report to file.")
def benchmark(
    testset: str,
    collection: str,
    mode: str,
    top_k: int,
    output: str | None,
) -> None:
    """Run a retrieval benchmark against a test set."""
    _run(_benchmark_async(testset, collection, mode, top_k, output))


async def _benchmark_async(
    testset: str,
    collection: str,
    mode: str,
    top_k: int,
    output: str | None,
) -> None:
    from .benchmark import format_report, run_benchmark

    try:
        async with _resources(need_db=False, need_ollama=True, need_qdrant=True) as res:
            with console.status("[bold]Running benchmark...[/bold]"):
                report = await run_benchmark(
                    Path(testset),
                    collection,
                    mode,
                    top_k,
                    res["ollama"],
                    res["qdrant"],
                )
            report_text = format_report(report)
            console.print(report_text)

            if output:
                Path(output).write_text(report_text)
                print_success(f"Report saved to {output}")
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


# --- status ---


@main.command()
def status() -> None:
    """Check service health and configuration."""
    _run(_status_async())


async def _status_async() -> None:
    config = get_config()
    try:
        async with _resources(need_db=False, need_ollama=True, need_qdrant=True) as res:
            ollama_ok = await res["ollama"].health()

            qdrant_ok = False
            try:
                await res["qdrant"].list_collections()
                qdrant_ok = True
            except Exception:
                pass

            print_status(ollama_ok, qdrant_ok, config)
    except RAGError as exc:
        print_error(str(exc), hint=_error_hint(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)
