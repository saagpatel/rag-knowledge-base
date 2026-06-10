"""Rich output helpers for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from rag_kb.ingestion.orchestrator import BatchIngestionResult, IngestionResult
    from rag_kb.retrieval.engine import RetrievalResponse

console = Console()
error_console = Console(stderr=True)


def print_error(message: str, hint: str | None = None) -> None:
    """Print error message to stderr, with optional hint."""
    error_console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        error_console.print(f"[dim]{hint}[/dim]")


def print_success(message: str) -> None:
    """Print success message with green checkmark."""
    console.print(f"[green]\\u2713[/green] {message}")


def print_ingestion_result(result: IngestionResult) -> None:
    """Print single-file ingestion result."""
    from rag_kb.models.schema import DocumentStatus

    completed = DocumentStatus.COMPLETED
    icon = "[green]\\u2713[/green]" if result.status == completed else "[red]\\u2717[/red]"
    path = result.file_path
    if result.status == DocumentStatus.COMPLETED:
        if result.chunk_count > 0:
            console.print(f"{icon} {path} ({result.chunk_count} chunks)")
        else:
            console.print(f"[yellow]\\u25CB[/yellow] {path} (unchanged, skipped)")
    else:
        console.print(f"{icon} {path}: {result.error_message or 'unknown error'}")


def print_batch_summary(result: BatchIngestionResult) -> None:
    """Print batch ingestion results as a table with summary footer."""
    from rag_kb.models.schema import DocumentStatus

    table = Table(title="Ingestion Results")
    table.add_column("Status", width=8)
    table.add_column("File", style="cyan")
    table.add_column("Chunks", justify="right")

    for r in result.results:
        if r.status == DocumentStatus.COMPLETED and r.chunk_count > 0:
            status = "[green]OK[/green]"
        elif r.status == DocumentStatus.COMPLETED and r.chunk_count == 0:
            status = "[yellow]SKIP[/yellow]"
        else:
            status = "[red]FAIL[/red]"
        chunks = str(r.chunk_count) if r.chunk_count > 0 else "-"
        table.add_row(status, r.file_path, chunks)

    console.print(table)
    console.print(
        f"\n[bold]{result.total_files}[/bold] files: "
        f"[green]{result.processed} processed[/green], "
        f"[red]{result.failed} failed[/red], "
        f"[yellow]{result.skipped} skipped[/yellow]"
    )


def print_search_results(response: RetrievalResponse) -> None:
    """Print search results as a table."""
    if not response.results:
        console.print("[dim]0 results[/dim]")
        return

    table = Table()
    table.add_column("#", justify="right", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("File", style="cyan")
    table.add_column("Chunk", justify="right", width=6)
    table.add_column("Preview")

    for i, r in enumerate(response.results, 1):
        preview = r.content.replace("\n", " ")[:80]
        if len(r.content) > 80:
            preview += "..."
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.file_path,
            f"{r.chunk_index}/{r.total_chunks}",
            preview,
        )

    console.print(table)
    console.print(
        f"\n[dim]{response.total} results in {response.latency_ms:.0f}ms "
        f"({response.mode} search)[/dim]"
    )


def print_answer(
    answer: str,
    sources: list[dict[str, Any]],
    latency_ms: float,
    model: str,
) -> None:
    """Print generation answer with sources and metadata."""
    console.print(Panel(answer, title="Answer", border_style="green"))

    if sources:
        table = Table(title="Sources")
        table.add_column("#", justify="right", width=3)
        table.add_column("File", style="cyan")
        table.add_column("Chunk", justify="right")
        table.add_column("Score", justify="right")

        for i, s in enumerate(sources, 1):
            table.add_row(
                str(i),
                str(s.get("file_path", "")),
                f"{s.get('chunk_index', 0)}/{s.get('total_chunks', 0)}",
                f"{s.get('score', 0):.4f}",
            )
        console.print(table)

    console.print(f"\n[dim]Model: {model} | Latency: {latency_ms:.0f}ms[/dim]")


def print_documents_table(documents: list[dict[str, Any]]) -> None:
    """Print documents as a table."""
    if not documents:
        console.print("[dim]No documents found.[/dim]")
        return

    table = Table(title="Documents")
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Filename", style="cyan")
    table.add_column("Type", width=8)
    table.add_column("Chunks", justify="right", width=7)
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for d in documents:
        status_str = d.get("status", "unknown")
        if status_str == "completed":
            status = "[green]OK[/green]"
        elif status_str == "failed":
            status = "[red]FAIL[/red]"
        else:
            status = f"[yellow]{status_str}[/yellow]"

        table.add_row(
            str(d.get("id", ""))[:12],
            str(d.get("filename", "")),
            str(d.get("file_type", "")),
            str(d.get("chunk_count", 0)),
            status,
            str(d.get("created_at", ""))[:19],
        )
    console.print(table)


def print_document_detail(doc: dict[str, Any]) -> None:
    """Print detailed document info."""
    console.print(f"[bold]Document {doc['id']}[/bold]")
    console.print(f"  Filename:    {doc.get('filename', '')}")
    console.print(f"  Path:        {doc.get('file_path', '')}")
    console.print(f"  Type:        {doc.get('file_type', '')}")
    console.print(f"  Hash:        {doc.get('file_hash', '')}")
    console.print(f"  Chunks:      {doc.get('chunk_count', 0)}")
    console.print(f"  Status:      {doc.get('status', '')}")
    if doc.get("error_message"):
        console.print(f"  Error:       [red]{doc['error_message']}[/red]")
    console.print(f"  Created:     {doc.get('created_at', '')}")
    console.print(f"  Updated:     {doc.get('updated_at', '')}")


def print_collections_table(collections: list[dict[str, Any]]) -> None:
    """Print collections as a table."""
    if not collections:
        console.print("[dim]No collections found.[/dim]")
        return

    table = Table(title="Collections")
    table.add_column("Name", style="cyan")
    table.add_column("Points", justify="right")
    table.add_column("Vectors", justify="right")
    table.add_column("Status")

    for c in collections:
        table.add_row(
            str(c["name"]),
            str(c.get("points_count", 0)),
            str(c.get("vectors_count", 0)),
            str(c.get("status", "unknown")),
        )
    console.print(table)


def print_status(
    ollama_ok: bool,
    qdrant_ok: bool,
    config: Any,
) -> None:
    """Print service health and config summary."""
    ollama_icon = "[green]\\u2713 Online[/green]" if ollama_ok else "[red]\\u2717 Offline[/red]"
    qdrant_icon = "[green]\\u2713 Online[/green]" if qdrant_ok else "[red]\\u2717 Offline[/red]"

    console.print("[bold]Service Status[/bold]")
    console.print(f"  Ollama:  {ollama_icon}  ({config.ollama.host})")
    console.print(f"  Qdrant:  {qdrant_icon}  ({config.qdrant.host}:{config.qdrant.port})")
    console.print()
    console.print("[bold]Configuration[/bold]")
    console.print(f"  Embedding model: {config.ollama.embedding_model}")
    console.print(f"  Generation model: {config.ollama.generation_model}")
    console.print(f"  Database: {config.sqlite.path}")
