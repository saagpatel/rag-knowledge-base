"""Benchmark scoring functions and runner for retrieval quality evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_kb.models.schema import SearchMode
from rag_kb.retrieval.engine import RetrievalEngine, RetrievalRequest

if TYPE_CHECKING:
    from rag_kb.core.ollama_client import OllamaClient
    from rag_kb.core.qdrant_client import QdrantManager


@dataclass
class QueryResult:
    query: str
    description: str
    expected_files: list[str]
    retrieved_files: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_ms: float


@dataclass
class BenchmarkReport:
    results: list[QueryResult] = field(default_factory=list)
    avg_recall_at_5: float = 0.0
    avg_recall_at_10: float = 0.0
    avg_mrr: float = 0.0
    avg_latency_ms: float = 0.0
    collection: str = "default"
    mode: str = "hybrid"
    top_k: int = 10
    total_queries: int = 0


def recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int) -> float:
    """Fraction of expected files found in top-k results (substring match).

    Returns 0.0 if expected_files is empty.
    """
    if not expected_files:
        return 0.0
    top_k_files = retrieved_files[:k]
    found = sum(
        1 for exp in expected_files
        if any(exp in ret for ret in top_k_files)
    )
    return found / len(expected_files)


def reciprocal_rank(retrieved_files: list[str], expected_files: list[str]) -> float:
    """1/rank of the first relevant result (substring match).

    Returns 0.0 if no relevant result found or expected_files is empty.
    """
    if not expected_files:
        return 0.0
    for i, ret in enumerate(retrieved_files, 1):
        if any(exp in ret for exp in expected_files):
            return 1.0 / i
    return 0.0


def load_testset(path: Path) -> list[dict[str, Any]]:
    """Load a benchmark test set from a JSON file."""
    data = json.loads(path.read_text())
    return data["queries"]


async def run_benchmark(
    testset_path: Path,
    collection: str,
    mode: str,
    top_k: int,
    ollama: OllamaClient,
    qdrant: QdrantManager,
) -> BenchmarkReport:
    """Load test set, run queries, compute metrics, return report."""
    queries = load_testset(testset_path)
    engine = RetrievalEngine(ollama, qdrant)

    report = BenchmarkReport(
        collection=collection,
        mode=mode,
        top_k=top_k,
        total_queries=len(queries),
    )

    for q in queries:
        query_text = q["query"]
        expected = q.get("expected_files", [])
        description = q.get("description", "")

        start = time.perf_counter()
        request = RetrievalRequest(
            query=query_text,
            collection=collection,
            mode=SearchMode(mode),
            top_k=top_k,
        )
        response = await engine.search(request)
        latency = (time.perf_counter() - start) * 1000

        retrieved = [r.file_path for r in response.results]

        result = QueryResult(
            query=query_text,
            description=description,
            expected_files=expected,
            retrieved_files=retrieved,
            recall_at_5=recall_at_k(retrieved, expected, 5),
            recall_at_10=recall_at_k(retrieved, expected, 10),
            mrr=reciprocal_rank(retrieved, expected),
            latency_ms=round(latency, 2),
        )
        report.results.append(result)

    if report.results:
        n = len(report.results)
        report.avg_recall_at_5 = round(sum(r.recall_at_5 for r in report.results) / n, 4)
        report.avg_recall_at_10 = round(sum(r.recall_at_10 for r in report.results) / n, 4)
        report.avg_mrr = round(sum(r.mrr for r in report.results) / n, 4)
        report.avg_latency_ms = round(sum(r.latency_ms for r in report.results) / n, 2)

    return report


def format_report(report: BenchmarkReport) -> str:
    """Format a BenchmarkReport as a markdown table."""
    lines = [
        "# Benchmark Report",
        "",
        f"- Collection: {report.collection}",
        f"- Mode: {report.mode}",
        f"- Top-K: {report.top_k}",
        f"- Queries: {report.total_queries}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Recall@5 | {report.avg_recall_at_5:.4f} |",
        f"| Recall@10 | {report.avg_recall_at_10:.4f} |",
        f"| MRR | {report.avg_mrr:.4f} |",
        f"| Avg Latency | {report.avg_latency_ms:.2f}ms |",
        "",
        "## Per-Query Results",
        "",
        "| Query | R@5 | R@10 | MRR | Latency |",
        "|-------|-----|------|-----|---------|",
    ]

    for r in report.results:
        q = r.query[:40] + "..." if len(r.query) > 40 else r.query
        row = (
            f"| {q} | {r.recall_at_5:.2f} | {r.recall_at_10:.2f} "
            f"| {r.mrr:.2f} | {r.latency_ms:.0f}ms |"
        )
        lines.append(row)

    return "\n".join(lines)
