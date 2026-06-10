"""RAG prompt builder — pure functions, no async, no external deps."""

from __future__ import annotations

from rag_kb.core.config import GenerationConfig, get_config
from rag_kb.core.errors import PromptTooLargeError
from rag_kb.models.search import RetrievalResult

_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the user's question using ONLY the "
    "context provided below. If the context does not contain enough information to "
    "answer the question, say so clearly — do not make up information.\n\n"
    "When citing information, reference the source number (e.g., [Source 1]).\n"
    "Be concise and accurate."
)

_CONTEXT_BLOCK_TEMPLATE = "[Source {index}: {file_path} (chunk {chunk_index}/{total_chunks})]"

_NO_CONTEXT_FALLBACK = "No relevant context found."


def build_messages(
    query: str,
    results: list[RetrievalResult],
    config: GenerationConfig | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the LLM from query and retrieval results.

    Returns ``[{"role": "system", ...}, {"role": "user", ...}]``.

    Raises ``PromptTooLargeError`` if the query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise PromptTooLargeError("Query cannot be empty")

    config = config or get_config().generation
    selected = results[: config.max_context_chunks]

    if selected:
        context_parts: list[str] = []
        for i, r in enumerate(selected, 1):
            header = _CONTEXT_BLOCK_TEMPLATE.format(
                index=i,
                file_path=r.file_path,
                chunk_index=r.chunk_index,
                total_chunks=r.total_chunks,
            )
            context_parts.append(f"{header}\n{r.content}")
        context_block = "\n\n".join(context_parts)
    else:
        context_block = _NO_CONTEXT_FALLBACK

    user_message = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def extract_sources(results: list[RetrievalResult]) -> list[dict[str, object]]:
    """Pull citation metadata from retrieval results."""
    return [
        {
            "file_path": r.file_path,
            "score": r.score,
            "chunk_index": r.chunk_index,
            "total_chunks": r.total_chunks,
            "file_type": r.file_type,
        }
        for r in results
    ]
