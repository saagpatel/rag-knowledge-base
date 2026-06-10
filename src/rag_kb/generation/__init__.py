"""Generation engine — RAG prompt construction + LLM answers."""

from .engine import GenerationEngine, GenerationResult
from .prompt import build_messages, extract_sources

__all__ = [
    "GenerationEngine",
    "GenerationResult",
    "build_messages",
    "extract_sources",
]
