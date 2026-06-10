"""Configuration loader — Pydantic models + YAML + env var overrides."""

from __future__ import annotations

import logging as _logging
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel, Field

_config_logger = _logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    host: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "mistral:7b"
    timeout: int = 120
    max_retries: int = 3


class QdrantConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6333
    grpc_port: int = 6334
    collection_prefix: str = "rag_"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200


class SqliteConfig(BaseModel):
    path: str = "data/rag_kb.db"


class ChunkingConfig(BaseModel):
    default_size: int = 512
    default_overlap: int = 50
    min_chunk_size: int = 50
    max_chunk_size: int = 2048


class SearchConfig(BaseModel):
    default_top_k: int = 10
    max_top_k: int = 100
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    default_mode: str = "hybrid"


class GenerationConfig(BaseModel):
    max_context_chunks: int = 10
    temperature: float = 0.1
    max_tokens: int = 2048


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/rag_kb.log"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    strict_startup: bool = False
    api_key: str = ""
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10


class CacheConfig(BaseModel):
    embedding_cache_size: int = 1000
    enabled: bool = True


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    sqlite: SqliteConfig = Field(default_factory=SqliteConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply RAG_SECTION__KEY env vars on top of YAML-loaded config."""
    import os

    prefix = "RAG_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        if len(parts) == 2:
            section, field = parts
            if section not in data:
                data[section] = {}
            data[section][field] = value
    return data


def _warn_unknown_keys(data: dict[str, Any], model_class: type[BaseModel], prefix: str = "") -> None:
    """Log warnings for unknown config keys."""
    known_fields = set(model_class.model_fields.keys())
    for key in data:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in known_fields:
            _config_logger.warning("Unknown config key: %s (ignored)", full_key)
        elif isinstance(data[key], dict):
            # Recurse into nested sections
            field_info = model_class.model_fields.get(key)
            if field_info and field_info.annotation is not None:
                annotation = field_info.annotation
                # Unwrap Optional / Union types
                origin = get_origin(annotation)
                if origin is not None:
                    args = get_args(annotation)
                    # Pick the first non-None arg for Optional[X]
                    for arg in args:
                        if arg is not type(None) and isinstance(arg, type) and issubclass(arg, BaseModel):
                            annotation = arg
                            break
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    _warn_unknown_keys(data[key], annotation, full_key)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load config from YAML file, merge env var overrides."""
    config_path = Path(path)
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    data = _apply_env_overrides(data)
    _warn_unknown_keys(data, AppConfig)
    return AppConfig(**data)


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> AppConfig:
    """Singleton accessor — loads once, caches.

    Resolution order for config path:
    1. Explicit ``path`` argument
    2. ``RAG_CONFIG_PATH`` environment variable
    3. Default ``config.yaml``
    """
    import os

    resolved = path or os.environ.get("RAG_CONFIG_PATH", "config.yaml")
    return load_config(resolved)
