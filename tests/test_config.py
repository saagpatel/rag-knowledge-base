"""Tests for config loader."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from rag_kb.core.config import AppConfig, load_config


def test_load_default_config():
    """Loading config.yaml produces correct defaults."""
    config = load_config("config.yaml")
    assert config.ollama.host == "http://127.0.0.1:11434"
    assert config.ollama.embedding_model == "nomic-embed-text"
    assert config.qdrant.port == 6333
    assert config.chunking.default_size == 512
    assert config.search.default_mode == "hybrid"
    assert config.server.host == "127.0.0.1"


def test_env_var_override(tmp_dir: Path, monkeypatch):
    """RAG_SECTION__KEY env vars override YAML values."""
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"ollama": {"host": "http://original:11434"}}, f)

    monkeypatch.setenv("RAG_OLLAMA__HOST", "http://overridden:11434")
    config = load_config(config_path)
    assert config.ollama.host == "http://overridden:11434"


def test_missing_config_uses_defaults(tmp_dir: Path):
    """No YAML file falls back to all defaults."""
    config = load_config(tmp_dir / "nonexistent.yaml")
    assert isinstance(config, AppConfig)
    assert config.ollama.host == "http://127.0.0.1:11434"
    assert config.sqlite.path == "data/rag_kb.db"


def test_invalid_config_raises(tmp_dir: Path):
    """Invalid values are rejected by Pydantic."""
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"qdrant": {"port": "not-a-number"}}, f)

    try:
        load_config(config_path)
        assert False, "Should have raised"
    except Exception:
        pass


# --- Unknown key warnings ---


def test_unknown_top_level_key_warns(tmp_dir: Path, caplog):
    """Unknown top-level keys produce a warning."""
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"bogus_section": {"foo": "bar"}}, f)

    with caplog.at_level(logging.WARNING, logger="rag_kb.core.config"):
        load_config(config_path)

    assert any("Unknown config key: bogus_section" in msg for msg in caplog.messages)


def test_unknown_nested_key_warns(tmp_dir: Path, caplog):
    """Unknown nested keys produce a warning with dotted path."""
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"ollama": {"host": "http://127.0.0.1:11434", "nonexistent_key": 42}}, f)

    with caplog.at_level(logging.WARNING, logger="rag_kb.core.config"):
        load_config(config_path)

    assert any("Unknown config key: ollama.nonexistent_key" in msg for msg in caplog.messages)


def test_known_keys_no_warning(tmp_dir: Path, caplog):
    """Valid config keys do not produce warnings."""
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"ollama": {"host": "http://127.0.0.1:11434"}}, f)

    with caplog.at_level(logging.WARNING, logger="rag_kb.core.config"):
        load_config(config_path)

    config_warnings = [msg for msg in caplog.messages if "Unknown config key" in msg]
    assert config_warnings == []


def test_empty_config_no_warning(tmp_dir: Path, caplog):
    """Empty config produces no warnings."""
    with caplog.at_level(logging.WARNING, logger="rag_kb.core.config"):
        load_config(tmp_dir / "nonexistent.yaml")

    config_warnings = [msg for msg in caplog.messages if "Unknown config key" in msg]
    assert config_warnings == []
