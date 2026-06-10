"""Shared test fixtures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from rag_kb.core.config import AppConfig, load_config


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def tmp_config(tmp_dir: Path) -> AppConfig:
    """Create a temp config.yaml and return loaded AppConfig."""
    config_data = {
        "ollama": {"host": "http://127.0.0.1:11434"},
        "qdrant": {"host": "127.0.0.1", "port": 6333},
        "sqlite": {"path": str(tmp_dir / "test.db")},
        "logging": {"level": "DEBUG", "file": str(tmp_dir / "test.log")},
    }
    config_path = tmp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return load_config(config_path)


@pytest.fixture
async def tmp_db(tmp_dir: Path):
    """Create a temp SQLite DB with migrations applied."""
    from rag_kb.core.database import init_db

    db_path = str(tmp_dir / "test.db")
    migrations_dir = Path(__file__).parent.parent / "migrations"
    db = await init_db(db_path=db_path, migrations_dir=migrations_dir)
    yield db
    await db.close()
