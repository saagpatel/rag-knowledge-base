"""SQLite database + migration runner."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from rag_kb.core.config import get_config

logger = logging.getLogger(__name__)

_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS _migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def run_migrations(db: aiosqlite.Connection, migrations_dir: str | Path) -> None:
    """Run pending SQL migration files in order."""
    await db.execute(_MIGRATIONS_TABLE)
    await db.commit()

    migrations_path = Path(migrations_dir)
    if not migrations_path.exists():
        logger.warning("Migrations directory not found: %s", migrations_path)
        return

    applied: set[str] = set()
    async with db.execute("SELECT filename FROM _migrations") as cursor:
        async for row in cursor:
            applied.add(row[0])

    sql_files = sorted(migrations_path.glob("*.sql"))
    for sql_file in sql_files:
        if sql_file.name in applied:
            continue
        logger.info("Applying migration: %s", sql_file.name)
        sql = sql_file.read_text()
        await db.executescript(sql)
        await db.execute(
            "INSERT INTO _migrations (filename) VALUES (?)", (sql_file.name,)
        )
        await db.commit()
        logger.info("Applied migration: %s", sql_file.name)


async def init_db(
    db_path: str | None = None, migrations_dir: str | Path = "migrations"
) -> aiosqlite.Connection:
    """Create DB, run pending migrations, return connection."""
    if db_path is None:
        config = get_config()
        db_path = config.sqlite.path

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await run_migrations(db, migrations_dir)
    return db


@asynccontextmanager
async def get_db(
    db_path: str | None = None,
) -> AsyncGenerator[aiosqlite.Connection]:
    """Async context manager for database connections."""
    if db_path is None:
        config = get_config()
        db_path = config.sqlite.path

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
    finally:
        await db.close()
