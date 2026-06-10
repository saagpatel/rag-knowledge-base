"""Tests for database + migration runner."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from rag_kb.core.database import init_db, run_migrations


async def test_migration_creates_tables(tmp_db: aiosqlite.Connection):
    """All 4 tables exist after migration."""
    async with tmp_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cursor:
        tables = [row[0] for row in await cursor.fetchall()]

    assert "collections" in tables
    assert "documents" in tables
    assert "ingestion_jobs" in tables
    assert "queries" in tables
    assert "_migrations" in tables


async def test_migration_idempotent(tmp_dir: Path):
    """Running migrations twice causes no error."""
    db_path = str(tmp_dir / "test.db")
    migrations_dir = Path(__file__).parent.parent / "migrations"

    db = await init_db(db_path=db_path, migrations_dir=migrations_dir)
    # Run again — should be a no-op
    await run_migrations(db, migrations_dir)
    await db.close()


async def test_insert_collection(tmp_db: aiosqlite.Connection):
    """Can insert and read back a collection."""
    await tmp_db.execute(
        "INSERT INTO collections (id, name, description) VALUES (?, ?, ?)",
        ("col-1", "test-collection", "A test collection"),
    )
    await tmp_db.commit()

    async with tmp_db.execute(
        "SELECT id, name, description FROM collections WHERE id = ?", ("col-1",)
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[0] == "col-1"
    assert row[1] == "test-collection"


async def test_insert_document(tmp_db: aiosqlite.Connection):
    """Can insert a document with foreign key to collection."""
    await tmp_db.execute(
        "INSERT INTO collections (id, name) VALUES (?, ?)",
        ("col-1", "test-collection"),
    )
    await tmp_db.execute(
        """INSERT INTO documents
           (id, collection_id, filename, file_path, file_type, file_hash, file_size)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("doc-1", "col-1", "test.md", "/path/test.md", "markdown", "abc123", 1024),
    )
    await tmp_db.commit()

    async with tmp_db.execute(
        "SELECT id, collection_id, filename FROM documents WHERE id = ?", ("doc-1",)
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row[1] == "col-1"
    assert row[2] == "test.md"


async def test_migration_tracking(tmp_db: aiosqlite.Connection):
    """_migrations table tracks applied migrations."""
    async with tmp_db.execute("SELECT filename FROM _migrations") as cursor:
        rows = await cursor.fetchall()

    filenames = [row[0] for row in rows]
    assert "001_initial_schema.sql" in filenames
