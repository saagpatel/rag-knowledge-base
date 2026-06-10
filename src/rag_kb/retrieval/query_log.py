"""Query logging — persist search queries to SQLite."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cuid2 import cuid_wrapper

from rag_kb.models.schema import Interface, QueryType

if TYPE_CHECKING:
    import aiosqlite

    from .engine import RetrievalResponse

_cuid = cuid_wrapper()


async def log_query(
    db: aiosqlite.Connection,
    response: RetrievalResponse,
    query_type: QueryType = QueryType.SEARCH,
    interface: Interface = Interface.API,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Insert a query record into SQLite. Returns the generated query ID."""
    query_id = _cuid()
    now = datetime.now().isoformat()

    # Resolve collection FK (NULL if collection not found)
    collection_id = None
    async with db.execute(
        "SELECT id FROM collections WHERE name = ?", (response.collection,)
    ) as cursor:
        row = await cursor.fetchone()
    if row:
        collection_id = row[0]

    metadata_json = json.dumps(metadata) if metadata else "{}"

    await db.execute(
        "INSERT INTO queries "
        "(id, collection_id, query_text, query_type, search_mode, "
        "result_count, latency_ms, interface, metadata, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            query_id,
            collection_id,
            response.query,
            query_type,
            response.mode,
            response.total,
            response.latency_ms,
            interface,
            metadata_json,
            now,
        ),
    )
    await db.commit()
    return query_id
