"""Async Qdrant SDK wrapper — collections, upsert, dense/sparse/hybrid search."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    MatchValue,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag_kb.core.config import QdrantConfig, get_config
from rag_kb.core.errors import QdrantCollectionError, QdrantError
from rag_kb.models.search import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class QdrantManager:
    """Async wrapper around the Qdrant SDK with collection prefixing and hybrid search.

    Usage::

        async with QdrantManager() as qm:
            await qm.create_collection("docs")
            await qm.upsert_points("docs", points)
            results = await qm.search_hybrid("docs", dense_vec, sparse_vec)
    """

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self._config = config or get_config().qdrant
        self._client = AsyncQdrantClient(
            host=self._config.host,
            port=self._config.port,
            grpc_port=self._config.grpc_port,
            timeout=60,
        )

    async def __aenter__(self) -> QdrantManager:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    # --- Collection prefix ---

    def _prefixed(self, name: str) -> str:
        """Add collection prefix if not already present."""
        prefix = self._config.collection_prefix
        if name.startswith(prefix):
            return name
        return f"{prefix}{name}"

    # --- Collection management ---

    async def create_collection(self, name: str, dense_dim: int = 768) -> None:
        """Create a collection with dense + sparse vector config. Idempotent."""
        prefixed = self._prefixed(name)
        try:
            if await self._client.collection_exists(prefixed):
                logger.debug("Collection %s already exists, skipping create", prefixed)
                return

            await self._client.create_collection(
                collection_name=prefixed,
                vectors_config={
                    "dense": VectorParams(
                        size=dense_dim,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=self._config.hnsw_m,
                            ef_construct=self._config.hnsw_ef_construct,
                        ),
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(),
                },
            )
            logger.info("Created collection %s (dense_dim=%d)", prefixed, dense_dim)
        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to create collection {prefixed}: {exc}", cause=exc
            ) from exc

    async def delete_collection(self, name: str) -> None:
        """Delete a collection. Idempotent — no-op if not exists."""
        prefixed = self._prefixed(name)
        try:
            if not await self._client.collection_exists(prefixed):
                return
            await self._client.delete_collection(prefixed)
            logger.info("Deleted collection %s", prefixed)
        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to delete collection {prefixed}: {exc}", cause=exc
            ) from exc

    async def list_collections(self) -> list[str]:
        """List collections matching our prefix."""
        try:
            response = await self._client.get_collections()
            prefix = self._config.collection_prefix
            return [
                c.name for c in response.collections if c.name.startswith(prefix)
            ]
        except Exception as exc:
            raise QdrantError(
                f"Failed to list collections: {exc}", cause=exc
            ) from exc

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        prefixed = self._prefixed(name)
        try:
            return await self._client.collection_exists(prefixed)
        except Exception as exc:
            raise QdrantError(
                f"Failed to check collection {prefixed}: {exc}", cause=exc
            ) from exc

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return collection metadata as a dict."""
        prefixed = self._prefixed(name)
        try:
            info = await self._client.get_collection(prefixed)
            return {
                "name": prefixed,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            }
        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to get info for {prefixed}: {exc}", cause=exc
            ) from exc

    # --- Points ---

    async def upsert_points(
        self,
        collection: str,
        points: list[PointStruct],
        batch_size: int = 100,
    ) -> None:
        """Upsert points in batches for durability."""
        prefixed = self._prefixed(collection)
        try:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                await self._client.upsert(
                    collection_name=prefixed,
                    points=batch,
                    wait=True,
                )
            logger.debug("Upserted %d points into %s", len(points), prefixed)
        except Exception as exc:
            raise QdrantError(
                f"Failed to upsert into {prefixed}: {exc}", cause=exc
            ) from exc

    async def delete_points(self, collection: str, point_ids: list[str | int]) -> None:
        """Delete specific points by ID."""
        prefixed = self._prefixed(collection)
        try:
            await self._client.delete(
                collection_name=prefixed,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception as exc:
            raise QdrantError(
                f"Failed to delete points from {prefixed}: {exc}", cause=exc
            ) from exc

    async def delete_points_by_filter(
        self, collection: str, filters: dict[str, Any]
    ) -> None:
        """Delete points matching payload filter (e.g. file_path)."""
        from qdrant_client.models import FilterSelector

        prefixed = self._prefixed(collection)
        try:
            qf = self._build_filter(filters)
            await self._client.delete(
                collection_name=prefixed,
                points_selector=FilterSelector(filter=qf),
            )
        except Exception as exc:
            raise QdrantError(
                f"Failed to delete by filter from {prefixed}: {exc}", cause=exc
            ) from exc

    # --- Search ---

    async def search_dense(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Dense vector search using the 'dense' named vector."""
        prefixed = self._prefixed(collection)
        try:
            qf = self._build_filter(filters) if filters else None
            results = await self._client.query_points(
                collection_name=prefixed,
                query=vector,
                using="dense",
                limit=top_k,
                with_payload=True,
                query_filter=qf,
            )
            return self._to_search_response(results, "dense")
        except Exception as exc:
            raise QdrantError(
                f"Dense search failed on {prefixed}: {exc}", cause=exc
            ) from exc

    async def search_sparse(
        self,
        collection: str,
        sparse_vector: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Sparse vector search using the 'sparse' named vector."""
        prefixed = self._prefixed(collection)
        try:
            qf = self._build_filter(filters) if filters else None
            results = await self._client.query_points(
                collection_name=prefixed,
                query=sparse_vector,
                using="sparse",
                limit=top_k,
                with_payload=True,
                query_filter=qf,
            )
            return self._to_search_response(results, "sparse")
        except Exception as exc:
            raise QdrantError(
                f"Sparse search failed on {prefixed}: {exc}", cause=exc
            ) from exc

    async def search_hybrid(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Hybrid search using Qdrant server-side RRF fusion."""
        prefixed = self._prefixed(collection)
        prefetch_limit = min(top_k * 2, 100)
        try:
            qf = self._build_filter(filters) if filters else None
            results = await self._client.query_points(
                collection_name=prefixed,
                prefetch=[
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=prefetch_limit,
                        filter=qf,
                    ),
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=prefetch_limit,
                        filter=qf,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return self._to_search_response(results, "hybrid")
        except Exception as exc:
            raise QdrantError(
                f"Hybrid search failed on {prefixed}: {exc}", cause=exc
            ) from exc

    # --- Internal helpers ---

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a simple key-value dict to a Qdrant Filter."""
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
        ]
        return Filter(must=conditions)

    @staticmethod
    def _to_search_response(results: Any, mode: str) -> SearchResponse:
        """Convert Qdrant query response to our SearchResponse model."""
        items = []
        for point in results.points:
            items.append(
                SearchResult(
                    id=point.id,
                    score=point.score if point.score is not None else 0.0,
                    payload=dict(point.payload) if point.payload else {},
                )
            )
        return SearchResponse(results=items, total=len(items), search_mode=mode)
