"""Qdrant vector store backend.

Connects to a Qdrant instance (local or cloud) for persistent vector
storage with rich metadata filtering.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from spectra.vectorstores.base import BaseVectorStore, SearchResult, VectorStoreConfig

logger = logging.getLogger(__name__)


class QdrantConfig(VectorStoreConfig):
    """Configuration for the Qdrant vector store."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    prefer_grpc: bool = False
    on_disk: bool = False
    in_memory: bool = True  # For testing; use url for production


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store.

    Supports both in-memory and persistent Qdrant instances with
    metadata filtering.

    Parameters
    ----------
    config:
        Qdrant store configuration.

    Note
    ----
    Requires ``qdrant-client``: ``pip install qdrant-client``.
    """

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config: QdrantConfig = config or QdrantConfig()
        super().__init__(self.config)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            ) from exc

        self._qdrant = __import__("qdrant_client")
        self._models = self._qdrant.models

        if self.config.in_memory:
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                prefer_grpc=self.config.prefer_grpc,
            )

        # Map metric names
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "l2": Distance.EUCLID,
        }
        distance = distance_map.get(self.config.metric, Distance.COSINE)

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if self.config.collection_name not in collections:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.dimension,
                    distance=distance,
                    on_disk=self.config.on_disk,
                ),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the Qdrant collection."""
        from qdrant_client.models import PointStruct

        points = []
        for i, doc_id in enumerate(ids):
            payload = metadatas[i] if metadatas and i < len(metadatas) else {}
            points.append(
                PointStruct(
                    id=self._to_qdrant_id(doc_id),
                    vector=vectors[i].tolist(),
                    payload={**payload, "_spectra_id": doc_id},
                )
            )

        # Upsert in batches
        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )

    def search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbours in Qdrant."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        hits = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
        )

        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    id=payload.get("_spectra_id", str(hit.id)),
                    score=float(hit.score),
                    metadata={k: v for k, v in payload.items() if k != "_spectra_id"},
                )
            )
        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by ID from Qdrant."""
        from qdrant_client.models import PointIdsList

        qdrant_ids = [self._to_qdrant_id(doc_id) for doc_id in ids]
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        info = self._client.get_collection(self.config.collection_name)
        return info.points_count or 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_qdrant_id(doc_id: str) -> str:
        """Convert a document ID to a Qdrant-compatible UUID string."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))
