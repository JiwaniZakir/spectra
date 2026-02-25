"""Vector store backends for persistent document storage and retrieval."""

from __future__ import annotations

from spectra.vectorstores.base import BaseVectorStore, VectorStoreConfig
from spectra.vectorstores.faiss_store import FAISSStore, FAISSConfig
from spectra.vectorstores.qdrant_store import QdrantStore, QdrantConfig
from spectra.vectorstores.pgvector_store import PgVectorStore, PgVectorConfig

__all__ = [
    "BaseVectorStore",
    "VectorStoreConfig",
    "FAISSStore",
    "FAISSConfig",
    "QdrantStore",
    "QdrantConfig",
    "PgVectorStore",
    "PgVectorConfig",
]
