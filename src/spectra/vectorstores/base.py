"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from spectra.retrieval.base import Document


class VectorStoreConfig(BaseModel):
    """Base configuration for all vector stores."""

    collection_name: str = "spectra_default"
    dimension: int = Field(default=384, ge=1)
    metric: str = Field(default="cosine", pattern="^(cosine|dot|l2)$")

    model_config = {"frozen": True}


class SearchResult(BaseModel):
    """A single search result from the vector store."""

    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends.

    All vector stores must support adding vectors, searching by vector,
    and deletion.  Backends may support additional features like
    filtering and persistence.
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the store.

        Parameters
        ----------
        ids:
            Unique identifiers for each vector.
        vectors:
            Dense vectors of shape ``(n, dimension)``.
        metadatas:
            Optional metadata dicts, one per vector.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbours.

        Parameters
        ----------
        query_vector:
            Query vector of shape ``(dimension,)``.
        top_k:
            Number of results to return.
        filter_metadata:
            Optional metadata filter.
        """
        ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        ...

    def add_documents(
        self,
        documents: Sequence[Document],
        vectors: NDArray[np.float32],
    ) -> None:
        """Convenience method to add documents with their embeddings."""
        ids = [doc.id for doc in documents]
        metadatas = [
            {**doc.metadata, "content": doc.content} for doc in documents
        ]
        self.add(ids, vectors, metadatas)
