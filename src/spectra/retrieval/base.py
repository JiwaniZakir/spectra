"""Abstract base classes and shared data models for all retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single document or chunk with content, metadata, and optional embedding.

    Attributes
    ----------
    id:
        Unique identifier for the document.
    content:
        The raw text content.
    metadata:
        Arbitrary key-value metadata (source, page number, etc.).
    embedding:
        Optional pre-computed dense vector.
    """

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    model_config = {"frozen": False}


class RetrievalResult(BaseModel):
    """A scored document returned by a retriever.

    Attributes
    ----------
    document:
        The retrieved document.
    score:
        Relevance score (higher is better, normalized to [0, 1] where possible).
    strategy:
        Name of the retrieval strategy that produced this result.
    metadata:
        Extra information from the retrieval process (e.g. reasoning traces).
    """

    document: Document
    score: float
    strategy: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieverConfig(BaseModel):
    """Base configuration shared by all retrievers."""

    top_k: int = Field(default=10, ge=1, le=1000)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}


class RetrieverCapability(str, Enum):
    """Capabilities that a retriever may advertise."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GENERATIVE = "generative"
    ITERATIVE = "iterative"
    GRAPH = "graph"
    RERANKING = "reranking"
    LATE_INTERACTION = "late_interaction"


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies.

    Every retriever must implement :meth:`retrieve` and :meth:`add_documents`.
    Async variants are provided with default implementations that delegate to
    the sync versions, but subclasses may override for true async behavior.

    Parameters
    ----------
    config:
        Strategy-specific configuration.
    """

    strategy_name: str = "base"
    capabilities: frozenset[RetrieverCapability] = frozenset()

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve documents relevant to *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Override the configured ``top_k``.

        Returns
        -------
        list[RetrievalResult]
            Scored results ordered by descending relevance.
        """
        ...

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Ingest documents into the retriever's index.

        Parameters
        ----------
        documents:
            Documents to index.
        """
        ...

    # ------------------------------------------------------------------
    # Default async interface (delegates to sync)
    # ------------------------------------------------------------------

    async def aretrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Async variant of :meth:`retrieve`."""
        return self.retrieve(query, top_k=top_k)

    async def aadd_documents(self, documents: Sequence[Document]) -> None:
        """Async variant of :meth:`add_documents`."""
        self.add_documents(documents)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_top_k(self, top_k: int | None) -> int:
        return top_k if top_k is not None else self.config.top_k

    def _filter_by_threshold(
        self, results: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if self.config.score_threshold > 0:
            return [r for r in results if r.score >= self.config.score_threshold]
        return results

    def __repr__(self) -> str:
        return f"{type(self).__name__}(strategy={self.strategy_name!r})"
