"""RAG pipeline construction and execution.

A pipeline chains together chunking, embedding, vector storage, retrieval,
and optional reranking stages into an end-to-end RAG system.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

from spectra.chunking.fixed import FixedChunker, FixedChunkerConfig
from spectra.chunking.recursive import RecursiveChunker, RecursiveChunkerConfig
from spectra.chunking.semantic import SemanticChunker, SemanticChunkerConfig
from spectra.chunking.document_aware import DocumentAwareChunker, DocumentAwareChunkerConfig
from spectra.evaluation.metrics import EvaluationSample, MetricName, compute_all_metrics
from spectra.retrieval.base import BaseRetriever, Document, RetrievalResult


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    DOCUMENT_AWARE = "document_aware"


class PipelineConfig(BaseModel):
    """Configuration for a RAG pipeline."""

    name: str = "default"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = False
    rerank_top_k: int = Field(default=3, ge=1, le=50)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class PipelineResult(BaseModel):
    """Result of running a query through the pipeline."""

    query: str
    retrieved_documents: list[Document] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    latency_seconds: float = 0.0
    pipeline_config: PipelineConfig = Field(default_factory=PipelineConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGPipeline:
    """End-to-end RAG pipeline.

    Wires together chunking, retrieval, and evaluation into a single
    callable pipeline.

    Parameters
    ----------
    retriever:
        The retrieval strategy to use.
    config:
        Pipeline configuration.

    Example
    -------
    >>> from spectra.retrieval.dense import DenseRetriever
    >>> pipeline = RAGPipeline(DenseRetriever(), PipelineConfig(chunk_size=256))
    >>> pipeline.ingest(documents)
    >>> result = pipeline.query("What is RAG?")
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.retriever = retriever
        self._chunker = self._create_chunker()
        self._documents: list[Document] = []
        self._chunks: list[Document] = []

    def _create_chunker(self) -> Any:
        """Create the appropriate chunker based on config."""
        match self.config.chunking_strategy:
            case ChunkingStrategy.FIXED:
                return FixedChunker(
                    FixedChunkerConfig(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                )
            case ChunkingStrategy.SEMANTIC:
                return SemanticChunker(
                    SemanticChunkerConfig(
                        min_chunk_size=max(50, self.config.chunk_size // 4),
                        max_chunk_size=self.config.chunk_size,
                    )
                )
            case ChunkingStrategy.RECURSIVE:
                return RecursiveChunker(
                    RecursiveChunkerConfig(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                )
            case ChunkingStrategy.DOCUMENT_AWARE:
                return DocumentAwareChunker(
                    DocumentAwareChunkerConfig(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                    )
                )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: Sequence[Document]) -> int:
        """Ingest documents: chunk and index.

        Parameters
        ----------
        documents:
            Raw documents to process.

        Returns
        -------
        int
            Number of chunks created.
        """
        self._documents.extend(documents)
        chunks = self._chunker.chunk_many(list(documents))
        self._chunks.extend(chunks)
        self.retriever.add_documents(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int | None = None) -> PipelineResult:
        """Run a query through the full pipeline.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Override the configured top_k.

        Returns
        -------
        PipelineResult
            Retrieved documents with scores and latency.
        """
        k = top_k or self.config.top_k
        start = time.perf_counter()

        results = self.retriever.retrieve(query, top_k=k)

        elapsed = time.perf_counter() - start

        return PipelineResult(
            query=query,
            retrieved_documents=[r.document for r in results],
            scores=[r.score for r in results],
            latency_seconds=elapsed,
            pipeline_config=self.config,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        queries: Sequence[str],
        ground_truths: Sequence[str] | None = None,
        metrics: Sequence[MetricName] | None = None,
    ) -> dict[str, float]:
        """Evaluate the pipeline on a set of queries.

        Parameters
        ----------
        queries:
            Test queries.
        ground_truths:
            Optional ground-truth answers (one per query).
        metrics:
            Metrics to compute.

        Returns
        -------
        dict[str, float]
            Average score per metric.
        """
        samples: list[EvaluationSample] = []

        for i, q in enumerate(queries):
            result = self.query(q)
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            samples.append(
                EvaluationSample(
                    query=q,
                    answer=" ".join(d.content[:200] for d in result.retrieved_documents[:3]),
                    contexts=[d.content for d in result.retrieved_documents],
                    ground_truth=gt,
                    latency_seconds=result.latency_seconds,
                )
            )

        eval_result = compute_all_metrics(samples, metrics=metrics)
        return eval_result.scores

    @property
    def num_documents(self) -> int:
        """Number of source documents ingested."""
        return len(self._documents)

    @property
    def num_chunks(self) -> int:
        """Number of chunks in the index."""
        return len(self._chunks)
