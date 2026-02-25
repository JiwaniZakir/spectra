"""Tests for pipeline construction and optimization."""

from __future__ import annotations

import pytest

from spectra.optimization.pipeline import (
    ChunkingStrategy,
    PipelineConfig,
    PipelineResult,
    RAGPipeline,
)
from spectra.retrieval.base import Document
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.sparse import BM25Retriever


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(id="1", content="Python is a high-level programming language."),
        Document(id="2", content="Java is a statically typed programming language."),
        Document(id="3", content="Rust focuses on safety and performance."),
    ]


class TestPipelineConfig:
    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.top_k == 5
        assert config.chunking_strategy == ChunkingStrategy.RECURSIVE

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            name="custom",
            chunking_strategy=ChunkingStrategy.FIXED,
            chunk_size=256,
            top_k=3,
        )
        assert config.name == "custom"
        assert config.chunk_size == 256


class TestRAGPipeline:
    def test_ingest_and_query_bm25(self, documents: list[Document]) -> None:
        pipeline = RAGPipeline(
            BM25Retriever(),
            PipelineConfig(name="test", chunk_size=256, top_k=2),
        )
        num_chunks = pipeline.ingest(documents)
        assert num_chunks > 0
        assert pipeline.num_documents == 3

        result = pipeline.query("programming language")
        assert isinstance(result, PipelineResult)
        assert len(result.retrieved_documents) > 0
        assert result.latency_seconds > 0

    @pytest.mark.slow
    def test_ingest_and_query_dense(self, documents: list[Document]) -> None:
        pipeline = RAGPipeline(
            DenseRetriever(DenseRetrieverConfig(use_faiss=False)),
            PipelineConfig(name="test_dense", chunk_size=256, top_k=2),
        )
        num_chunks = pipeline.ingest(documents)
        assert num_chunks > 0

        result = pipeline.query("programming language")
        assert len(result.retrieved_documents) > 0

    def test_chunking_strategies(self, documents: list[Document]) -> None:
        for strategy in ChunkingStrategy:
            pipeline = RAGPipeline(
                BM25Retriever(),
                PipelineConfig(
                    name=f"test_{strategy.value}",
                    chunking_strategy=strategy,
                    chunk_size=256,
                    top_k=2,
                ),
            )
            num_chunks = pipeline.ingest(documents)
            assert num_chunks > 0, f"Strategy {strategy} produced no chunks"

    def test_evaluate(self, documents: list[Document]) -> None:
        pipeline = RAGPipeline(
            BM25Retriever(),
            PipelineConfig(name="eval_test", chunk_size=256, top_k=2),
        )
        pipeline.ingest(documents)

        scores = pipeline.evaluate(
            ["What is Python?"],
            ground_truths=["Python is a programming language."],
        )
        assert isinstance(scores, dict)
        assert len(scores) > 0
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_pipeline_result_structure(self, documents: list[Document]) -> None:
        pipeline = RAGPipeline(
            BM25Retriever(),
            PipelineConfig(name="struct_test", chunk_size=256, top_k=2),
        )
        pipeline.ingest(documents)

        result = pipeline.query("Python")
        assert result.query == "Python"
        assert len(result.scores) == len(result.retrieved_documents)
        assert result.pipeline_config.name == "struct_test"
