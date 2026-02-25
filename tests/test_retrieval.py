"""Tests for retrieval strategies."""

from __future__ import annotations

import pytest

from spectra.retrieval.base import Document, RetrievalResult
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.sparse import BM25Retriever, BM25Config
from spectra.retrieval.hybrid import HybridRetriever, HybridConfig


# ---- Fixtures ----

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(id="1", content="Python is a programming language created by Guido van Rossum."),
        Document(id="2", content="Machine learning uses algorithms to learn from data."),
        Document(id="3", content="Natural language processing deals with text and speech."),
        Document(id="4", content="Deep learning is a subset of machine learning using neural networks."),
        Document(id="5", content="Information retrieval helps find relevant documents from a corpus."),
    ]


# ---- BM25 Tests ----

class TestBM25Retriever:
    def test_add_and_retrieve(self, sample_documents: list[Document]) -> None:
        retriever = BM25Retriever(BM25Config(top_k=3))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("machine learning algorithms")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_relevance_ordering(self, sample_documents: list[Document]) -> None:
        retriever = BM25Retriever(BM25Config(top_k=5))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("machine learning")
        assert len(results) > 0
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_name(self) -> None:
        retriever = BM25Retriever()
        assert retriever.strategy_name == "bm25"

    def test_empty_corpus(self) -> None:
        retriever = BM25Retriever()
        results = retriever.retrieve("anything")
        assert results == []

    def test_score_threshold(self, sample_documents: list[Document]) -> None:
        retriever = BM25Retriever(BM25Config(top_k=5, score_threshold=0.5))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("Python programming language Guido")
        assert all(r.score >= 0.5 for r in results)

    def test_custom_parameters(self, sample_documents: list[Document]) -> None:
        retriever = BM25Retriever(BM25Config(top_k=2, k1=2.0, b=0.5))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("neural networks deep learning")
        assert len(results) <= 2


class TestDenseRetriever:
    @pytest.mark.slow
    def test_add_and_retrieve(self, sample_documents: list[Document]) -> None:
        retriever = DenseRetriever(DenseRetrieverConfig(top_k=3, use_faiss=False))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("programming language")
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.slow
    def test_numpy_fallback(self, sample_documents: list[Document]) -> None:
        retriever = DenseRetriever(DenseRetrieverConfig(top_k=3, use_faiss=False))
        retriever.add_documents(sample_documents)

        results = retriever.retrieve("machine learning")
        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestHybridRetriever:
    def test_rrf_fusion(self, sample_documents: list[Document]) -> None:
        dense = BM25Retriever(BM25Config(top_k=5))
        sparse = BM25Retriever(BM25Config(top_k=5, k1=2.0))

        hybrid = HybridRetriever(
            retrievers=[(dense, 0.5), (sparse, 0.5)],
            config=HybridConfig(top_k=3),
        )
        hybrid.add_documents(sample_documents)

        results = hybrid.retrieve("machine learning algorithms")
        assert len(results) > 0
        assert all(r.strategy == "hybrid" for r in results)

    def test_empty_retrievers(self) -> None:
        hybrid = HybridRetriever(config=HybridConfig(top_k=3))
        results = hybrid.retrieve("anything")
        assert results == []

    def test_rrf_scores_normalized(self, sample_documents: list[Document]) -> None:
        r1 = BM25Retriever(BM25Config(top_k=5))
        r2 = BM25Retriever(BM25Config(top_k=5))

        hybrid = HybridRetriever(
            retrievers=[(r1, 1.0), (r2, 1.0)],
            config=HybridConfig(top_k=5),
        )
        hybrid.add_documents(sample_documents)

        results = hybrid.retrieve("information retrieval")
        if results:
            assert results[0].score <= 1.0
            assert results[0].score >= 0.0
