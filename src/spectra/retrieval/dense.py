"""Dense retrieval using bi-encoder embeddings with FAISS or NumPy fallback."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)
from spectra.utils.embeddings import EmbeddingConfig, EmbeddingModel


class DenseRetrieverConfig(RetrieverConfig):
    """Configuration for the dense bi-encoder retriever."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    use_faiss: bool = True
    nprobe: int = 8


class DenseRetriever(BaseRetriever):
    """Bi-encoder dense retrieval with optional FAISS acceleration.

    Documents are encoded into dense vectors using a sentence-transformer
    model.  At query time the query is encoded and nearest neighbours are
    found via either FAISS (if available and enabled) or brute-force cosine
    similarity over a NumPy matrix.

    References
    ----------
    Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question
    Answering", EMNLP 2020.

    Parameters
    ----------
    config:
        Dense retriever configuration.
    """

    strategy_name = "dense"
    capabilities = frozenset({RetrieverCapability.DENSE})

    def __init__(self, config: DenseRetrieverConfig | None = None) -> None:
        self.config: DenseRetrieverConfig = config or DenseRetrieverConfig()
        super().__init__(self.config)
        self._embedding_model = EmbeddingModel(self.config.embedding)
        self._documents: list[Document] = []
        self._index: object | None = None  # FAISS index
        self._matrix: NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Encode and index *documents*."""
        if not documents:
            return

        docs = list(documents)
        texts = [d.content for d in docs]
        embeddings = self._embedding_model.encode(texts)

        for doc, emb in zip(docs, embeddings, strict=True):
            doc.embedding = emb.tolist()

        self._documents.extend(docs)
        self._build_index(embeddings)

    def _build_index(self, new_embeddings: NDArray[np.float32]) -> None:
        """Rebuild or extend the search index with *new_embeddings*."""
        if self._matrix is not None:
            self._matrix = np.vstack([self._matrix, new_embeddings])
        else:
            self._matrix = new_embeddings

        if self.config.use_faiss:
            try:
                import faiss

                dim = self._matrix.shape[1]
                self._index = faiss.IndexFlatIP(dim)
                self._index.add(self._matrix)  # type: ignore[union-attr]
                return
            except ImportError:
                pass

        # Fallback: no FAISS -- use raw matrix for brute-force search
        self._index = None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve nearest neighbours for *query*."""
        k = self._effective_top_k(top_k)

        if self._matrix is None or len(self._documents) == 0:
            return []

        query_vec = self._embedding_model.encode_query(query).reshape(1, -1)
        scores, indices = self._search(query_vec, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores, indices, strict=True):
            if idx < 0:
                continue
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=float(score),
                    strategy=self.strategy_name,
                )
            )

        return self._filter_by_threshold(results)

    def _search(
        self, query_vec: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
        """Return (scores, indices) for *query_vec*."""
        k = min(k, len(self._documents))

        if self._index is not None:
            # FAISS path
            scores, indices = self._index.search(query_vec, k)  # type: ignore[union-attr]
            return scores[0], indices[0]

        # Brute-force cosine similarity
        assert self._matrix is not None
        sims = (self._matrix @ query_vec.T).squeeze(-1)
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return sims[top_idx], top_idx
