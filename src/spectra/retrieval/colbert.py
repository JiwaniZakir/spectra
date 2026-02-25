"""ColBERT -- Late-interaction retrieval with token-level MaxSim scoring.

Implements the late-interaction paradigm from Khattab & Zaharia (2020)
where query and document are encoded independently at the token level
and scoring uses MaxSim (maximum similarity) per query token.
"""

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


class ColBERTConfig(RetrieverConfig):
    """Configuration for ColBERT-style late-interaction retrieval."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    query_max_tokens: int = Field(default=32, ge=1)
    doc_max_tokens: int = Field(default=180, ge=1)
    similarity: str = Field(default="cosine", pattern="^(cosine|dot)$")


class ColBERTRetriever(BaseRetriever):
    """ColBERT late-interaction retrieval.

    Unlike bi-encoder models that produce a single embedding per text,
    ColBERT retains token-level embeddings. The relevance score between
    a query *Q* and document *D* is computed as the sum of MaxSim scores:

        score(Q, D) = sum_{q in Q} max_{d in D} sim(q, d)

    This allows fine-grained, token-level matching while keeping document
    encoding offline.

    This implementation approximates ColBERT by splitting text into
    overlapping chunks of tokens and embedding each chunk independently
    using a sentence-transformer. The MaxSim scoring is computed over
    chunk embeddings.

    References
    ----------
    Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search
    via Contextualized Late Interaction over BERT", SIGIR 2020.

    Parameters
    ----------
    config:
        ColBERT configuration.
    """

    strategy_name = "colbert"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.LATE_INTERACTION}
    )

    def __init__(self, config: ColBERTConfig | None = None) -> None:
        self.config: ColBERTConfig = config or ColBERTConfig()
        super().__init__(self.config)
        self._embedding_model = EmbeddingModel(self.config.embedding)
        self._documents: list[Document] = []
        # Each document gets a matrix of token-level embeddings
        self._doc_token_embeddings: list[NDArray[np.float32]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Encode *documents* at the token level."""
        for doc in documents:
            chunks = self._split_into_token_chunks(doc.content, self.config.doc_max_tokens)
            if not chunks:
                chunks = [doc.content[:200] if doc.content else "empty"]
            embeddings = self._embedding_model.encode(chunks)
            self._doc_token_embeddings.append(embeddings)
            self._documents.append(doc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve using MaxSim late-interaction scoring."""
        k = self._effective_top_k(top_k)

        if not self._documents:
            return []

        # Encode query at the token level
        query_chunks = self._split_into_token_chunks(query, self.config.query_max_tokens)
        if not query_chunks:
            query_chunks = [query]
        query_embeddings = self._embedding_model.encode(query_chunks)

        # Score every document
        scores: list[float] = []
        for doc_embeddings in self._doc_token_embeddings:
            score = self._maxsim_score(query_embeddings, doc_embeddings)
            scores.append(score)

        scores_arr = np.array(scores)

        # Normalize to [0, 1]
        max_score = scores_arr.max()
        if max_score > 0:
            scores_arr = scores_arr / max_score

        k = min(k, len(self._documents))
        top_idx = np.argpartition(scores_arr, -k)[-k:]
        top_idx = top_idx[np.argsort(scores_arr[top_idx])[::-1]]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=float(scores_arr[idx]),
                    strategy=self.strategy_name,
                )
            )

        return self._filter_by_threshold(results)

    # ------------------------------------------------------------------
    # MaxSim computation
    # ------------------------------------------------------------------

    @staticmethod
    def _maxsim_score(
        query_embs: NDArray[np.float32],
        doc_embs: NDArray[np.float32],
    ) -> float:
        """Compute the ColBERT MaxSim score.

        For each query embedding, find the maximum cosine similarity with
        any document embedding, then sum across all query embeddings.

        Parameters
        ----------
        query_embs:
            Shape ``(n_query_tokens, dim)``.
        doc_embs:
            Shape ``(n_doc_tokens, dim)``.

        Returns
        -------
        float
            The MaxSim score.
        """
        # Similarity matrix: (n_query, n_doc)
        sim_matrix = query_embs @ doc_embs.T

        # MaxSim: for each query token, take max over doc tokens
        max_sims = sim_matrix.max(axis=1)

        # Sum over query tokens
        return float(max_sims.sum())

    # ------------------------------------------------------------------
    # Token-level splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_token_chunks(text: str, max_tokens: int) -> list[str]:
        """Split *text* into overlapping word-level chunks.

        A simple approximation: split on whitespace, group into windows of
        *max_tokens* words with 50% overlap.
        """
        words = text.split()
        if not words:
            return []

        stride = max(1, max_tokens // 2)
        chunks: list[str] = []
        for start in range(0, len(words), stride):
            chunk_words = words[start : start + max_tokens]
            chunks.append(" ".join(chunk_words))
            if start + max_tokens >= len(words):
                break
        return chunks
