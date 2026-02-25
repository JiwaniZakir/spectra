"""Cross-encoder reranker for re-scoring retrieved documents.

Uses a cross-encoder model that jointly encodes the query and document
to produce a more accurate relevance score than bi-encoder retrieval alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pydantic import Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)


class RerankerConfig(RetrieverConfig):
    """Configuration for the cross-encoder reranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512


class Reranker(BaseRetriever):
    """Cross-encoder reranker.

    Takes a set of candidate documents (from any first-stage retriever)
    and re-scores them using a cross-encoder that jointly attends to
    the query and document.  Cross-encoders are more accurate than
    bi-encoders but too slow for exhaustive search, making them ideal
    as a second stage.

    Parameters
    ----------
    first_stage:
        The retriever that produces initial candidates.
    config:
        Reranker configuration.
    """

    strategy_name = "reranker"
    capabilities = frozenset({RetrieverCapability.RERANKING})

    def __init__(
        self,
        first_stage: BaseRetriever | None = None,
        config: RerankerConfig | None = None,
    ) -> None:
        self.config: RerankerConfig = config or RerankerConfig()
        super().__init__(self.config)
        self._first_stage = first_stage
        self._cross_encoder: object | None = None

    def _load_cross_encoder(self) -> object:
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "pip install sentence-transformers"
                ) from exc
            self._cross_encoder = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
            )
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Indexing -- delegates to first stage
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add *documents* to the first-stage retriever."""
        if self._first_stage is not None:
            self._first_stage.add_documents(documents)

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve from the first stage, then rerank with the cross-encoder."""
        k = self._effective_top_k(top_k)

        if self._first_stage is None:
            return []

        # Get more candidates than needed, then rerank
        candidates = self._first_stage.retrieve(query, top_k=k * 3)
        if not candidates:
            return []

        return self.rerank(query, candidates, top_k=k)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank an existing list of candidates.

        Parameters
        ----------
        query:
            The query string.
        candidates:
            Candidate results from a first-stage retriever.
        top_k:
            Number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Reranked results ordered by cross-encoder score.
        """
        k = self._effective_top_k(top_k)
        if not candidates:
            return []

        cross_encoder = self._load_cross_encoder()

        # Build query-document pairs
        pairs = [(query, c.document.content) for c in candidates]

        # Score in batches
        scores: list[float] = []
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i : i + self.config.batch_size]
            batch_scores = cross_encoder.predict(batch)  # type: ignore[union-attr]
            if isinstance(batch_scores, np.ndarray):
                scores.extend(batch_scores.tolist())
            else:
                scores.extend(batch_scores)

        # Normalize scores to [0, 1] using sigmoid
        scores_arr = np.array(scores)
        normalized = 1.0 / (1.0 + np.exp(-scores_arr))

        # Pair candidates with scores and sort
        scored = list(zip(candidates, normalized, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievalResult] = []
        for candidate, score in scored[:k]:
            results.append(
                RetrievalResult(
                    document=candidate.document,
                    score=float(score),
                    strategy=self.strategy_name,
                    metadata={
                        "original_score": candidate.score,
                        "original_strategy": candidate.strategy,
                        "cross_encoder_raw": float(scores_arr[candidates.index(candidate)]),
                    },
                )
            )

        return self._filter_by_threshold(results)
