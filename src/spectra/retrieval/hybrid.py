"""Hybrid retrieval combining dense and sparse signals via Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from pydantic import Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)


class HybridConfig(RetrieverConfig):
    """Configuration for hybrid retrieval."""

    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Reciprocal Rank Fusion constant (higher = more weight to lower ranks).",
    )
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description="Multiply top_k by this factor when fetching from sub-retrievers.",
    )


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF) over multiple sub-retrievers.

    Combines results from dense and sparse (or any set of) retrievers using
    the RRF formula::

        score(d) = sum_r  w_r / (k + rank_r(d))

    where *k* is the RRF constant and *w_r* is the weight for retriever *r*.

    References
    ----------
    Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods", SIGIR 2009.

    Parameters
    ----------
    retrievers:
        A sequence of ``(retriever, weight)`` pairs.
    config:
        Hybrid retrieval configuration.
    """

    strategy_name = "hybrid"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.SPARSE, RetrieverCapability.HYBRID}
    )

    def __init__(
        self,
        retrievers: Sequence[tuple[BaseRetriever, float]] | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self.config: HybridConfig = config or HybridConfig()
        super().__init__(self.config)
        self._retrievers: list[tuple[BaseRetriever, float]] = list(retrievers or [])

    def add_retriever(self, retriever: BaseRetriever, weight: float = 1.0) -> None:
        """Register a sub-retriever with the given fusion weight."""
        self._retrievers.append((retriever, weight))

    # ------------------------------------------------------------------
    # Indexing -- delegates to all sub-retrievers
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Add *documents* to every sub-retriever."""
        for retriever, _ in self._retrievers:
            retriever.add_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve using all sub-retrievers and fuse with RRF."""
        k = self._effective_top_k(top_k)
        fetch_k = k * self.config.fetch_multiplier

        ranked_lists: list[tuple[list[RetrievalResult], float]] = []
        for retriever, weight in self._retrievers:
            results = retriever.retrieve(query, top_k=fetch_k)
            ranked_lists.append((results, weight))

        fused = self._reciprocal_rank_fusion(ranked_lists, rrf_k=self.config.rrf_k)

        # Sort by fused score descending and take top-k
        fused_sorted = sorted(fused.values(), key=lambda r: r.score, reverse=True)[:k]
        return self._filter_by_threshold(fused_sorted)

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[tuple[list[RetrievalResult], float]],
        rrf_k: int = 60,
    ) -> dict[str, RetrievalResult]:
        """Compute RRF scores across multiple ranked lists.

        Parameters
        ----------
        ranked_lists:
            Each element is ``(results, weight)`` where results are ordered
            by descending relevance.
        rrf_k:
            The RRF smoothing constant.

        Returns
        -------
        dict mapping document ID to the best-scored ``RetrievalResult``.
        """
        scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, RetrievalResult] = {}

        for results, weight in ranked_lists:
            for rank, result in enumerate(results, start=1):
                doc_id = result.document.id
                scores[doc_id] += weight / (rrf_k + rank)

                if doc_id not in best_result or result.score > best_result[doc_id].score:
                    best_result[doc_id] = result

        # Normalize
        max_score = max(scores.values()) if scores else 1.0

        fused: dict[str, RetrievalResult] = {}
        for doc_id, score in scores.items():
            original = best_result[doc_id]
            fused[doc_id] = RetrievalResult(
                document=original.document,
                score=score / max_score if max_score > 0 else 0.0,
                strategy="hybrid",
                metadata={
                    "rrf_raw_score": score,
                    "original_strategy": original.strategy,
                    "original_score": original.score,
                },
            )
        return fused
