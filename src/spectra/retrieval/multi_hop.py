"""Multi-hop iterative retrieval for complex, multi-step questions.

At each hop the model reads the retrieved context so far, generates a
follow-up query, retrieves more documents, and accumulates evidence until
the question can be answered or a maximum number of hops is reached.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pydantic import Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.utils.embeddings import EmbeddingConfig
from spectra.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class MultiHopConfig(RetrieverConfig):
    """Configuration for multi-hop retrieval."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_hops: int = Field(default=3, ge=1, le=10)
    docs_per_hop: int = Field(default=3, ge=1, le=20)
    stop_on_answer: bool = True


class MultiHopRetriever(BaseRetriever):
    """Iterative multi-hop retriever for compositional questions.

    Multi-hop retrieval decomposes complex questions into a series of
    retrieval steps.  At each hop:

    1. The LLM examines the question and the evidence gathered so far.
    2. It generates a focused sub-query targeting missing information.
    3. The sub-query is used to retrieve additional passages.
    4. The LLM decides if enough evidence has been gathered.

    This is inspired by IRCoT (Trivedi et al., 2023) and multi-step
    retriever-reader architectures.

    Parameters
    ----------
    config:
        Multi-hop retrieval configuration.
    """

    strategy_name = "multi_hop"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.ITERATIVE, RetrieverCapability.GENERATIVE}
    )

    def __init__(self, config: MultiHopConfig | None = None) -> None:
        self.config: MultiHopConfig = config or MultiHopConfig()
        super().__init__(self.config)
        self._llm = LLMClient(self.config.llm)
        self._inner = DenseRetriever(
            DenseRetrieverConfig(
                top_k=self.config.docs_per_hop,
                embedding=self.config.embedding,
            )
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index *documents* in the underlying dense retriever."""
        self._inner.add_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Perform multi-hop iterative retrieval."""
        k = self._effective_top_k(top_k)

        all_results: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        chain_of_queries: list[str] = [query]

        for hop in range(self.config.max_hops):
            current_query = chain_of_queries[-1]
            logger.debug("Multi-hop %d: query=%r", hop, current_query)

            candidates = self._inner.retrieve(current_query, top_k=self.config.docs_per_hop)

            new_results: list[RetrievalResult] = []
            for result in candidates:
                if result.document.id in seen_ids:
                    continue
                seen_ids.add(result.document.id)
                new_results.append(
                    RetrievalResult(
                        document=result.document,
                        score=result.score * (1.0 - 0.1 * hop),  # decay later hops slightly
                        strategy=self.strategy_name,
                        metadata={"hop": hop, "sub_query": current_query},
                    )
                )
            all_results.extend(new_results)

            # Check if we should stop
            if self.config.stop_on_answer and self._has_sufficient_evidence(query, all_results):
                logger.debug("Multi-hop: sufficient evidence found at hop %d", hop)
                break

            # Generate next sub-query
            if hop < self.config.max_hops - 1:
                next_query = self._generate_sub_query(query, all_results)
                chain_of_queries.append(next_query)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return self._filter_by_threshold(all_results[:k])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_sub_query(
        self, original_query: str, results: list[RetrievalResult]
    ) -> str:
        """Generate a follow-up sub-query targeting missing information."""
        evidence = "\n".join(
            f"[{i+1}] {r.document.content[:300]}"
            for i, r in enumerate(results[-5:])
        )
        prompt = (
            "You are decomposing a complex question into retrieval steps.\n\n"
            f"Original question: {original_query}\n\n"
            f"Evidence gathered so far:\n{evidence}\n\n"
            "What specific piece of information is still missing? Generate a "
            "focused search query to find it. Output only the query."
        )
        response = self._llm.complete(prompt, max_tokens=100)
        return response.content.strip() or original_query

    def _has_sufficient_evidence(
        self, query: str, results: list[RetrievalResult]
    ) -> bool:
        """Ask the LLM whether enough evidence has been gathered."""
        if len(results) < 2:
            return False

        evidence = "\n".join(
            f"[{i+1}] {r.document.content[:200]}"
            for i, r in enumerate(results[-6:])
        )
        prompt = (
            "Given the following question and evidence passages, can the "
            "question be fully answered? Respond with exactly YES or NO.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "Answer:"
        )
        response = self._llm.complete(prompt, max_tokens=8)
        return "yes" in response.content.strip().lower()
