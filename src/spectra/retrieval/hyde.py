"""HyDE -- Hypothetical Document Embeddings retrieval.

Implements the approach from Gao et al. (2022) where a language model
generates a hypothetical answer to the query, and the embedding of that
hypothetical document is used as the retrieval vector.
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
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.utils.embeddings import EmbeddingConfig, EmbeddingModel
from spectra.utils.llm import LLMClient, LLMConfig


_HYDE_PROMPT = (
    "Please write a passage that directly answers the following question. "
    "Write only the passage, no preamble.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


class HyDEConfig(RetrieverConfig):
    """Configuration for HyDE retrieval."""

    num_hypotheses: int = Field(
        default=1, ge=1, le=8,
        description="Number of hypothetical documents to generate and average.",
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    prompt_template: str = _HYDE_PROMPT


class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings (HyDE) retrieval.

    HyDE uses an LLM to generate a hypothetical document that answers the
    query, then embeds that document and searches the vector store. This
    bridges the query-document gap in embedding space.

    The approach averages embeddings from ``num_hypotheses`` generated
    documents for robustness.

    References
    ----------
    Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
    Labels", ACL 2023 (arXiv:2212.10496, 2022).

    Parameters
    ----------
    config:
        HyDE configuration.
    """

    strategy_name = "hyde"
    capabilities = frozenset({RetrieverCapability.DENSE, RetrieverCapability.GENERATIVE})

    def __init__(self, config: HyDEConfig | None = None) -> None:
        self.config: HyDEConfig = config or HyDEConfig()
        super().__init__(self.config)
        self._embedding_model = EmbeddingModel(self.config.embedding)
        self._llm = LLMClient(self.config.llm)
        self._inner = DenseRetriever(
            DenseRetrieverConfig(
                top_k=self.config.top_k,
                embedding=self.config.embedding,
            )
        )

    # ------------------------------------------------------------------
    # Indexing -- delegates to inner dense retriever
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index *documents* in the underlying dense retriever."""
        self._inner.add_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Generate hypothetical documents, embed, and retrieve."""
        k = self._effective_top_k(top_k)

        # Step 1: Generate hypothetical documents
        hypotheses = self._generate_hypotheses(query)

        # Step 2: Embed hypothetical docs and average
        hyp_embeddings = self._embedding_model.encode(hypotheses)
        avg_embedding = hyp_embeddings.mean(axis=0, keepdims=True).astype(np.float32)

        # Step 3: Search inner index with averaged hypothetical embedding
        if self._inner._matrix is None:
            return []

        scores, indices = self._inner._search(avg_embedding, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores, indices, strict=True):
            if idx < 0:
                continue
            results.append(
                RetrievalResult(
                    document=self._inner._documents[idx],
                    score=float(score),
                    strategy=self.strategy_name,
                    metadata={"hypotheses": hypotheses},
                )
            )

        return self._filter_by_threshold(results)

    def _generate_hypotheses(self, query: str) -> list[str]:
        """Generate one or more hypothetical answer documents."""
        hypotheses: list[str] = []
        for _ in range(self.config.num_hypotheses):
            prompt = self.config.prompt_template.format(query=query)
            response = self._llm.complete(prompt, temperature=0.7)
            hypotheses.append(response.content.strip())
        return hypotheses
