"""Self-RAG -- Self-Reflective Retrieval-Augmented Generation.

Implements the framework from Asai et al. (2023) where the model learns
to retrieve, generate, and critique its own output via reflection tokens.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, Field

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


class ReflectionToken(str, Enum):
    """Reflection tokens as defined in Self-RAG."""

    RETRIEVE = "Retrieve"
    ISREL = "IsRel"
    ISSUP = "IsSup"
    ISUSE = "IsUse"


class CritiqueResult(BaseModel):
    """Result of a self-critique step."""

    is_relevant: bool = True
    is_supported: bool = True
    is_useful: bool = True
    confidence: float = 0.0
    reasoning: str = ""


class SelfRAGConfig(RetrieverConfig):
    """Configuration for Self-RAG retrieval."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_iterations: int = Field(default=3, ge=1, le=10)
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    support_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class SelfRAGRetriever(BaseRetriever):
    """Self-RAG retrieval with reflection-driven adaptive retrieval.

    Self-RAG trains an LM to generate *reflection tokens* that control
    whether to retrieve, whether retrieved passages are relevant, whether
    the generation is supported, and whether it is useful.

    This implementation approximates the reflection mechanism via prompted
    critique: after each retrieval-generation step, the LLM evaluates its
    own output and decides whether to retrieve more context.

    References
    ----------
    Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique
    through Self-Reflection", ICLR 2024 (arXiv:2310.11511, 2023).

    Parameters
    ----------
    config:
        Self-RAG configuration.
    """

    strategy_name = "self_rag"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.GENERATIVE, RetrieverCapability.ITERATIVE}
    )

    def __init__(self, config: SelfRAGConfig | None = None) -> None:
        self.config: SelfRAGConfig = config or SelfRAGConfig()
        super().__init__(self.config)
        self._llm = LLMClient(self.config.llm)
        self._inner = DenseRetriever(
            DenseRetrieverConfig(
                top_k=self.config.top_k,
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
    # Retrieval with self-reflection
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Self-reflective retrieval: retrieve, critique, optionally re-retrieve."""
        k = self._effective_top_k(top_k)

        all_results: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        refined_query = query

        for iteration in range(self.config.max_iterations):
            # Step 1: Decide whether retrieval is needed
            if iteration > 0 and not self._should_retrieve(query, all_results):
                logger.debug("Self-RAG: stopping retrieval at iteration %d", iteration)
                break

            # Step 2: Retrieve
            candidates = self._inner.retrieve(refined_query, top_k=k * 2)

            # Step 3: Critique each candidate
            for result in candidates:
                if result.document.id in seen_ids:
                    continue
                seen_ids.add(result.document.id)

                critique = self._critique_passage(query, result.document.content)

                if critique.is_relevant and critique.is_supported:
                    adjusted_score = result.score * critique.confidence
                    all_results.append(
                        RetrievalResult(
                            document=result.document,
                            score=adjusted_score,
                            strategy=self.strategy_name,
                            metadata={
                                "iteration": iteration,
                                "critique": critique.model_dump(),
                            },
                        )
                    )

            # Step 4: Possibly refine query for next iteration
            if iteration < self.config.max_iterations - 1:
                refined_query = self._refine_query(query, all_results)

        all_results.sort(key=lambda r: r.score, reverse=True)
        return self._filter_by_threshold(all_results[:k])

    # ------------------------------------------------------------------
    # Reflection helpers
    # ------------------------------------------------------------------

    def _should_retrieve(self, query: str, current_results: list[RetrievalResult]) -> bool:
        """Predict the Retrieve reflection token -- should we retrieve more?"""
        if not current_results:
            return True

        prompt = (
            "Given the question and the passages already retrieved, decide whether "
            "additional retrieval is needed to fully answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Number of passages retrieved so far: {len(current_results)}\n\n"
            "Average relevance score: "
            f"{sum(r.score for r in current_results) / len(current_results):.3f}\n\n"
            "Answer with exactly YES or NO."
        )
        response = self._llm.complete(prompt, max_tokens=8)
        return "yes" in response.content.strip().lower()

    def _critique_passage(self, query: str, passage: str) -> CritiqueResult:
        """Evaluate a passage for relevance, support, and usefulness."""
        prompt = (
            "Evaluate the following passage for answering the given question. "
            "Respond in JSON with keys: is_relevant (bool), is_supported (bool), "
            "is_useful (bool), confidence (float 0-1), reasoning (string).\n\n"
            f"Question: {query}\n\n"
            f"Passage: {passage[:1000]}\n\n"
            "JSON:"
        )
        response = self._llm.complete(prompt, max_tokens=256)
        return self._parse_critique(response.content)

    def _refine_query(self, original_query: str, results: list[RetrievalResult]) -> str:
        """Generate a refined query based on what has been retrieved so far."""
        context_snippets = "\n".join(
            f"- {r.document.content[:200]}" for r in results[-3:]
        )
        prompt = (
            "Given the original question and some retrieved passages, generate "
            "a refined search query that will find additional relevant information "
            "not yet covered. Output only the refined query.\n\n"
            f"Original question: {original_query}\n\n"
            f"Already retrieved:\n{context_snippets}\n\n"
            "Refined query:"
        )
        response = self._llm.complete(prompt, max_tokens=128)
        return response.content.strip() or original_query

    @staticmethod
    def _parse_critique(text: str) -> CritiqueResult:
        """Best-effort JSON parsing of the critique response."""
        text = text.strip()
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                return CritiqueResult(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: assume relevant if we can't parse
        return CritiqueResult(
            is_relevant=True,
            is_supported=True,
            is_useful=True,
            confidence=0.5,
            reasoning="Could not parse critique response.",
        )
