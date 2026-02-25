"""IRCoT -- Interleaved Retrieval with Chain-of-Thought.

Implements the interleaving approach from Trivedi et al. (2023) where
chain-of-thought reasoning steps alternate with retrieval steps to
solve complex, multi-step questions.
"""

from __future__ import annotations

import logging
from typing import Sequence

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


class CoTStep(BaseModel):
    """A single chain-of-thought reasoning step with its retrieval results."""

    thought: str
    retrieval_query: str
    retrieved_docs: list[str] = Field(default_factory=list)
    step_number: int = 0


class IRCoTConfig(RetrieverConfig):
    """Configuration for IRCoT retrieval."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_steps: int = Field(default=4, ge=1, le=10)
    docs_per_step: int = Field(default=3, ge=1, le=20)
    cot_temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class IRCoTRetriever(BaseRetriever):
    """Interleaved Retrieval Chain-of-Thought (IRCoT).

    IRCoT interleaves chain-of-thought reasoning with retrieval:

    1. Given the question and accumulated evidence, the LLM generates the
       next reasoning step (thought).
    2. The thought is used as a retrieval query to find supporting passages.
    3. New passages are added to the evidence and the next step begins.
    4. This continues until the LLM signals completion or the maximum
       number of steps is reached.

    References
    ----------
    Trivedi et al., "Interleaving Retrieval with Chain-of-Thought
    Reasoning for Knowledge-Intensive Multi-Step Questions", ACL 2023.

    Parameters
    ----------
    config:
        IRCoT configuration.
    """

    strategy_name = "ircot"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.ITERATIVE, RetrieverCapability.GENERATIVE}
    )

    def __init__(self, config: IRCoTConfig | None = None) -> None:
        self.config: IRCoTConfig = config or IRCoTConfig()
        super().__init__(self.config)
        self._llm = LLMClient(self.config.llm)
        self._inner = DenseRetriever(
            DenseRetrieverConfig(
                top_k=self.config.docs_per_step,
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
        """Perform interleaved CoT + retrieval."""
        k = self._effective_top_k(top_k)

        all_results: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        cot_steps: list[CoTStep] = []

        for step_num in range(self.config.max_steps):
            # Step 1: Generate next CoT thought
            thought, retrieval_query = self._generate_thought(
                query, cot_steps, all_results
            )

            logger.debug(
                "IRCoT step %d: thought=%r, query=%r",
                step_num,
                thought[:80],
                retrieval_query[:80],
            )

            # Step 2: Retrieve using the thought as query
            candidates = self._inner.retrieve(
                retrieval_query, top_k=self.config.docs_per_step
            )

            step = CoTStep(
                thought=thought,
                retrieval_query=retrieval_query,
                step_number=step_num,
            )

            for result in candidates:
                if result.document.id in seen_ids:
                    continue
                seen_ids.add(result.document.id)
                step.retrieved_docs.append(result.document.id)
                # Decay score by step number
                score = result.score * (1.0 - 0.05 * step_num)
                all_results.append(
                    RetrievalResult(
                        document=result.document,
                        score=score,
                        strategy=self.strategy_name,
                        metadata={
                            "step": step_num,
                            "thought": thought,
                            "retrieval_query": retrieval_query,
                        },
                    )
                )

            cot_steps.append(step)

            # Step 3: Check if CoT indicates completion
            if self._is_complete(thought):
                logger.debug("IRCoT: reasoning complete at step %d", step_num)
                break

        all_results.sort(key=lambda r: r.score, reverse=True)
        return self._filter_by_threshold(all_results[:k])

    # ------------------------------------------------------------------
    # CoT generation
    # ------------------------------------------------------------------

    def _generate_thought(
        self,
        query: str,
        cot_steps: list[CoTStep],
        results: list[RetrievalResult],
    ) -> tuple[str, str]:
        """Generate the next reasoning step and a retrieval query.

        Returns
        -------
        tuple[str, str]
            (thought, retrieval_query)
        """
        # Build context from previous steps
        context_parts: list[str] = []
        for step in cot_steps:
            context_parts.append(f"Thought {step.step_number + 1}: {step.thought}")

        evidence = "\n".join(
            f"[{i+1}] {r.document.content[:250]}"
            for i, r in enumerate(results[-5:])
        )

        cot_history = "\n".join(context_parts) if context_parts else "(no previous steps)"

        prompt = (
            "You are reasoning step-by-step to answer a question. "
            "At each step, produce:\n"
            "1. A THOUGHT explaining your reasoning.\n"
            "2. A QUERY to retrieve supporting evidence.\n\n"
            "If you have enough information, write THOUGHT: [DONE] and a final summary.\n\n"
            f"Question: {query}\n\n"
            f"Previous reasoning:\n{cot_history}\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            "Next step:\nTHOUGHT:"
        )

        response = self._llm.complete(
            prompt,
            temperature=self.config.cot_temperature,
            max_tokens=256,
        )
        return self._parse_thought_response(response.content, query)

    @staticmethod
    def _parse_thought_response(text: str, fallback_query: str) -> tuple[str, str]:
        """Parse the LLM response into (thought, query)."""
        thought = text.strip()
        query = fallback_query

        lines = text.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("QUERY:"):
                thought = "\n".join(lines[:i]).strip()
                query = line.split(":", 1)[1].strip()
                break

        if not thought:
            thought = text.strip()

        return thought, query

    @staticmethod
    def _is_complete(thought: str) -> bool:
        """Check if the thought signals reasoning is complete."""
        return "[done]" in thought.lower() or "[complete]" in thought.lower()
