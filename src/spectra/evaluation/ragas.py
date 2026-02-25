"""RAGAS (Retrieval Augmented Generation Assessment) framework implementation.

Implements the evaluation methodology from Es et al. (2023), which defines
component-wise metrics for assessing RAG pipeline quality without requiring
human annotations.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field

from spectra.evaluation.metrics import (
    EvaluationSample,
    MetricName,
    answer_correctness,
    answer_relevance,
    coherence,
    context_precision,
    context_recall,
    context_relevance,
    faithfulness,
)
from spectra.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class RAGASConfig(BaseModel):
    """Configuration for the RAGAS evaluator."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevance",
            "context_relevance",
            "context_recall",
        ]
    )
    batch_size: int = Field(default=8, ge=1)
    use_llm_judge: bool = Field(
        default=False,
        description="Use LLM-as-judge for more accurate but slower evaluation.",
    )

    model_config = {"frozen": True}


class RAGASResult(BaseModel):
    """Structured RAGAS evaluation result."""

    overall_score: float = 0.0
    metric_scores: dict[str, float] = Field(default_factory=dict)
    per_sample_scores: list[dict[str, float]] = Field(default_factory=list)
    num_samples: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGASEvaluator:
    """RAGAS evaluation framework.

    RAGAS evaluates RAG pipelines along four component-wise dimensions:

    1. **Faithfulness**: Is the answer grounded in the retrieved context?
    2. **Answer Relevance**: Does the answer address the question?
    3. **Context Relevance**: Is the retrieved context pertinent?
    4. **Context Recall**: Does the context cover the ground truth?

    The overall RAGAS score is the harmonic mean of the component scores.

    When ``use_llm_judge`` is enabled, an LLM is used for more nuanced
    evaluation (e.g. claim-level faithfulness checking).

    References
    ----------
    Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented
    Generation", EMNLP 2023 (arXiv:2309.15217, 2023).

    Parameters
    ----------
    config:
        RAGAS configuration.
    """

    def __init__(self, config: RAGASConfig | None = None) -> None:
        self.config = config or RAGASConfig()
        self._llm: LLMClient | None = None
        if self.config.use_llm_judge:
            self._llm = LLMClient(self.config.llm)

    def evaluate(self, samples: Sequence[EvaluationSample]) -> RAGASResult:
        """Evaluate a batch of samples using the RAGAS framework.

        Parameters
        ----------
        samples:
            Evaluation samples with query, answer, contexts, and optionally
            ground_truth.

        Returns
        -------
        RAGASResult
            Overall and per-metric scores.
        """
        per_sample: list[dict[str, float]] = []
        metric_accumulators: dict[str, list[float]] = {m: [] for m in self.config.metrics}

        for sample in samples:
            sample_scores = self._evaluate_sample(sample)
            per_sample.append(sample_scores)
            for metric, score in sample_scores.items():
                if metric in metric_accumulators:
                    metric_accumulators[metric].append(score)

        # Compute averages
        metric_scores = {
            name: float(np.mean(vals)) for name, vals in metric_accumulators.items() if vals
        }

        # Overall score: harmonic mean of component scores
        overall = self._harmonic_mean(list(metric_scores.values()))

        return RAGASResult(
            overall_score=overall,
            metric_scores=metric_scores,
            per_sample_scores=per_sample,
            num_samples=len(samples),
            metadata={"config": self.config.model_dump()},
        )

    def _evaluate_sample(self, sample: EvaluationSample) -> dict[str, float]:
        """Evaluate a single sample across all configured metrics."""
        scores: dict[str, float] = {}

        metric_fns = {
            "faithfulness": self._eval_faithfulness,
            "answer_relevance": lambda s: answer_relevance(s),
            "context_relevance": lambda s: context_relevance(s),
            "context_recall": lambda s: context_recall(s),
            "context_precision": lambda s: context_precision(s),
            "answer_correctness": lambda s: answer_correctness(s),
            "coherence": lambda s: coherence(s),
        }

        for metric_name in self.config.metrics:
            fn = metric_fns.get(metric_name)
            if fn:
                scores[metric_name] = fn(sample)
            else:
                logger.warning("Unknown metric: %s", metric_name)

        return scores

    def _eval_faithfulness(self, sample: EvaluationSample) -> float:
        """Evaluate faithfulness, optionally using LLM-as-judge."""
        if not self.config.use_llm_judge or self._llm is None:
            return faithfulness(sample)

        # LLM-based claim decomposition and verification
        return self._llm_faithfulness(sample)

    def _llm_faithfulness(self, sample: EvaluationSample) -> float:
        """LLM-based faithfulness evaluation via claim decomposition."""
        if not sample.answer or not sample.contexts:
            return 0.0

        assert self._llm is not None

        # Step 1: Decompose answer into atomic claims
        claims_prompt = (
            "Decompose the following answer into a list of atomic factual claims. "
            "Return one claim per line.\n\n"
            f"Answer: {sample.answer}\n\n"
            "Claims:"
        )
        claims_response = self._llm.complete(claims_prompt, max_tokens=512)
        claims = [
            c.strip().lstrip("- ").lstrip("0123456789.)")
            for c in claims_response.content.strip().splitlines()
            if c.strip()
        ]

        if not claims:
            return 1.0

        # Step 2: Verify each claim against context
        context_text = "\n\n".join(sample.contexts)
        supported = 0
        for claim in claims:
            verify_prompt = (
                "Determine if the following claim is supported by the given context. "
                "Answer with exactly YES or NO.\n\n"
                f"Context: {context_text[:3000]}\n\n"
                f"Claim: {claim}\n\n"
                "Supported:"
            )
            response = self._llm.complete(verify_prompt, max_tokens=8)
            if "yes" in response.content.strip().lower():
                supported += 1

        return supported / len(claims)

    @staticmethod
    def _harmonic_mean(values: list[float]) -> float:
        """Compute the harmonic mean, treating zeros as zero."""
        if not values:
            return 0.0
        positive = [v for v in values if v > 0]
        if not positive:
            return 0.0
        return float(len(positive) / sum(1.0 / v for v in positive))
