"""Eight evaluation metrics for RAG pipeline quality assessment.

Metrics
-------
1. **Faithfulness** -- fraction of claims in the answer that are supported by context.
2. **Answer Relevance** -- semantic similarity between the answer and the question.
3. **Context Relevance** -- fraction of retrieved context that is relevant to the question.
4. **Coherence** -- logical consistency and flow of the generated answer.
5. **Context Recall** -- fraction of ground-truth facts covered by the retrieved context.
6. **Context Precision** -- how many of the top-k retrieved docs are relevant.
7. **Answer Correctness** -- F1-like overlap between answer and ground truth.
8. **Latency Score** -- inverse-normalized retrieval + generation latency.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from spectra.utils.embeddings import EmbeddingModel


class MetricName(str, Enum):
    """Enumeration of all supported evaluation metrics."""

    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCE = "answer_relevance"
    CONTEXT_RELEVANCE = "context_relevance"
    COHERENCE = "coherence"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_PRECISION = "context_precision"
    ANSWER_CORRECTNESS = "answer_correctness"
    LATENCY_SCORE = "latency_score"


class EvaluationSample(BaseModel):
    """A single evaluation data point.

    Attributes
    ----------
    query:
        The input question.
    answer:
        The generated answer.
    contexts:
        Retrieved context passages.
    ground_truth:
        The reference (gold) answer, if available.
    latency_seconds:
        End-to-end latency for retrieval + generation.
    metadata:
        Additional metadata.
    """

    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: str | None = None
    latency_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of evaluating one or more samples."""

    scores: dict[str, float] = Field(default_factory=dict)
    per_sample: list[dict[str, float]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ======================================================================
# Individual metric implementations
# ======================================================================

_embedding_model: EmbeddingModel | None = None


def _get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _token_overlap(text_a: str, text_b: str) -> float:
    """Compute token-level F1 overlap between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    precision = len(intersection) / len(tokens_a)
    recall = len(intersection) / len(tokens_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------
# 1. Faithfulness
# ------------------------------------------------------------------

def faithfulness(sample: EvaluationSample) -> float:
    """Measure what fraction of answer claims are supported by the context.

    Approximated by computing the token overlap between the answer and
    the concatenated context. A claim is considered supported if its
    tokens appear in the context.

    Returns a score in [0, 1].
    """
    if not sample.answer or not sample.contexts:
        return 0.0

    context_text = " ".join(sample.contexts).lower()
    context_tokens = set(context_text.split())

    answer_tokens = sample.answer.lower().split()
    if not answer_tokens:
        return 0.0

    supported = sum(1 for t in answer_tokens if t in context_tokens)
    return supported / len(answer_tokens)


# ------------------------------------------------------------------
# 2. Answer Relevance
# ------------------------------------------------------------------

def answer_relevance(sample: EvaluationSample) -> float:
    """Measure semantic similarity between the answer and the query.

    Uses embedding cosine similarity as a proxy for relevance.

    Returns a score in [0, 1].
    """
    if not sample.answer or not sample.query:
        return 0.0

    model = _get_embedding_model()
    embeddings = model.encode([sample.query, sample.answer])
    return max(0.0, _cosine_similarity(embeddings[0], embeddings[1]))


# ------------------------------------------------------------------
# 3. Context Relevance
# ------------------------------------------------------------------

def context_relevance(sample: EvaluationSample) -> float:
    """Measure what fraction of retrieved contexts are relevant to the query.

    Each context passage is compared to the query via embedding similarity.
    Passages with similarity above 0.5 are considered relevant.

    Returns a score in [0, 1].
    """
    if not sample.contexts or not sample.query:
        return 0.0

    model = _get_embedding_model()
    query_emb = model.encode_query(sample.query)
    context_embs = model.encode(sample.contexts)

    relevant = 0
    for ctx_emb in context_embs:
        sim = _cosine_similarity(query_emb, ctx_emb)
        if sim >= 0.5:
            relevant += 1

    return relevant / len(sample.contexts)


# ------------------------------------------------------------------
# 4. Coherence
# ------------------------------------------------------------------

def coherence(sample: EvaluationSample) -> float:
    """Measure the logical flow and consistency of the answer.

    Approximated by measuring the average pairwise cosine similarity
    between consecutive sentence embeddings in the answer. A coherent
    answer has high inter-sentence similarity.

    Returns a score in [0, 1].
    """
    if not sample.answer:
        return 0.0

    sentences = [s.strip() for s in sample.answer.split(".") if s.strip()]
    if len(sentences) <= 1:
        return 1.0

    model = _get_embedding_model()
    embeddings = model.encode(sentences)

    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(max(0.0, sim))

    return float(np.mean(similarities))


# ------------------------------------------------------------------
# 5. Context Recall
# ------------------------------------------------------------------

def context_recall(sample: EvaluationSample) -> float:
    """Measure how much of the ground truth is covered by retrieved contexts.

    Computes token overlap between the ground truth and the concatenated
    retrieved context.

    Returns a score in [0, 1].
    """
    if not sample.ground_truth or not sample.contexts:
        return 0.0

    context_text = " ".join(sample.contexts).lower()
    context_tokens = set(context_text.split())

    gt_tokens = set(sample.ground_truth.lower().split())
    if not gt_tokens:
        return 0.0

    recalled = sum(1 for t in gt_tokens if t in context_tokens)
    return recalled / len(gt_tokens)


# ------------------------------------------------------------------
# 6. Context Precision
# ------------------------------------------------------------------

def context_precision(sample: EvaluationSample) -> float:
    """Measure precision of retrieved contexts (fraction that are relevant).

    Uses embedding similarity to the ground truth to determine relevance.
    If no ground truth is available, falls back to query similarity.

    Returns a score in [0, 1].
    """
    if not sample.contexts:
        return 0.0

    reference = sample.ground_truth or sample.query
    if not reference:
        return 0.0

    model = _get_embedding_model()
    ref_emb = model.encode_query(reference)
    ctx_embs = model.encode(sample.contexts)

    relevant = 0
    for ctx_emb in ctx_embs:
        sim = _cosine_similarity(ref_emb, ctx_emb)
        if sim >= 0.5:
            relevant += 1

    return relevant / len(sample.contexts)


# ------------------------------------------------------------------
# 7. Answer Correctness
# ------------------------------------------------------------------

def answer_correctness(sample: EvaluationSample) -> float:
    """Measure correctness of the answer against the ground truth.

    Combines token-level F1 (50%) with embedding similarity (50%).

    Returns a score in [0, 1].
    """
    if not sample.answer or not sample.ground_truth:
        return 0.0

    token_f1 = _token_overlap(sample.answer, sample.ground_truth)

    model = _get_embedding_model()
    embeddings = model.encode([sample.answer, sample.ground_truth])
    semantic_sim = max(0.0, _cosine_similarity(embeddings[0], embeddings[1]))

    return 0.5 * token_f1 + 0.5 * semantic_sim


# ------------------------------------------------------------------
# 8. Latency Score
# ------------------------------------------------------------------

def latency_score(
    sample: EvaluationSample,
    max_acceptable_latency: float = 10.0,
) -> float:
    """Normalize latency into a [0, 1] quality score (lower latency = higher score).

    Parameters
    ----------
    sample:
        Must have ``latency_seconds`` set.
    max_acceptable_latency:
        Latency (in seconds) that maps to score 0.

    Returns a score in [0, 1].
    """
    if sample.latency_seconds is None:
        return 0.0
    if sample.latency_seconds <= 0:
        return 1.0
    score = 1.0 - min(sample.latency_seconds / max_acceptable_latency, 1.0)
    return max(0.0, score)


# ======================================================================
# Registry and batch evaluation
# ======================================================================

_METRIC_REGISTRY: dict[MetricName, Callable[[EvaluationSample], float]] = {
    MetricName.FAITHFULNESS: faithfulness,
    MetricName.ANSWER_RELEVANCE: answer_relevance,
    MetricName.CONTEXT_RELEVANCE: context_relevance,
    MetricName.COHERENCE: coherence,
    MetricName.CONTEXT_RECALL: context_recall,
    MetricName.CONTEXT_PRECISION: context_precision,
    MetricName.ANSWER_CORRECTNESS: answer_correctness,
    MetricName.LATENCY_SCORE: latency_score,
}


def compute_metric(metric: MetricName, sample: EvaluationSample) -> float:
    """Compute a single metric for one sample.

    Parameters
    ----------
    metric:
        Which metric to compute.
    sample:
        The evaluation sample.

    Returns
    -------
    float
        Score in [0, 1].
    """
    fn = _METRIC_REGISTRY[metric]
    return fn(sample)


def compute_all_metrics(
    samples: Sequence[EvaluationSample],
    metrics: Sequence[MetricName] | None = None,
) -> EvaluationResult:
    """Compute all (or selected) metrics across a batch of samples.

    Parameters
    ----------
    samples:
        Evaluation samples.
    metrics:
        Subset of metrics to compute. Defaults to all 8.

    Returns
    -------
    EvaluationResult
        Aggregated and per-sample scores.
    """
    if metrics is None:
        metrics = list(MetricName)

    per_sample: list[dict[str, float]] = []
    aggregated: dict[str, list[float]] = {m.value: [] for m in metrics}

    for sample in samples:
        sample_scores: dict[str, float] = {}
        for metric in metrics:
            score = compute_metric(metric, sample)
            sample_scores[metric.value] = score
            aggregated[metric.value].append(score)
        per_sample.append(sample_scores)

    avg_scores = {name: float(np.mean(vals)) for name, vals in aggregated.items() if vals}

    return EvaluationResult(
        scores=avg_scores,
        per_sample=per_sample,
        metadata={"num_samples": len(samples), "metrics": [m.value for m in metrics]},
    )
