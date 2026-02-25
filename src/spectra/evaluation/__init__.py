"""Evaluation framework with 8 quality metrics, RAGAS, A/B testing, and synthetic data."""

from __future__ import annotations

from spectra.evaluation.metrics import (
    EvaluationResult,
    EvaluationSample,
    MetricName,
    compute_all_metrics,
    compute_metric,
)
from spectra.evaluation.ragas import RAGASEvaluator, RAGASConfig, RAGASResult
from spectra.evaluation.ab_testing import ABTest, ABTestConfig, ABTestResult
from spectra.evaluation.synthetic import SyntheticGenerator, SyntheticConfig

__all__ = [
    "EvaluationResult",
    "EvaluationSample",
    "MetricName",
    "compute_all_metrics",
    "compute_metric",
    "RAGASEvaluator",
    "RAGASConfig",
    "RAGASResult",
    "ABTest",
    "ABTestConfig",
    "ABTestResult",
    "SyntheticGenerator",
    "SyntheticConfig",
]
