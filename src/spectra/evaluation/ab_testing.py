"""Statistical A/B testing for comparing RAG pipeline configurations.

Provides rigorous statistical tests (Welch's t-test, Mann-Whitney U,
bootstrap confidence intervals) to determine whether one pipeline
configuration significantly outperforms another.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from spectra.evaluation.metrics import (
    EvaluationResult,
    EvaluationSample,
    MetricName,
    compute_metric,
)

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Statistical test types."""

    WELCH_T = "welch_t"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP = "bootstrap"
    PAIRED_T = "paired_t"


class ABTestConfig(BaseModel):
    """Configuration for A/B testing."""

    test_type: TestType = TestType.WELCH_T
    significance_level: float = Field(default=0.05, ge=0.001, le=0.5)
    metrics: list[MetricName] = Field(
        default_factory=lambda: [MetricName.FAITHFULNESS, MetricName.ANSWER_RELEVANCE]
    )
    bootstrap_samples: int = Field(default=10_000, ge=100)
    min_sample_size: int = Field(default=30, ge=5)
    apply_bonferroni: bool = Field(
        default=True,
        description="Apply Bonferroni correction for multiple comparisons.",
    )

    model_config = {"frozen": True}


class ABTestResult(BaseModel):
    """Result of an A/B test between two pipeline configurations."""

    pipeline_a_name: str
    pipeline_b_name: str
    metric: str
    test_type: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    effect_size: float  # Cohen's d
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    significant: bool
    winner: str | None = None
    num_samples: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ABTest:
    """Statistical A/B testing engine for RAG pipelines.

    Compares two pipeline configurations across one or more metrics using
    parametric or non-parametric statistical tests.

    Parameters
    ----------
    config:
        A/B test configuration.

    Example
    -------
    >>> ab = ABTest()
    >>> results = ab.compare(samples_a, samples_b, "dense", "hybrid")
    >>> for r in results:
    ...     print(f"{r.metric}: p={r.p_value:.4f}, winner={r.winner}")
    """

    def __init__(self, config: ABTestConfig | None = None) -> None:
        self.config = config or ABTestConfig()

    def compare(
        self,
        samples_a: Sequence[EvaluationSample],
        samples_b: Sequence[EvaluationSample],
        name_a: str = "Pipeline A",
        name_b: str = "Pipeline B",
    ) -> list[ABTestResult]:
        """Compare two sets of evaluation samples across all configured metrics.

        Parameters
        ----------
        samples_a:
            Samples from pipeline A.
        samples_b:
            Samples from pipeline B.
        name_a:
            Display name for pipeline A.
        name_b:
            Display name for pipeline B.

        Returns
        -------
        list[ABTestResult]
            One result per metric.
        """
        results: list[ABTestResult] = []
        num_tests = len(self.config.metrics)
        alpha = self.config.significance_level
        if self.config.apply_bonferroni and num_tests > 1:
            alpha /= num_tests

        for metric in self.config.metrics:
            scores_a = np.array([compute_metric(metric, s) for s in samples_a])
            scores_b = np.array([compute_metric(metric, s) for s in samples_b])

            result = self._run_test(
                scores_a, scores_b, metric.value, name_a, name_b, alpha
            )
            results.append(result)

        return results

    def compare_scores(
        self,
        scores_a: Sequence[float],
        scores_b: Sequence[float],
        metric_name: str = "metric",
        name_a: str = "Pipeline A",
        name_b: str = "Pipeline B",
    ) -> ABTestResult:
        """Compare two raw score arrays for a single metric.

        Parameters
        ----------
        scores_a:
            Scores from pipeline A.
        scores_b:
            Scores from pipeline B.
        metric_name:
            Name of the metric being compared.
        name_a:
            Display name for pipeline A.
        name_b:
            Display name for pipeline B.
        """
        return self._run_test(
            np.array(scores_a),
            np.array(scores_b),
            metric_name,
            name_a,
            name_b,
            self.config.significance_level,
        )

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def _run_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        metric_name: str,
        name_a: str,
        name_b: str,
        alpha: float,
    ) -> ABTestResult:
        """Run the configured statistical test."""
        mean_a = float(np.mean(scores_a))
        mean_b = float(np.mean(scores_b))
        std_a = float(np.std(scores_a, ddof=1)) if len(scores_a) > 1 else 0.0
        std_b = float(np.std(scores_b, ddof=1)) if len(scores_b) > 1 else 0.0
        effect = self._cohens_d(scores_a, scores_b)

        match self.config.test_type:
            case TestType.WELCH_T:
                stat, pval, ci = self._welch_t_test(scores_a, scores_b)
            case TestType.MANN_WHITNEY_U:
                stat, pval, ci = self._mann_whitney_test(scores_a, scores_b)
            case TestType.BOOTSTRAP:
                stat, pval, ci = self._bootstrap_test(scores_a, scores_b)
            case TestType.PAIRED_T:
                stat, pval, ci = self._paired_t_test(scores_a, scores_b)
            case _:
                stat, pval, ci = self._welch_t_test(scores_a, scores_b)

        significant = pval < alpha
        winner = None
        if significant:
            winner = name_a if mean_a > mean_b else name_b

        return ABTestResult(
            pipeline_a_name=name_a,
            pipeline_b_name=name_b,
            metric=metric_name,
            test_type=self.config.test_type.value,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            effect_size=effect,
            statistic=stat,
            p_value=pval,
            confidence_interval=ci,
            significant=significant,
            winner=winner,
            num_samples=len(scores_a) + len(scores_b),
        )

    @staticmethod
    def _welch_t_test(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float, tuple[float, float]]:
        """Welch's t-test (unequal variance t-test)."""
        if len(a) < 2 or len(b) < 2:
            return 0.0, 1.0, (0.0, 0.0)
        result = stats.ttest_ind(a, b, equal_var=False)
        stat = float(result.statistic)
        pval = float(result.pvalue)
        ci = _mean_diff_ci(a, b, alpha=0.05)
        return stat, pval, ci

    @staticmethod
    def _mann_whitney_test(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float, tuple[float, float]]:
        """Mann-Whitney U test (non-parametric)."""
        if len(a) < 2 or len(b) < 2:
            return 0.0, 1.0, (0.0, 0.0)
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        stat = float(result.statistic)
        pval = float(result.pvalue)
        ci = _mean_diff_ci(a, b, alpha=0.05)
        return stat, pval, ci

    def _bootstrap_test(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float, tuple[float, float]]:
        """Bootstrap permutation test."""
        observed_diff = float(np.mean(a) - np.mean(b))
        combined = np.concatenate([a, b])
        n_a = len(a)
        rng = np.random.default_rng(42)

        count_extreme = 0
        diffs: list[float] = []
        for _ in range(self.config.bootstrap_samples):
            perm = rng.permutation(combined)
            perm_a = perm[:n_a]
            perm_b = perm[n_a:]
            diff = float(np.mean(perm_a) - np.mean(perm_b))
            diffs.append(diff)
            if abs(diff) >= abs(observed_diff):
                count_extreme += 1

        pval = count_extreme / self.config.bootstrap_samples
        ci_low = float(np.percentile(diffs, 2.5))
        ci_high = float(np.percentile(diffs, 97.5))
        return observed_diff, pval, (ci_low, ci_high)

    @staticmethod
    def _paired_t_test(
        a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float, tuple[float, float]]:
        """Paired t-test (requires equal-length arrays)."""
        n = min(len(a), len(b))
        if n < 2:
            return 0.0, 1.0, (0.0, 0.0)
        a, b = a[:n], b[:n]
        result = stats.ttest_rel(a, b)
        stat = float(result.statistic)
        pval = float(result.pvalue)
        diff = a - b
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(n))
        t_crit = float(stats.t.ppf(0.975, n - 1))
        ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
        return stat, pval, ci

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return 0.0
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled_std < 1e-10:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_std)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mean_diff_ci(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute confidence interval for the difference in means."""
    mean_diff = float(np.mean(a) - np.mean(b))
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    df = len(a) + len(b) - 2
    t_crit = float(stats.t.ppf(1 - alpha / 2, max(df, 1)))
    margin = t_crit * float(se)
    return (mean_diff - margin, mean_diff + margin)
