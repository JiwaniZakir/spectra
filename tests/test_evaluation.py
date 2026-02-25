"""Tests for evaluation metrics and A/B testing."""

from __future__ import annotations

import numpy as np
import pytest

from spectra.evaluation.ab_testing import ABTest, ABTestConfig, ABTestResult, TestType
from spectra.evaluation.metrics import (
    EvaluationResult,
    EvaluationSample,
    MetricName,
    answer_correctness,
    coherence,
    compute_all_metrics,
    compute_metric,
    context_recall,
    faithfulness,
    latency_score,
)
from spectra.optimization.pareto import ParetoAnalyzer, ParetoPoint


@pytest.fixture
def sample() -> EvaluationSample:
    return EvaluationSample(
        query="What is machine learning?",
        answer="Machine learning is a field of AI that learns from data.",
        contexts=[
            "Machine learning is a subset of artificial intelligence.",
            "ML systems learn from data to make predictions.",
        ],
        ground_truth="Machine learning is a field of AI that uses algorithms to learn from data.",
        latency_seconds=0.5,
    )


@pytest.fixture
def empty_sample() -> EvaluationSample:
    return EvaluationSample(query="", answer="", contexts=[], ground_truth=None)


class TestFaithfulness:
    def test_basic_faithfulness(self, sample: EvaluationSample) -> None:
        score = faithfulness(sample)
        assert 0.0 <= score <= 1.0

    def test_empty_answer(self, empty_sample: EvaluationSample) -> None:
        assert faithfulness(empty_sample) == 0.0

    def test_high_overlap(self) -> None:
        s = EvaluationSample(
            query="test",
            answer="the cat sat on the mat",
            contexts=["the cat sat on the mat"],
        )
        score = faithfulness(s)
        assert score == 1.0

    def test_no_overlap(self) -> None:
        s = EvaluationSample(
            query="test",
            answer="xyz abc",
            contexts=["completely different content"],
        )
        score = faithfulness(s)
        assert score < 0.5


class TestContextRecall:
    def test_basic_recall(self, sample: EvaluationSample) -> None:
        score = context_recall(sample)
        assert 0.0 <= score <= 1.0

    def test_perfect_recall(self) -> None:
        s = EvaluationSample(
            query="test",
            answer="anything",
            contexts=["the answer is 42"],
            ground_truth="the answer is 42",
        )
        assert context_recall(s) == 1.0

    def test_no_ground_truth(self) -> None:
        s = EvaluationSample(query="q", answer="a", contexts=["c"])
        assert context_recall(s) == 0.0


class TestAnswerCorrectness:
    def test_identical(self) -> None:
        s = EvaluationSample(
            query="q",
            answer="the answer is correct",
            ground_truth="the answer is correct",
        )
        score = answer_correctness(s)
        # Should be near 1.0 (exact token overlap + high semantic sim)
        assert score > 0.8

    def test_no_ground_truth(self) -> None:
        s = EvaluationSample(query="q", answer="a")
        assert answer_correctness(s) == 0.0


class TestCoherence:
    def test_single_sentence(self) -> None:
        s = EvaluationSample(query="q", answer="Just one sentence")
        assert coherence(s) == 1.0

    def test_empty_answer(self) -> None:
        s = EvaluationSample(query="q", answer="")
        assert coherence(s) == 0.0


class TestLatencyScore:
    def test_zero_latency(self) -> None:
        s = EvaluationSample(query="q", answer="a", latency_seconds=0.0)
        assert latency_score(s) == 1.0

    def test_max_latency(self) -> None:
        s = EvaluationSample(query="q", answer="a", latency_seconds=10.0)
        assert latency_score(s) == 0.0

    def test_mid_latency(self) -> None:
        s = EvaluationSample(query="q", answer="a", latency_seconds=5.0)
        assert latency_score(s) == pytest.approx(0.5, abs=0.01)

    def test_no_latency(self) -> None:
        s = EvaluationSample(query="q", answer="a")
        assert latency_score(s) == 0.0


class TestComputeAllMetrics:
    def test_all_metrics(self, sample: EvaluationSample) -> None:
        result = compute_all_metrics([sample])
        assert isinstance(result, EvaluationResult)
        assert len(result.scores) == len(MetricName)
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_selected_metrics(self, sample: EvaluationSample) -> None:
        result = compute_all_metrics(
            [sample], metrics=[MetricName.FAITHFULNESS, MetricName.LATENCY_SCORE]
        )
        assert len(result.scores) == 2
        assert "faithfulness" in result.scores
        assert "latency_score" in result.scores

    def test_per_sample_scores(self, sample: EvaluationSample) -> None:
        result = compute_all_metrics([sample, sample])
        assert len(result.per_sample) == 2


class TestABTesting:
    def test_welch_t_test(self) -> None:
        rng = np.random.default_rng(42)
        scores_a = rng.normal(0.7, 0.1, 50).tolist()
        scores_b = rng.normal(0.8, 0.1, 50).tolist()

        ab = ABTest(ABTestConfig(test_type=TestType.WELCH_T))
        result = ab.compare_scores(scores_a, scores_b, "test_metric", "A", "B")

        assert isinstance(result, ABTestResult)
        assert result.p_value >= 0.0
        assert result.mean_b > result.mean_a

    def test_mann_whitney(self) -> None:
        rng = np.random.default_rng(42)
        scores_a = rng.uniform(0.5, 0.8, 30).tolist()
        scores_b = rng.uniform(0.6, 0.9, 30).tolist()

        ab = ABTest(ABTestConfig(test_type=TestType.MANN_WHITNEY_U))
        result = ab.compare_scores(scores_a, scores_b)
        assert 0.0 <= result.p_value <= 1.0

    def test_bootstrap(self) -> None:
        rng = np.random.default_rng(42)
        scores_a = rng.normal(0.7, 0.05, 30).tolist()
        scores_b = rng.normal(0.7, 0.05, 30).tolist()

        ab = ABTest(ABTestConfig(test_type=TestType.BOOTSTRAP, bootstrap_samples=1000))
        result = ab.compare_scores(scores_a, scores_b)
        # Similar distributions should not be significantly different
        assert result.p_value > 0.01

    def test_effect_size(self) -> None:
        # Large effect size
        scores_a = [0.5] * 30
        scores_b = [0.9] * 30

        ab = ABTest()
        result = ab.compare_scores(scores_a, scores_b)
        # Cohen's d should be very large (or inf for zero variance)
        assert abs(result.effect_size) > 0 or result.std_a == 0.0


class TestParetoAnalyzer:
    def test_simple_frontier(self) -> None:
        points = [
            ParetoPoint(config={"name": "A"}, objectives={"quality": 0.9, "speed": 0.3}),
            ParetoPoint(config={"name": "B"}, objectives={"quality": 0.7, "speed": 0.8}),
            ParetoPoint(config={"name": "C"}, objectives={"quality": 0.5, "speed": 0.5}),
        ]

        analyzer = ParetoAnalyzer({"quality": "maximize", "speed": "maximize"})
        frontier = analyzer.analyze(points)

        # A and B should be Pareto-optimal, C is dominated
        assert len(frontier.frontier_points) == 2
        assert len(frontier.dominated_points) == 1
        frontier_names = {p.config["name"] for p in frontier.frontier_points}
        assert "A" in frontier_names
        assert "B" in frontier_names

    def test_all_pareto_optimal(self) -> None:
        points = [
            ParetoPoint(config={"name": "A"}, objectives={"x": 1.0, "y": 0.0}),
            ParetoPoint(config={"name": "B"}, objectives={"x": 0.0, "y": 1.0}),
        ]

        analyzer = ParetoAnalyzer()
        frontier = analyzer.analyze(points)
        assert len(frontier.frontier_points) == 2

    def test_suggest_best(self) -> None:
        points = [
            ParetoPoint(config={"name": "A"}, objectives={"quality": 0.9, "speed": 0.3}),
            ParetoPoint(config={"name": "B"}, objectives={"quality": 0.7, "speed": 0.8}),
        ]

        analyzer = ParetoAnalyzer({"quality": "maximize", "speed": "maximize"})
        frontier = analyzer.analyze(points)

        # Quality-focused preference
        best = analyzer.suggest_best(frontier, preference={"quality": 0.9, "speed": 0.1})
        assert best is not None
        assert best.config["name"] == "A"

    def test_empty_points(self) -> None:
        analyzer = ParetoAnalyzer()
        frontier = analyzer.analyze([])
        assert frontier.num_total == 0
