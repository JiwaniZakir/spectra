"""Optuna-based pipeline optimizer.

Searches over chunking strategy, chunk size, retrieval strategy,
top-k, and other hyperparameters to maximize one or more evaluation
metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
from pydantic import BaseModel, Field

from spectra.evaluation.metrics import EvaluationSample, MetricName, compute_all_metrics
from spectra.optimization.pipeline import (
    ChunkingStrategy,
    PipelineConfig,
    RAGPipeline,
)
from spectra.retrieval.base import BaseRetriever, Document
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.sparse import BM25Retriever, BM25Config
from spectra.retrieval.hybrid import HybridRetriever, HybridConfig

logger = logging.getLogger(__name__)


class OptimizerConfig(BaseModel):
    """Configuration for the pipeline optimizer."""

    n_trials: int = Field(default=50, ge=1)
    timeout_seconds: float | None = Field(default=None)
    metric: MetricName = MetricName.FAITHFULNESS
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$")
    study_name: str = "spectra_optimization"
    storage: str | None = None  # Optuna storage URL
    seed: int = 42
    chunk_sizes: list[int] = Field(default_factory=lambda: [128, 256, 512, 1024])
    chunk_overlaps: list[int] = Field(default_factory=lambda: [0, 32, 64, 128])
    top_k_range: tuple[int, int] = (3, 20)
    retrieval_strategies: list[str] = Field(
        default_factory=lambda: ["dense", "bm25", "hybrid"]
    )
    chunking_strategies: list[str] = Field(
        default_factory=lambda: ["fixed", "recursive", "document_aware"]
    )

    model_config = {"frozen": True}


class OptimizationResult(BaseModel):
    """Result of an optimization run."""

    best_config: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    n_trials_completed: int = 0
    all_trials: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineOptimizer:
    """Optuna-based hyperparameter optimizer for RAG pipelines.

    Explores the space of pipeline configurations (chunking strategy,
    chunk size, retrieval strategy, top-k, etc.) and selects the
    configuration that maximizes (or minimizes) the chosen evaluation
    metric.

    Parameters
    ----------
    documents:
        Corpus documents to use for indexing.
    eval_queries:
        Evaluation queries.
    eval_ground_truths:
        Optional ground-truth answers.
    config:
        Optimizer configuration.

    Example
    -------
    >>> optimizer = PipelineOptimizer(docs, queries, ground_truths)
    >>> result = optimizer.optimize()
    >>> print(result.best_config)
    """

    def __init__(
        self,
        documents: Sequence[Document],
        eval_queries: Sequence[str],
        eval_ground_truths: Sequence[str] | None = None,
        config: OptimizerConfig | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self._documents = list(documents)
        self._queries = list(eval_queries)
        self._ground_truths = list(eval_ground_truths) if eval_ground_truths else None

    def optimize(self) -> OptimizationResult:
        """Run the optimization study.

        Returns
        -------
        OptimizationResult
            Best configuration and scores.
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("optuna is required. pip install optuna") from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=sampler,
            storage=self.config.storage,
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
        )

        all_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            }
            for t in study.trials
        ]

        return OptimizationResult(
            best_config=study.best_trial.params,
            best_score=study.best_trial.value or 0.0,
            n_trials_completed=len(study.trials),
            all_trials=all_trials,
            metadata={
                "metric": self.config.metric.value,
                "direction": self.config.direction,
            },
        )

    def _objective(self, trial: Any) -> float:
        """Optuna objective function for a single trial."""
        import optuna

        # Sample hyperparameters
        chunking = trial.suggest_categorical(
            "chunking_strategy", self.config.chunking_strategies
        )
        chunk_size = trial.suggest_categorical("chunk_size", self.config.chunk_sizes)
        chunk_overlap = trial.suggest_categorical("chunk_overlap", self.config.chunk_overlaps)
        top_k = trial.suggest_int(
            "top_k", self.config.top_k_range[0], self.config.top_k_range[1]
        )
        strategy = trial.suggest_categorical(
            "retrieval_strategy", self.config.retrieval_strategies
        )

        # Ensure overlap < chunk_size
        chunk_overlap = min(chunk_overlap, chunk_size // 2)

        # Create retriever
        retriever = self._create_retriever(strategy, top_k, trial)

        # Create pipeline
        pipeline_config = PipelineConfig(
            name=f"trial_{trial.number}",
            chunking_strategy=ChunkingStrategy(chunking),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )
        pipeline = RAGPipeline(retriever, pipeline_config)

        # Ingest documents
        try:
            pipeline.ingest(self._documents)
        except Exception as e:
            logger.warning("Ingestion failed for trial %d: %s", trial.number, e)
            raise optuna.TrialPruned() from e

        # Evaluate
        try:
            scores = pipeline.evaluate(
                self._queries,
                self._ground_truths,
                metrics=[self.config.metric],
            )
        except Exception as e:
            logger.warning("Evaluation failed for trial %d: %s", trial.number, e)
            raise optuna.TrialPruned() from e

        return scores.get(self.config.metric.value, 0.0)

    def _create_retriever(
        self, strategy: str, top_k: int, trial: Any
    ) -> BaseRetriever:
        """Create a retriever based on the sampled strategy."""
        match strategy:
            case "dense":
                return DenseRetriever(DenseRetrieverConfig(top_k=top_k))
            case "bm25":
                k1 = trial.suggest_float("bm25_k1", 0.5, 3.0)
                b = trial.suggest_float("bm25_b", 0.0, 1.0)
                return BM25Retriever(BM25Config(top_k=top_k, k1=k1, b=b))
            case "hybrid":
                dense_weight = trial.suggest_float("dense_weight", 0.1, 0.9)
                retriever = HybridRetriever(config=HybridConfig(top_k=top_k))
                retriever.add_retriever(
                    DenseRetriever(DenseRetrieverConfig(top_k=top_k)),
                    weight=dense_weight,
                )
                retriever.add_retriever(
                    BM25Retriever(BM25Config(top_k=top_k)),
                    weight=1.0 - dense_weight,
                )
                return retriever
            case _:
                return DenseRetriever(DenseRetrieverConfig(top_k=top_k))
