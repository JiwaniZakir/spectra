"""Spectra -- Production RAG Evaluation & Optimization Toolkit.

Systematic RAG evaluation implementing 12 retrieval strategies with automated
A/B testing, quality metrics, and Pareto-optimal pipeline selection.

Public API
----------
The most common classes are re-exported here for convenience so that users
can write ``from spectra import DenseRetriever, Document`` instead of
importing from the sub-module directly.
"""

from __future__ import annotations

from spectra.chunking.document_aware import DocumentAwareChunker, DocumentAwareChunkerConfig
from spectra.chunking.fixed import FixedChunker, FixedChunkerConfig
from spectra.chunking.recursive import RecursiveChunker, RecursiveChunkerConfig
from spectra.chunking.semantic import SemanticChunker, SemanticChunkerConfig
from spectra.evaluation.ab_testing import ABTest, ABTestConfig, ABTestResult
from spectra.evaluation.metrics import (
    EvaluationResult,
    EvaluationSample,
    MetricName,
    compute_all_metrics,
    compute_metric,
)
from spectra.optimization.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    PipelineOptimizer,
)
from spectra.optimization.pareto import ParetoAnalyzer, ParetoFrontier, ParetoPoint
from spectra.optimization.pipeline import (
    ChunkingStrategy,
    PipelineConfig,
    PipelineResult,
    RAGPipeline,
)
from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.hybrid import HybridConfig, HybridRetriever
from spectra.retrieval.sparse import BM25Config, BM25Retriever, SPLADEConfig, SPLADERetriever

__version__ = "0.1.0"

__all__ = [
    # version
    "__version__",
    # retrieval -- base
    "BaseRetriever",
    "Document",
    "RetrievalResult",
    "RetrieverCapability",
    "RetrieverConfig",
    # retrieval -- strategies
    "DenseRetriever",
    "DenseRetrieverConfig",
    "BM25Retriever",
    "BM25Config",
    "SPLADERetriever",
    "SPLADEConfig",
    "HybridRetriever",
    "HybridConfig",
    # chunking
    "FixedChunker",
    "FixedChunkerConfig",
    "SemanticChunker",
    "SemanticChunkerConfig",
    "RecursiveChunker",
    "RecursiveChunkerConfig",
    "DocumentAwareChunker",
    "DocumentAwareChunkerConfig",
    # evaluation
    "EvaluationSample",
    "EvaluationResult",
    "MetricName",
    "compute_metric",
    "compute_all_metrics",
    "ABTest",
    "ABTestConfig",
    "ABTestResult",
    # optimization
    "RAGPipeline",
    "PipelineConfig",
    "PipelineResult",
    "ChunkingStrategy",
    "PipelineOptimizer",
    "OptimizerConfig",
    "OptimizationResult",
    "ParetoAnalyzer",
    "ParetoFrontier",
    "ParetoPoint",
]
