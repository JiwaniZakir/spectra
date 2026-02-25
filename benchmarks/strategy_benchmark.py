"""Benchmark harness for comparing retrieval strategy performance.

Runs configurable benchmarks across retrieval strategies, measuring
latency, throughput, and memory usage.

Usage:
    python benchmarks/strategy_benchmark.py
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from spectra.retrieval.base import BaseRetriever, Document
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.sparse import BM25Retriever, BM25Config
from spectra.retrieval.hybrid import HybridRetriever, HybridConfig


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single strategy."""

    strategy: str
    num_documents: int
    num_queries: int
    indexing_time_s: float = 0.0
    avg_query_time_ms: float = 0.0
    p50_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    p99_query_time_ms: float = 0.0
    queries_per_second: float = 0.0
    memory_mb: float = 0.0
    avg_results_returned: float = 0.0


def generate_documents(n: int, avg_length: int = 500) -> list[Document]:
    """Generate synthetic documents for benchmarking."""
    import random
    rng = random.Random(42)
    words = (
        "the of and to a in is it that was for on are with as his they be at "
        "one have this from or had by not but what all were when we there can "
        "an each which she do their if will up about out many then them these "
        "machine learning neural network deep data model training algorithm "
        "retrieval document embedding vector search index query relevance "
        "score ranking evaluation metric precision recall faithfulness "
        "language processing natural text generation transformer attention"
    ).split()

    docs: list[Document] = []
    for i in range(n):
        length = rng.randint(avg_length // 2, avg_length * 2)
        content = " ".join(rng.choices(words, k=length))
        docs.append(Document(id=f"bench_doc_{i}", content=content))
    return docs


def generate_queries(n: int) -> list[str]:
    """Generate synthetic queries."""
    queries = [
        "machine learning neural network training",
        "document retrieval embedding search",
        "natural language processing text",
        "deep learning model evaluation",
        "vector search index ranking",
        "algorithm data model precision",
        "transformer attention mechanism",
        "query relevance scoring metric",
        "information retrieval system",
        "text generation language model",
    ]
    return (queries * (n // len(queries) + 1))[:n]


def benchmark_strategy(
    name: str,
    retriever: BaseRetriever,
    documents: list[Document],
    queries: list[str],
) -> BenchmarkResult:
    """Benchmark a single retrieval strategy."""
    # Memory before
    gc.collect()
    mem_before = _get_memory_mb()

    # Index documents
    t0 = time.perf_counter()
    retriever.add_documents(documents)
    indexing_time = time.perf_counter() - t0

    # Memory after indexing
    gc.collect()
    mem_after = _get_memory_mb()

    # Query
    query_times: list[float] = []
    total_results = 0

    for query in queries:
        t0 = time.perf_counter()
        results = retriever.retrieve(query, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        query_times.append(elapsed)
        total_results += len(results)

    import numpy as np
    times = np.array(query_times)

    return BenchmarkResult(
        strategy=name,
        num_documents=len(documents),
        num_queries=len(queries),
        indexing_time_s=indexing_time,
        avg_query_time_ms=float(np.mean(times)),
        p50_query_time_ms=float(np.percentile(times, 50)),
        p95_query_time_ms=float(np.percentile(times, 95)),
        p99_query_time_ms=float(np.percentile(times, 99)),
        queries_per_second=len(queries) / (sum(query_times) / 1000) if query_times else 0,
        memory_mb=mem_after - mem_before,
        avg_results_returned=total_results / len(queries) if queries else 0,
    )


def _get_memory_mb() -> float:
    """Get current process memory in MB (approximate via sys.getsizeof)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: KB
    except (ImportError, AttributeError):
        return 0.0


def main() -> None:
    num_docs = 500
    num_queries = 50

    print(f"Generating {num_docs} documents and {num_queries} queries...")
    documents = generate_documents(num_docs)
    queries = generate_queries(num_queries)

    strategies: dict[str, BaseRetriever] = {
        "BM25": BM25Retriever(BM25Config(top_k=10)),
    }

    # Only include dense/hybrid if sentence-transformers is available
    try:
        import sentence_transformers
        strategies["Dense"] = DenseRetriever(
            DenseRetrieverConfig(top_k=10, use_faiss=False)
        )
        hybrid = HybridRetriever(config=HybridConfig(top_k=10))
        hybrid.add_retriever(DenseRetriever(DenseRetrieverConfig(top_k=10, use_faiss=False)), 0.5)
        hybrid.add_retriever(BM25Retriever(BM25Config(top_k=10)), 0.5)
        strategies["Hybrid (Dense+BM25)"] = hybrid
    except ImportError:
        print("sentence-transformers not available, skipping Dense/Hybrid benchmarks")

    results: list[BenchmarkResult] = []

    for name, retriever in strategies.items():
        print(f"\nBenchmarking: {name}")
        result = benchmark_strategy(name, retriever, documents, queries)
        results.append(result)

        print(f"  Indexing time:     {result.indexing_time_s:.3f}s")
        print(f"  Avg query time:    {result.avg_query_time_ms:.2f}ms")
        print(f"  P50 query time:    {result.p50_query_time_ms:.2f}ms")
        print(f"  P95 query time:    {result.p95_query_time_ms:.2f}ms")
        print(f"  Queries/sec:       {result.queries_per_second:.1f}")
        print(f"  Memory delta:      {result.memory_mb:.1f}MB")
        print(f"  Avg results:       {result.avg_results_returned:.1f}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<25} {'Index(s)':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'QPS':<10}")
    print("=" * 80)
    for r in results:
        print(
            f"{r.strategy:<25} {r.indexing_time_s:<10.3f} "
            f"{r.avg_query_time_ms:<10.2f} {r.p95_query_time_ms:<10.2f} "
            f"{r.queries_per_second:<10.1f}"
        )


if __name__ == "__main__":
    main()
