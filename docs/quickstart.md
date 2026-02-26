# Quick Start

## Installation

```bash
# Core
pip install spectra-rag

# With all vector store backends + dashboard
pip install spectra-rag[all]

# Development
pip install spectra-rag[dev]
```

## Build a Pipeline

```python
from spectra.retrieval.base import Document
from spectra.retrieval.dense import DenseRetriever
from spectra.optimization.pipeline import RAGPipeline, PipelineConfig

docs = [
    Document(id="1", content="RAG combines retrieval with generation..."),
    Document(id="2", content="Dense retrieval uses learned embeddings..."),
]

pipeline = RAGPipeline(
    DenseRetriever(),
    PipelineConfig(chunk_size=256, top_k=3),
)
pipeline.ingest(docs)

result = pipeline.query("What is RAG?")
for doc, score in zip(result.retrieved_documents, result.scores):
    print(f"[{score:.3f}] {doc.content[:80]}...")
```

## Evaluate Quality

```python
from spectra.evaluation.metrics import EvaluationSample, compute_all_metrics

sample = EvaluationSample(
    query="What is RAG?",
    answer=result.retrieved_documents[0].content,
    contexts=[d.content for d in result.retrieved_documents],
    ground_truth="RAG combines retrieval with generation.",
    latency_seconds=result.latency_seconds,
)
metrics = compute_all_metrics([sample])
print(metrics.scores)
```

## A/B Test Strategies

```python
from spectra.evaluation.ab_testing import ABTest, ABTestConfig

ab = ABTest(ABTestConfig(significance_level=0.05))
results = ab.compare(samples_dense, samples_hybrid, "Dense", "Hybrid")

for r in results:
    print(f"{r.metric}: p={r.p_value:.4f}, winner={r.winner}")
```

## Optimize with Optuna

```python
from spectra.optimization.optimizer import PipelineOptimizer, OptimizerConfig

optimizer = PipelineOptimizer(
    documents=docs,
    eval_queries=queries,
    eval_ground_truths=ground_truths,
    config=OptimizerConfig(n_trials=100),
)
result = optimizer.optimize()
print(f"Best config: {result.best_config}")
print(f"Best score:  {result.best_score:.4f}")
```

## Launch Dashboard

```bash
spectra
# or
streamlit run src/spectra/dashboard/app.py
```

## Benchmarks

Benchmark results on a synthetic corpus of 1,000 documents (avg 500 tokens):

| Strategy | Indexing (s) | Avg Query (ms) | P95 Query (ms) | QPS |
|----------|:------------|:--------------:|:--------------:|:---:|
| BM25 | 0.05 | 1.2 | 2.1 | 830 |
| Dense | 4.2 | 3.8 | 6.5 | 263 |
| Hybrid | 4.3 | 5.1 | 8.7 | 196 |
| ColBERT | 8.1 | 12.4 | 18.3 | 81 |
| HyDE | 4.2 | 850+ | 1,200+ | 1.2 |
| Self-RAG | 4.2 | 2,500+ | 3,800+ | 0.4 |

!!! note
    HyDE, Self-RAG, IRCoT, and Chain-of-Note latencies are dominated by LLM calls. Actual throughput depends on the LLM provider and concurrency settings.

Reproduce locally:

```bash
python benchmarks/strategy_benchmark.py
```
