"""Optimize a RAG pipeline using Optuna and analyze the Pareto frontier.

This example demonstrates:
1. Optuna-based hyperparameter optimization
2. Multi-objective Pareto frontier analysis
3. Selecting the best pipeline configuration

Run with:
    python examples/optimize_pipeline.py
"""

from spectra.evaluation.metrics import MetricName
from spectra.optimization.optimizer import OptimizerConfig, PipelineOptimizer
from spectra.optimization.pareto import ParetoAnalyzer, ParetoPoint
from spectra.retrieval.base import Document


def make_corpus() -> list[Document]:
    """Create a sample document corpus for optimization."""
    topics = [
        ("Transformers are sequence-to-sequence models that use self-attention. "
         "They were introduced in 'Attention Is All You Need' by Vaswani et al. "
         "in 2017. Transformers have become the foundation of modern NLP."),
        ("BERT is a bidirectional transformer pre-trained on masked language modeling "
         "and next sentence prediction. It set new benchmarks on many NLP tasks."),
        ("GPT models are autoregressive transformers trained to predict the next token. "
         "GPT-3 demonstrated impressive few-shot learning capabilities."),
        ("Retrieval-augmented generation combines a retriever with a generator to "
         "ground language model outputs in external knowledge."),
        ("Vector databases are optimized for storing and querying high-dimensional "
         "vectors. They use algorithms like HNSW and IVF for approximate nearest "
         "neighbor search."),
        ("Sentence transformers produce dense sentence-level embeddings suitable "
         "for semantic similarity, clustering, and retrieval tasks."),
        ("BM25 remains a strong baseline for document retrieval, using term frequency "
         "and inverse document frequency to rank documents."),
        ("Cross-encoder models jointly encode query-document pairs for more accurate "
         "relevance scoring, but are too slow for exhaustive first-stage retrieval."),
    ]
    return [Document(id=f"doc_{i}", content=text) for i, text in enumerate(topics)]


def main() -> None:
    documents = make_corpus()
    queries = [
        "What are transformers in NLP?",
        "How does retrieval-augmented generation work?",
        "What is BM25 used for?",
    ]
    ground_truths = [
        "Transformers are sequence-to-sequence models using self-attention, introduced in 2017.",
        "RAG combines a retriever with a generator to ground LLM outputs in external knowledge.",
        "BM25 is used for document retrieval based on term frequency and IDF.",
    ]

    # ---- Optuna Optimization ----
    print("Starting Optuna optimization...")
    print("(This uses a small search space for demonstration)\n")

    optimizer = PipelineOptimizer(
        documents=documents,
        eval_queries=queries,
        eval_ground_truths=ground_truths,
        config=OptimizerConfig(
            n_trials=10,  # Small for demo; use 50-200 in practice
            metric=MetricName.FAITHFULNESS,
            retrieval_strategies=["dense", "bm25"],
            chunking_strategies=["fixed", "recursive"],
            chunk_sizes=[128, 256, 512],
            chunk_overlaps=[0, 32, 64],
            seed=42,
        ),
    )

    result = optimizer.optimize()

    print(f"Best score: {result.best_score:.4f}")
    print(f"Best config: {result.best_config}")
    print(f"Trials completed: {result.n_trials_completed}")

    # ---- Pareto Frontier Analysis ----
    print(f"\n{'='*50}")
    print("Pareto Frontier Analysis")
    print(f"{'='*50}")

    # Simulate multi-objective results from trials
    points: list[ParetoPoint] = []
    for trial in result.all_trials:
        if trial["value"] is not None:
            # Use the optimization score as "quality" and infer "speed"
            # from chunk_size (smaller chunks = more chunks = slower)
            chunk_size = trial["params"].get("chunk_size", 256)
            speed_score = min(1.0, chunk_size / 1024)
            points.append(
                ParetoPoint(
                    config=trial["params"],
                    objectives={
                        "quality": trial["value"],
                        "speed": speed_score,
                    },
                )
            )

    analyzer = ParetoAnalyzer(
        directions={"quality": "maximize", "speed": "maximize"}
    )
    frontier = analyzer.analyze(points)

    print(f"\nTotal configurations: {frontier.num_total}")
    print(f"Pareto-optimal: {len(frontier.frontier_points)}")
    print(f"Dominated: {len(frontier.dominated_points)}")

    print("\nPareto-optimal configurations:")
    for pt in frontier.frontier_points:
        print(f"  Quality={pt.objectives['quality']:.4f}, "
              f"Speed={pt.objectives['speed']:.4f}, "
              f"Config={pt.config}")

    # Suggest best with preference
    best = analyzer.suggest_best(frontier, preference={"quality": 0.7, "speed": 0.3})
    if best:
        print(f"\nRecommended (quality-focused): {best.config}")
        print(f"  Quality: {best.objectives['quality']:.4f}")
        print(f"  Speed: {best.objectives['speed']:.4f}")


if __name__ == "__main__":
    main()
