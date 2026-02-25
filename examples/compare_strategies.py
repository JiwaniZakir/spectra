"""Compare multiple retrieval strategies on the same query set.

This example demonstrates how to set up multiple retrieval strategies
and compare their performance using Spectra's evaluation framework
and A/B testing.

Run with:
    python examples/compare_strategies.py
"""

from spectra.evaluation.ab_testing import ABTest, ABTestConfig, TestType
from spectra.evaluation.metrics import EvaluationSample, MetricName, compute_all_metrics
from spectra.optimization.pipeline import ChunkingStrategy, PipelineConfig, RAGPipeline
from spectra.retrieval.base import Document
from spectra.retrieval.dense import DenseRetriever
from spectra.retrieval.hybrid import HybridRetriever
from spectra.retrieval.sparse import BM25Retriever


def make_documents() -> list[Document]:
    """Create a sample document corpus."""
    return [
        Document(
            id=f"doc_{i}",
            content=text,
        )
        for i, text in enumerate(
            [
                "Machine learning is a subset of artificial intelligence that focuses "
                "on building systems that learn from data. Deep learning uses neural "
                "networks with many layers.",
                "Natural language processing (NLP) enables computers to understand, "
                "interpret, and generate human language. Transformers have revolutionized "
                "NLP since their introduction in 2017.",
                "Information retrieval is the science of searching for information in "
                "documents. Modern systems combine sparse methods like BM25 with dense "
                "neural embeddings.",
                "Knowledge graphs represent information as entities and relationships. "
                "They enable structured querying and reasoning over large knowledge bases.",
                "Vector databases store high-dimensional vectors and support efficient "
                "nearest-neighbor search. Popular options include FAISS, Qdrant, and Weaviate.",
                "Prompt engineering involves designing effective prompts for language "
                "models. Techniques include few-shot learning, chain-of-thought, and "
                "retrieval augmentation.",
                "Evaluation of RAG systems requires measuring both retrieval quality "
                "and generation quality. Metrics include faithfulness, relevance, and "
                "answer correctness.",
                "Fine-tuning adapts a pre-trained model to a specific task or domain. "
                "Parameter-efficient methods like LoRA reduce computational costs while "
                "maintaining performance.",
            ]
        )
    ]


def main() -> None:
    documents = make_documents()
    queries = [
        "How does machine learning relate to deep learning?",
        "What are modern information retrieval techniques?",
        "How do you evaluate RAG systems?",
    ]
    ground_truths = [
        "Machine learning is a subset of AI, and deep learning uses neural networks with many layers.",
        "Modern IR combines sparse methods like BM25 with dense neural embeddings.",
        "RAG evaluation measures retrieval and generation quality using faithfulness, relevance, and correctness.",
    ]

    # ---- Set up strategies ----
    strategies: dict[str, RAGPipeline] = {}

    # Dense retriever
    strategies["dense"] = RAGPipeline(
        DenseRetriever(),
        PipelineConfig(name="dense", chunk_size=256, top_k=3),
    )

    # BM25 retriever
    strategies["bm25"] = RAGPipeline(
        BM25Retriever(),
        PipelineConfig(name="bm25", chunk_size=256, top_k=3),
    )

    # Hybrid retriever
    hybrid = HybridRetriever()
    hybrid.add_retriever(DenseRetriever(), weight=0.6)
    hybrid.add_retriever(BM25Retriever(), weight=0.4)
    strategies["hybrid"] = RAGPipeline(
        hybrid,
        PipelineConfig(name="hybrid", chunk_size=256, top_k=3),
    )

    # ---- Ingest and evaluate ----
    all_samples: dict[str, list[EvaluationSample]] = {}

    for name, pipeline in strategies.items():
        pipeline.ingest(documents)

        samples: list[EvaluationSample] = []
        for query, gt in zip(queries, ground_truths, strict=True):
            result = pipeline.query(query)
            samples.append(
                EvaluationSample(
                    query=query,
                    answer=" ".join(d.content[:200] for d in result.retrieved_documents),
                    contexts=[d.content for d in result.retrieved_documents],
                    ground_truth=gt,
                    latency_seconds=result.latency_seconds,
                )
            )
        all_samples[name] = samples

        eval_result = compute_all_metrics(samples)
        print(f"\n{'='*50}")
        print(f"Strategy: {name}")
        print(f"{'='*50}")
        for metric, score in eval_result.scores.items():
            print(f"  {metric}: {score:.4f}")

    # ---- A/B Testing ----
    print(f"\n{'='*50}")
    print("A/B Test: Dense vs Hybrid")
    print(f"{'='*50}")

    ab = ABTest(
        ABTestConfig(
            test_type=TestType.WELCH_T,
            metrics=[MetricName.FAITHFULNESS, MetricName.ANSWER_RELEVANCE],
            significance_level=0.05,
        )
    )
    results = ab.compare(all_samples["dense"], all_samples["hybrid"], "Dense", "Hybrid")

    for r in results:
        print(f"\n  Metric: {r.metric}")
        print(f"  Dense mean:  {r.mean_a:.4f} +/- {r.std_a:.4f}")
        print(f"  Hybrid mean: {r.mean_b:.4f} +/- {r.std_b:.4f}")
        print(f"  p-value: {r.p_value:.4f}")
        print(f"  Effect size (Cohen's d): {r.effect_size:.4f}")
        print(f"  Significant: {r.significant}")
        if r.winner:
            print(f"  Winner: {r.winner}")


if __name__ == "__main__":
    main()
