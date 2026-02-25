"""Spectra Quick Start -- minimal example of building and querying a RAG pipeline.

Run with:
    python examples/quickstart.py
"""

from spectra.chunking.recursive import RecursiveChunker
from spectra.evaluation.metrics import (
    EvaluationSample,
    MetricName,
    compute_all_metrics,
)
from spectra.optimization.pipeline import PipelineConfig, RAGPipeline
from spectra.retrieval.base import Document
from spectra.retrieval.dense import DenseRetriever


def main() -> None:
    # 1. Prepare some sample documents
    documents = [
        Document(
            id="doc_1",
            content=(
                "Retrieval-Augmented Generation (RAG) combines a retrieval system "
                "with a language model. The retriever finds relevant documents from "
                "a knowledge base, and the generator produces an answer conditioned "
                "on both the question and the retrieved context."
            ),
        ),
        Document(
            id="doc_2",
            content=(
                "Dense retrieval uses learned embeddings to represent queries and "
                "documents as dense vectors. Similarity is computed via cosine "
                "similarity or inner product in the embedding space. Models like "
                "DPR and Contriever are popular choices."
            ),
        ),
        Document(
            id="doc_3",
            content=(
                "BM25 is a classic sparse retrieval method based on term frequency "
                "and inverse document frequency. It remains a strong baseline and "
                "is often combined with dense retrieval in hybrid systems."
            ),
        ),
        Document(
            id="doc_4",
            content=(
                "HyDE (Hypothetical Document Embeddings) generates a hypothetical "
                "answer to the query using an LLM, then embeds that answer and "
                "searches for real documents similar to the hypothetical one. "
                "This bridges the query-document gap in embedding space."
            ),
        ),
    ]

    # 2. Create a pipeline with a dense retriever
    retriever = DenseRetriever()
    pipeline = RAGPipeline(
        retriever,
        PipelineConfig(
            name="quickstart",
            chunk_size=256,
            chunk_overlap=32,
            top_k=3,
        ),
    )

    # 3. Ingest documents
    num_chunks = pipeline.ingest(documents)
    print(f"Ingested {pipeline.num_documents} documents into {num_chunks} chunks")

    # 4. Query the pipeline
    query = "What is RAG and how does dense retrieval work?"
    result = pipeline.query(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(result.retrieved_documents)} documents in {result.latency_seconds:.4f}s")
    for i, (doc, score) in enumerate(
        zip(result.retrieved_documents, result.scores, strict=True)
    ):
        print(f"\n  [{i+1}] Score: {score:.4f}")
        print(f"      {doc.content[:120]}...")

    # 5. Evaluate with metrics
    sample = EvaluationSample(
        query=query,
        answer=" ".join(d.content[:200] for d in result.retrieved_documents),
        contexts=[d.content for d in result.retrieved_documents],
        ground_truth=(
            "RAG combines retrieval with generation. Dense retrieval uses "
            "learned embeddings for similarity search."
        ),
        latency_seconds=result.latency_seconds,
    )

    eval_result = compute_all_metrics([sample])
    print("\nEvaluation Metrics:")
    for metric, score in eval_result.scores.items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    main()
