# Spectra

**Benchmark, compare, and optimize RAG pipelines with rigorous statistical testing.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-58A6FF?style=for-the-badge)](https://github.com/JiwaniZakir/spectra/blob/main/LICENSE)

## Overview

Spectra is a systematic RAG evaluation toolkit that implements **12 retrieval strategies** spanning dense, sparse, hybrid, generative, iterative, and graph-based approaches. It provides automated A/B testing with statistical rigor (Welch's t-test, Mann-Whitney U, bootstrap), 8 quality metrics aligned with the RAGAS framework, and Pareto-optimal pipeline selection powered by Optuna -- everything needed to move from prototype to production RAG.

## Features

- **12 Retrieval Strategies** -- Dense, BM25, SPLADE, Hybrid RRF, HyDE, Self-RAG, ColBERT, Multi-hop, Graph-RAG, IRCoT, Chain-of-Note, Cross-encoder Reranker
- **8 Evaluation Metrics** -- Faithfulness, Answer Relevance, Context Relevance, Coherence, Context Recall, Context Precision, Answer Correctness, Latency Score
- **Statistical A/B Testing** -- Welch's t-test, Mann-Whitney U, paired t-test, bootstrap permutation with Bonferroni correction
- **Optuna-Powered Optimization** -- Automated hyperparameter search across chunking, embedding, retrieval, and re-ranking stages
- **Pareto Frontier Analysis** -- Multi-objective selection with NSGA-II crowding distance for quality vs. latency trade-offs
- **4 Chunking Strategies** -- Fixed, Semantic, Recursive, and Document-Aware splitting
- **3 Vector Store Backends** -- FAISS, Qdrant, pgvector
- **Streamlit Dashboard** -- Interactive visualization of metrics, comparisons, and optimization results
- **Synthetic Data Generation** -- Automatic evaluation dataset creation for rapid iteration

## Retrieval Strategies

| # | Strategy | Type | Key Idea |
|---|----------|------|----------|
| 1 | **Dense (Bi-encoder)** | Dense | Embed query & docs into shared vector space |
| 2 | **BM25** | Sparse | Classic TF-IDF lexical matching |
| 3 | **SPLADE** | Sparse | Learned sparse representations via MLM |
| 4 | **Hybrid (RRF)** | Hybrid | Reciprocal Rank Fusion of dense + sparse |
| 5 | **HyDE** | Generative | Embed LLM-generated hypothetical answers |
| 6 | **Self-RAG** | Iterative | Reflection tokens for adaptive retrieval |
| 7 | **ColBERT** | Late Interaction | Token-level MaxSim scoring |
| 8 | **Multi-hop** | Iterative | Decompose complex questions into retrieval steps |
| 9 | **Graph-RAG** | Graph | Knowledge graph traversal for entity-rich queries |
| 10 | **IRCoT** | Iterative | Interleaved chain-of-thought + retrieval |
| 11 | **Chain-of-Note** | Generative | Reading notes for robust retrieval |
| 12 | **Cross-encoder Reranker** | Reranking | Joint query-document scoring as second stage |

## Evaluation Metrics

All metrics return scores in `[0, 1]`:

| Metric | What It Measures | Ground Truth Required |
|--------|-----------------|:---------------------:|
| **Faithfulness** | Fraction of answer claims supported by context | No |
| **Answer Relevance** | Semantic similarity between answer and query | No |
| **Context Relevance** | Fraction of retrieved contexts relevant to query | No |
| **Coherence** | Logical flow and consistency of the answer | No |
| **Context Recall** | Coverage of ground-truth facts in retrieved context | Yes |
| **Context Precision** | Precision of retrieved docs against ground truth | Yes |
| **Answer Correctness** | Token F1 + semantic similarity vs ground truth | Yes |
| **Latency Score** | Normalized inverse latency (lower latency = higher) | No |

## Project Structure

```
spectra/
├── src/spectra/
│   ├── retrieval/          # 12 retrieval strategy implementations
│   ├── chunking/           # Fixed, Semantic, Recursive, Document-Aware
│   ├── evaluation/         # 8 metrics, A/B testing, RAGAS, synthetic data
│   ├── optimization/       # Optuna optimizer, Pareto frontier analysis
│   ├── vectorstores/       # FAISS, Qdrant, pgvector backends
│   ├── dashboard/          # Streamlit interactive UI
│   └── utils/              # Embedding models, LLM client
├── examples/               # quickstart, compare_strategies, optimize_pipeline
├── tests/                  # Unit & integration tests
├── benchmarks/             # Performance benchmark harness
└── pyproject.toml
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

```bash
git clone https://github.com/JiwaniZakir/spectra.git
cd spectra
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Released under the [Apache License 2.0](https://github.com/JiwaniZakir/spectra/blob/main/LICENSE).
