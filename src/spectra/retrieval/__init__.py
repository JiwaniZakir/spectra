"""Retrieval strategies implementing 12 distinct approaches to document retrieval."""

from __future__ import annotations

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverConfig,
)
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.retrieval.sparse import BM25Retriever, BM25Config
from spectra.retrieval.hybrid import HybridRetriever, HybridConfig
from spectra.retrieval.hyde import HyDERetriever, HyDEConfig
from spectra.retrieval.self_rag import SelfRAGRetriever, SelfRAGConfig
from spectra.retrieval.colbert import ColBERTRetriever, ColBERTConfig
from spectra.retrieval.multi_hop import MultiHopRetriever, MultiHopConfig
from spectra.retrieval.graph_rag import GraphRAGRetriever, GraphRAGConfig
from spectra.retrieval.ircot import IRCoTRetriever, IRCoTConfig
from spectra.retrieval.chain_of_note import ChainOfNoteRetriever, ChainOfNoteConfig
from spectra.retrieval.reranker import Reranker, RerankerConfig

__all__ = [
    "BaseRetriever",
    "Document",
    "RetrievalResult",
    "RetrieverConfig",
    "DenseRetriever",
    "DenseRetrieverConfig",
    "BM25Retriever",
    "BM25Config",
    "HybridRetriever",
    "HybridConfig",
    "HyDERetriever",
    "HyDEConfig",
    "SelfRAGRetriever",
    "SelfRAGConfig",
    "ColBERTRetriever",
    "ColBERTConfig",
    "MultiHopRetriever",
    "MultiHopConfig",
    "GraphRAGRetriever",
    "GraphRAGConfig",
    "IRCoTRetriever",
    "IRCoTConfig",
    "ChainOfNoteRetriever",
    "ChainOfNoteConfig",
    "Reranker",
    "RerankerConfig",
]
