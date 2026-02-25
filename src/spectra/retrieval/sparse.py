"""Sparse retrieval strategies: BM25 and SPLADE."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
from pydantic import Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)


# ======================================================================
# Tokenisation helpers
# ======================================================================

_SPLIT_RE = re.compile(r"\W+")


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace + punctuation tokenizer."""
    return [tok for tok in _SPLIT_RE.split(text.lower()) if tok]


# ======================================================================
# BM25 retriever (Okapi BM25)
# ======================================================================


class BM25Config(RetrieverConfig):
    """Configuration for BM25 retrieval."""

    k1: float = Field(default=1.5, ge=0.0, description="Term frequency saturation.")
    b: float = Field(default=0.75, ge=0.0, le=1.0, description="Length normalization.")


class BM25Retriever(BaseRetriever):
    """Okapi BM25 sparse retrieval.

    Implements the classic BM25 ranking function with configurable ``k1``
    and ``b`` parameters.  Tokenisation uses a simple lowercase split on
    non-word characters.

    References
    ----------
    Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and
    Beyond", Foundations and Trends in Information Retrieval, 2009.

    Parameters
    ----------
    config:
        BM25 configuration.
    """

    strategy_name = "bm25"
    capabilities = frozenset({RetrieverCapability.SPARSE})

    def __init__(self, config: BM25Config | None = None) -> None:
        self.config: BM25Config = config or BM25Config()
        super().__init__(self.config)
        self._documents: list[Document] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._tf: list[Counter[str]] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Tokenize and index *documents* for BM25."""
        for doc in documents:
            tokens = _tokenize(doc.content)
            tf = Counter(tokens)
            self._tf.append(tf)
            self._doc_lens.append(len(tokens))
            for term in set(tokens):
                self._doc_freqs[term] += 1
            self._documents.append(doc)

        total = sum(self._doc_lens)
        self._avg_dl = total / len(self._doc_lens) if self._doc_lens else 1.0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Score all documents against *query* using BM25."""
        k = self._effective_top_k(top_k)
        if not self._documents:
            return []

        query_tokens = _tokenize(query)
        n = len(self._documents)
        scores = np.zeros(n, dtype=np.float64)

        for term in query_tokens:
            if term not in self._doc_freqs:
                continue
            df = self._doc_freqs[term]
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for i in range(n):
                tf_val = self._tf[i].get(term, 0)
                dl = self._doc_lens[i]
                numerator = tf_val * (self.config.k1 + 1)
                denominator = tf_val + self.config.k1 * (
                    1 - self.config.b + self.config.b * dl / self._avg_dl
                )
                scores[i] += idf * numerator / denominator

        # Normalize scores to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores /= max_score

        k = min(k, n)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=float(scores[idx]),
                    strategy=self.strategy_name,
                )
            )

        return self._filter_by_threshold(results)


# ======================================================================
# SPLADE retriever (learned sparse representations)
# ======================================================================


class SPLADEConfig(RetrieverConfig):
    """Configuration for SPLADE retrieval."""

    model_name: str = "naver/splade-cocondenser-ensembledistil"
    device: str = "cpu"


class SPLADERetriever(BaseRetriever):
    """SPLADE learned sparse retrieval using log-saturated term weights.

    SPLADE expands queries and documents into sparse, high-dimensional
    term-weight vectors using a masked language model.  This implementation
    computes sparse representations and scores via dot-product.

    References
    ----------
    Formal et al., "SPLADE: Sparse Lexical and Expansion Model for First
    Stage Ranking", SIGIR 2021.

    Parameters
    ----------
    config:
        SPLADE configuration.
    """

    strategy_name = "splade"
    capabilities = frozenset({RetrieverCapability.SPARSE})

    def __init__(self, config: SPLADEConfig | None = None) -> None:
        self.config: SPLADEConfig = config or SPLADEConfig()
        super().__init__(self.config)
        self._documents: list[Document] = []
        self._sparse_vecs: list[dict[int, float]] = []
        self._tokenizer: object | None = None
        self._model: object | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for SPLADE. "
                "pip install transformers torch"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        self._model.eval()  # type: ignore[union-attr]

    def _encode_sparse(self, text: str) -> dict[int, float]:
        """Return a sparse vector {token_id: weight} for *text*."""
        import torch

        self._load_model()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)  # type: ignore[misc]
        with torch.no_grad():
            logits = self._model(**inputs).logits  # type: ignore[union-attr]
        # Log-saturated activation: max over sequence, then log(1 + ReLU(x))
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1),
            dim=1,
        ).values.squeeze(0)
        nonzero = weights.nonzero(as_tuple=True)[0]
        return {int(idx): float(weights[idx]) for idx in nonzero}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Encode and index *documents* as sparse SPLADE vectors."""
        for doc in documents:
            svec = self._encode_sparse(doc.content)
            self._sparse_vecs.append(svec)
            self._documents.append(doc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Score documents against *query* via sparse dot-product."""
        k = self._effective_top_k(top_k)
        if not self._documents:
            return []

        q_vec = self._encode_sparse(query)

        scores: list[float] = []
        for d_vec in self._sparse_vecs:
            dot = sum(q_vec.get(tid, 0.0) * w for tid, w in d_vec.items())
            scores.append(dot)

        scores_arr = np.array(scores)
        max_s = scores_arr.max()
        if max_s > 0:
            scores_arr /= max_s

        k = min(k, len(self._documents))
        top_idx = np.argpartition(scores_arr, -k)[-k:]
        top_idx = top_idx[np.argsort(scores_arr[top_idx])[::-1]]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            if scores_arr[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=float(scores_arr[idx]),
                    strategy=self.strategy_name,
                )
            )
        return self._filter_by_threshold(results)
