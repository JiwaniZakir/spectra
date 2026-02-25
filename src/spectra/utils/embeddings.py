"""Embedding model abstraction supporting sentence-transformers and API-based models."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True
    cache_size: int = 4096
    dimensions: int | None = None

    model_config = {"frozen": True}


class EmbeddingModel:
    """Wrapper around sentence-transformers with caching and batched encoding.

    Provides a unified interface for computing dense vector representations
    of text, with an LRU cache to avoid redundant computation.

    Parameters
    ----------
    config:
        Embedding configuration. Uses ``all-MiniLM-L6-v2`` by default.

    Example
    -------
    >>> model = EmbeddingModel()
    >>> vectors = model.encode(["hello world", "goodbye world"])
    >>> vectors.shape[1]  # embedding dimension
    384
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model: object | None = None
        self._cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. Install with: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the output embedding dimension."""
        if self.config.dimensions is not None:
            return self.config.dimensions
        model = self._load_model()
        dim: int = model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
        return dim

    def encode(
        self,
        texts: str | Sequence[str],
        *,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Encode one or more texts into dense vectors.

        Parameters
        ----------
        texts:
            A single string or a sequence of strings to encode.
        show_progress:
            Whether to show a progress bar during encoding.

        Returns
        -------
        NDArray[np.float32]
            A 2-D array of shape ``(n_texts, dimension)``.
        """
        if isinstance(texts, str):
            texts = [texts]

        texts = list(texts)
        results: list[NDArray[np.float32] | None] = [None] * len(texts)
        to_encode: list[tuple[int, str]] = []

        # Check cache first
        for idx, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                results[idx] = self._cache[key]
            else:
                to_encode.append((idx, text))

        # Encode cache misses
        if to_encode:
            model = self._load_model()
            batch_texts = [t for _, t in to_encode]
            embeddings: NDArray[np.float32] = model.encode(  # type: ignore[union-attr]
                batch_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize,
                convert_to_numpy=True,
            )
            for (idx, text), emb in zip(to_encode, embeddings, strict=True):
                vec = emb.astype(np.float32)
                results[idx] = vec
                self._put_cache(text, vec)

        return np.stack(results, axis=0)  # type: ignore[arg-type]

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Encode a single query string and return a 1-D vector."""
        return self.encode(query)[0]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _put_cache(self, text: str, vector: NDArray[np.float32]) -> None:
        key = self._cache_key(text)
        self._cache[key] = vector
        if len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Remove all cached embeddings."""
        self._cache.clear()
