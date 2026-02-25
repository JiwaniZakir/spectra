"""FAISS vector store backend.

Supports IndexFlatIP (inner product / cosine), IndexFlatL2, and
IndexIVFFlat for larger datasets.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from spectra.vectorstores.base import BaseVectorStore, SearchResult, VectorStoreConfig

logger = logging.getLogger(__name__)


class FAISSConfig(VectorStoreConfig):
    """Configuration for the FAISS vector store."""

    index_type: str = Field(
        default="flat",
        pattern="^(flat|ivf)$",
        description="FAISS index type: 'flat' for brute-force, 'ivf' for approximate.",
    )
    nlist: int = Field(default=100, ge=1, description="Number of IVF clusters.")
    nprobe: int = Field(default=8, ge=1, description="Number of clusters to probe at query time.")
    persist_path: str | None = None


class FAISSStore(BaseVectorStore):
    """FAISS-backed vector store.

    Supports both exact (``IndexFlatIP``) and approximate
    (``IndexIVFFlat``) search.  Vectors are normalised before insertion
    when using cosine similarity.

    Parameters
    ----------
    config:
        FAISS store configuration.
    """

    def __init__(self, config: FAISSConfig | None = None) -> None:
        self.config: FAISSConfig = config or FAISSConfig()
        super().__init__(self.config)
        self._index: Any = None
        self._id_map: list[str] = []  # positional index -> document ID
        self._metadata_map: dict[str, dict[str, Any]] = {}

        try:
            import faiss
            self._faiss = faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc

        self._create_index()

    def _create_index(self) -> None:
        """Create the FAISS index based on configuration."""
        dim = self.config.dimension
        if self.config.index_type == "ivf":
            quantizer = self._faiss.IndexFlatIP(dim)
            self._index = self._faiss.IndexIVFFlat(
                quantizer, dim, self.config.nlist, self._faiss.METRIC_INNER_PRODUCT
            )
        else:
            if self.config.metric == "l2":
                self._index = self._faiss.IndexFlatL2(dim)
            else:
                self._index = self._faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        ids: Sequence[str],
        vectors: NDArray[np.float32],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the FAISS index."""
        if len(ids) != vectors.shape[0]:
            raise ValueError("Number of IDs must match number of vectors.")

        # Normalize for cosine similarity
        if self.config.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            vectors = vectors / norms

        # Train IVF index if needed
        if self.config.index_type == "ivf" and not self._index.is_trained:
            self._index.train(vectors)

        self._index.add(vectors)

        for i, doc_id in enumerate(ids):
            self._id_map.append(doc_id)
            if metadatas and i < len(metadatas):
                self._metadata_map[doc_id] = metadatas[i]

    def search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbours in the FAISS index."""
        if self._index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)

        # Normalize query for cosine
        if self.config.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 1e-10:
                query = query / norm

        # Set nprobe for IVF
        if self.config.index_type == "ivf":
            self._index.nprobe = self.config.nprobe

        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            doc_id = self._id_map[idx]

            # Apply metadata filter
            if filter_metadata:
                meta = self._metadata_map.get(doc_id, {})
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(
                SearchResult(
                    id=doc_id,
                    score=float(score),
                    metadata=self._metadata_map.get(doc_id, {}),
                )
            )

        return results

    def delete(self, ids: Sequence[str]) -> None:
        """Delete vectors by ID (rebuilds the index without the deleted vectors)."""
        ids_set = set(ids)
        keep_indices = [i for i, doc_id in enumerate(self._id_map) if doc_id not in ids_set]

        if not keep_indices:
            self._id_map.clear()
            self._metadata_map = {k: v for k, v in self._metadata_map.items() if k not in ids_set}
            self._create_index()
            return

        # Reconstruct remaining vectors
        remaining_vectors = np.zeros((len(keep_indices), self.config.dimension), dtype=np.float32)
        for new_idx, old_idx in enumerate(keep_indices):
            remaining_vectors[new_idx] = self._index.reconstruct(old_idx)

        remaining_ids = [self._id_map[i] for i in keep_indices]

        # Rebuild
        for doc_id in ids:
            self._metadata_map.pop(doc_id, None)

        self._id_map.clear()
        self._create_index()
        self.add(remaining_ids, remaining_vectors)

    def count(self) -> int:
        """Return the number of vectors in the store."""
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save the index and metadata to disk."""
        save_path = path or self.config.persist_path
        if not save_path:
            raise ValueError("No persist_path configured.")

        os.makedirs(save_path, exist_ok=True)
        self._faiss.write_index(self._index, os.path.join(save_path, "index.faiss"))

        meta = {
            "id_map": self._id_map,
            "metadata_map": self._metadata_map,
        }
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str | None = None) -> None:
        """Load the index and metadata from disk."""
        load_path = path or self.config.persist_path
        if not load_path:
            raise ValueError("No persist_path configured.")

        self._index = self._faiss.read_index(os.path.join(load_path, "index.faiss"))

        with open(os.path.join(load_path, "metadata.json")) as f:
            meta = json.load(f)
        self._id_map = meta["id_map"]
        self._metadata_map = meta["metadata_map"]
