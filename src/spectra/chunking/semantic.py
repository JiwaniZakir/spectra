"""Semantic chunking -- split on embedding-distance breakpoints.

Segments text by detecting semantic shift between consecutive sentences,
placing chunk boundaries where the embedding similarity drops below a
computed threshold.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from spectra.retrieval.base import Document
from spectra.utils.embeddings import EmbeddingConfig, EmbeddingModel


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunkerConfig(BaseModel):
    """Configuration for semantic chunking."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    breakpoint_percentile: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Percentile of distance distribution to use as breakpoint threshold.",
    )
    min_chunk_size: int = Field(default=50, ge=1)
    max_chunk_size: int = Field(default=2000, ge=1)
    buffer_size: int = Field(
        default=1,
        ge=0,
        description="Number of sentences on each side to include in the embedding window.",
    )

    model_config = {"frozen": True}


class SemanticChunker:
    """Embedding-based semantic chunker.

    Splits a document into sentences, embeds groups of sentences (with
    a sliding buffer), and places chunk boundaries at points where the
    cosine distance between consecutive groups exceeds a percentile-based
    threshold.

    Parameters
    ----------
    config:
        Semantic chunker configuration.

    Example
    -------
    >>> chunker = SemanticChunker()
    >>> doc = Document(id="1", content="First topic sentence. " * 20 + "Completely different topic. " * 20)
    >>> chunks = chunker.chunk(doc)
    """

    def __init__(self, config: SemanticChunkerConfig | None = None) -> None:
        self.config = config or SemanticChunkerConfig()
        self._embedding_model = EmbeddingModel(self.config.embedding)

    def chunk(self, document: Document) -> list[Document]:
        """Split *document* at semantic breakpoints.

        Parameters
        ----------
        document:
            The document to chunk.

        Returns
        -------
        list[Document]
            Semantically coherent chunks.
        """
        sentences = self._split_sentences(document.content)
        if len(sentences) <= 1:
            return [document] if document.content.strip() else []

        # Build sentence groups (with buffer context)
        groups = self._build_groups(sentences)

        # Embed groups
        embeddings = self._embedding_model.encode(groups)

        # Compute distances between consecutive groups
        distances = self._compute_distances(embeddings)

        # Find breakpoints
        breakpoints = self._find_breakpoints(distances)

        # Build chunks
        return self._build_chunks(sentences, breakpoints, document)

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents."""
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = _SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _build_groups(self, sentences: list[str]) -> list[str]:
        """Build sentence groups with buffer context for embedding."""
        groups: list[str] = []
        buf = self.config.buffer_size
        for i in range(len(sentences)):
            start = max(0, i - buf)
            end = min(len(sentences), i + buf + 1)
            group = " ".join(sentences[start:end])
            groups.append(group)
        return groups

    @staticmethod
    def _compute_distances(embeddings: NDArray[np.float32]) -> NDArray[np.float64]:
        """Compute cosine distances between consecutive embeddings."""
        # Cosine similarity between consecutive pairs
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms

        distances = np.zeros(len(embeddings) - 1)
        for i in range(len(distances)):
            sim = float(np.dot(normalized[i], normalized[i + 1]))
            distances[i] = 1.0 - sim

        return distances

    def _find_breakpoints(self, distances: NDArray[np.float64]) -> list[int]:
        """Find indices where semantic distance exceeds the threshold."""
        if len(distances) == 0:
            return []
        threshold = float(np.percentile(distances, self.config.breakpoint_percentile))
        return [i + 1 for i, d in enumerate(distances) if d >= threshold]

    def _build_chunks(
        self,
        sentences: list[str],
        breakpoints: list[int],
        source: Document,
    ) -> list[Document]:
        """Build chunk documents from sentences and breakpoints."""
        boundaries = [0] + breakpoints + [len(sentences)]
        chunks: list[Document] = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_text = " ".join(sentences[start:end]).strip()

            if not chunk_text:
                continue

            # Enforce min/max size by merging or splitting
            if len(chunk_text) < self.config.min_chunk_size and chunks:
                # Merge with previous chunk
                prev = chunks[-1]
                chunks[-1] = Document(
                    id=prev.id,
                    content=prev.content + " " + chunk_text,
                    metadata=prev.metadata,
                )
                continue

            if len(chunk_text) > self.config.max_chunk_size:
                # Hard split at max size
                for sub_start in range(0, len(chunk_text), self.config.max_chunk_size):
                    sub_text = chunk_text[sub_start : sub_start + self.config.max_chunk_size].strip()
                    if sub_text:
                        chunks.append(
                            Document(
                                id=f"{source.id}_semchunk_{len(chunks)}",
                                content=sub_text,
                                metadata={
                                    **source.metadata,
                                    "source_doc_id": source.id,
                                    "chunk_index": len(chunks),
                                    "sentence_start": start,
                                    "sentence_end": end,
                                    "chunking_strategy": "semantic",
                                },
                            )
                        )
                continue

            chunks.append(
                Document(
                    id=f"{source.id}_semchunk_{len(chunks)}",
                    content=chunk_text,
                    metadata={
                        **source.metadata,
                        "source_doc_id": source.id,
                        "chunk_index": len(chunks),
                        "sentence_start": start,
                        "sentence_end": end,
                        "chunking_strategy": "semantic",
                    },
                )
            )

        return chunks
