"""Chunking strategies for splitting documents into retrievable units."""

from __future__ import annotations

from spectra.chunking.fixed import FixedChunker, FixedChunkerConfig
from spectra.chunking.semantic import SemanticChunker, SemanticChunkerConfig
from spectra.chunking.recursive import RecursiveChunker, RecursiveChunkerConfig
from spectra.chunking.document_aware import DocumentAwareChunker, DocumentAwareChunkerConfig

__all__ = [
    "FixedChunker",
    "FixedChunkerConfig",
    "SemanticChunker",
    "SemanticChunkerConfig",
    "RecursiveChunker",
    "RecursiveChunkerConfig",
    "DocumentAwareChunker",
    "DocumentAwareChunkerConfig",
]
