"""Recursive character text splitter.

Recursively splits text using a hierarchy of separators (paragraphs,
sentences, words) to produce chunks that respect natural boundaries
while staying within size limits.
"""

from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, Field

from spectra.retrieval.base import Document


# Default separator hierarchy: double-newline, newline, sentence-end, space
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class RecursiveChunkerConfig(BaseModel):
    """Configuration for recursive chunking."""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    separators: list[str] = Field(default_factory=lambda: list(_DEFAULT_SEPARATORS))
    strip_whitespace: bool = True
    keep_separator: bool = True

    model_config = {"frozen": True}


class RecursiveChunker:
    """Recursive text splitter.

    Attempts to split text using the highest-level separator (e.g.
    double newline) first.  If any resulting piece still exceeds
    ``chunk_size``, it recurses with the next separator.  This produces
    chunks that respect document structure (paragraphs > sentences >
    words) while staying within the target size.

    This is modeled after LangChain's ``RecursiveCharacterTextSplitter``.

    Parameters
    ----------
    config:
        Recursive chunker configuration.

    Example
    -------
    >>> chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=200))
    >>> doc = Document(id="1", content="Para one.\\n\\nPara two is longer. " * 10)
    >>> chunks = chunker.chunk(doc)
    """

    def __init__(self, config: RecursiveChunkerConfig | None = None) -> None:
        self.config = config or RecursiveChunkerConfig()

    def chunk(self, document: Document) -> list[Document]:
        """Split *document* recursively.

        Parameters
        ----------
        document:
            The document to chunk.

        Returns
        -------
        list[Document]
            Chunks respecting natural text boundaries.
        """
        texts = self._split_text(document.content, self.config.separators)
        merged = self._merge_with_overlap(texts)

        chunks: list[Document] = []
        for i, text in enumerate(merged):
            if self.config.strip_whitespace:
                text = text.strip()
            if not text:
                continue
            chunks.append(
                Document(
                    id=f"{document.id}_recchunk_{i}",
                    content=text,
                    metadata={
                        **document.metadata,
                        "source_doc_id": document.id,
                        "chunk_index": i,
                        "chunking_strategy": "recursive",
                    },
                )
            )
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents."""
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    # ------------------------------------------------------------------
    # Recursive splitting
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the separator hierarchy."""
        if len(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        # Find the best separator for this text
        separator = ""
        remaining_separators = separators

        for i, sep in enumerate(separators):
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        if not separator:
            # No separator works: hard-split by chunk_size
            return self._hard_split(text)

        # Split on the chosen separator
        if self.config.keep_separator:
            # Keep separator at the end of each piece
            pieces = self._split_keeping_separator(text, separator)
        else:
            pieces = text.split(separator)

        # Collect final chunks, recursing on oversized pieces
        result: list[str] = []
        for piece in pieces:
            if not piece.strip():
                continue
            if len(piece) <= self.config.chunk_size:
                result.append(piece)
            elif remaining_separators:
                result.extend(self._split_text(piece, remaining_separators))
            else:
                result.extend(self._hard_split(piece))

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Character-level hard split when no separator works."""
        pieces: list[str] = []
        stride = max(1, self.config.chunk_size - self.config.chunk_overlap)
        for start in range(0, len(text), stride):
            end = start + self.config.chunk_size
            piece = text[start:end]
            if piece.strip():
                pieces.append(piece)
            if end >= len(text):
                break
        return pieces

    @staticmethod
    def _split_keeping_separator(text: str, separator: str) -> list[str]:
        """Split text on *separator* but keep the separator attached."""
        parts = text.split(separator)
        result: list[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            elif part:
                result.append(part)
        return result

    def _merge_with_overlap(self, texts: list[str]) -> list[str]:
        """Merge small adjacent texts and add overlap context."""
        if not texts:
            return []

        merged: list[str] = []
        current = texts[0]

        for text in texts[1:]:
            combined = current + text
            if len(combined) <= self.config.chunk_size:
                current = combined
            else:
                merged.append(current)
                # Add overlap from the end of the previous chunk
                if self.config.chunk_overlap > 0:
                    overlap_text = current[-self.config.chunk_overlap :]
                    current = overlap_text + text
                else:
                    current = text

        if current.strip():
            merged.append(current)

        return merged
