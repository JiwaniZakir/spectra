"""Fixed-size chunking with token or character-based splitting."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from spectra.retrieval.base import Document


class SplitUnit(str, Enum):
    """Units for measuring chunk size."""

    CHARACTERS = "characters"
    TOKENS = "tokens"


class FixedChunkerConfig(BaseModel):
    """Configuration for fixed-size chunking."""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    split_unit: SplitUnit = SplitUnit.CHARACTERS
    tokenizer_model: str = "cl100k_base"
    strip_whitespace: bool = True

    model_config = {"frozen": True}


class FixedChunker:
    """Fixed-size document chunker.

    Splits documents into fixed-size chunks with configurable overlap.
    Supports both character-based and token-based splitting.

    Parameters
    ----------
    config:
        Chunker configuration.

    Example
    -------
    >>> chunker = FixedChunker(FixedChunkerConfig(chunk_size=100, chunk_overlap=20))
    >>> doc = Document(id="1", content="A " * 200)
    >>> chunks = chunker.chunk(doc)
    >>> len(chunks) > 1
    True
    """

    def __init__(self, config: FixedChunkerConfig | None = None) -> None:
        self.config = config or FixedChunkerConfig()
        self._tokenizer: object | None = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise ImportError(
                    "tiktoken is required for token-based chunking. "
                    "pip install tiktoken"
                ) from exc
            self._tokenizer = tiktoken.get_encoding(self.config.tokenizer_model)
        return self._tokenizer

    def chunk(self, document: Document) -> list[Document]:
        """Split *document* into fixed-size chunks.

        Parameters
        ----------
        document:
            The document to chunk.

        Returns
        -------
        list[Document]
            A list of chunk documents with metadata pointing back to
            the source document.
        """
        text = document.content
        if self.config.strip_whitespace:
            text = text.strip()

        if not text:
            return []

        if self.config.split_unit == SplitUnit.TOKENS:
            return self._chunk_by_tokens(text, document)
        return self._chunk_by_characters(text, document)

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents."""
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.chunk(doc))
        return chunks

    # ------------------------------------------------------------------
    # Character-based splitting
    # ------------------------------------------------------------------

    def _chunk_by_characters(self, text: str, source: Document) -> list[Document]:
        chunks: list[Document] = []
        stride = self.config.chunk_size - self.config.chunk_overlap
        stride = max(stride, 1)

        for start in range(0, len(text), stride):
            end = start + self.config.chunk_size
            chunk_text = text[start:end]
            if self.config.strip_whitespace:
                chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                Document(
                    id=f"{source.id}_chunk_{len(chunks)}",
                    content=chunk_text,
                    metadata={
                        **source.metadata,
                        "source_doc_id": source.id,
                        "chunk_index": len(chunks),
                        "char_start": start,
                        "char_end": min(end, len(text)),
                        "chunking_strategy": "fixed_character",
                    },
                )
            )

            if end >= len(text):
                break

        return chunks

    # ------------------------------------------------------------------
    # Token-based splitting
    # ------------------------------------------------------------------

    def _chunk_by_tokens(self, text: str, source: Document) -> list[Document]:
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(text)
        stride = self.config.chunk_size - self.config.chunk_overlap
        stride = max(stride, 1)

        chunks: list[Document] = []
        for start in range(0, len(tokens), stride):
            end = start + self.config.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            if self.config.strip_whitespace:
                chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                Document(
                    id=f"{source.id}_chunk_{len(chunks)}",
                    content=chunk_text,
                    metadata={
                        **source.metadata,
                        "source_doc_id": source.id,
                        "chunk_index": len(chunks),
                        "token_start": start,
                        "token_end": min(end, len(tokens)),
                        "chunking_strategy": "fixed_token",
                    },
                )
            )

            if end >= len(tokens):
                break

        return chunks
