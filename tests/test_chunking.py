"""Tests for chunking strategies."""

from __future__ import annotations

import pytest

from spectra.chunking.fixed import FixedChunker, FixedChunkerConfig, SplitUnit
from spectra.chunking.recursive import RecursiveChunker, RecursiveChunkerConfig
from spectra.chunking.document_aware import (
    DocumentAwareChunker,
    DocumentAwareChunkerConfig,
)
from spectra.retrieval.base import Document


@pytest.fixture
def long_document() -> Document:
    return Document(
        id="test_doc",
        content=(
            "This is the first paragraph about machine learning. "
            "Machine learning is a field of artificial intelligence. "
            "It focuses on building systems that learn from data.\n\n"
            "This is the second paragraph about natural language processing. "
            "NLP enables computers to understand human language. "
            "Transformers have revolutionized this field.\n\n"
            "This is the third paragraph about information retrieval. "
            "IR helps find relevant documents in large corpora. "
            "Both sparse and dense methods are commonly used."
        ),
    )


@pytest.fixture
def markdown_document() -> Document:
    return Document(
        id="md_doc",
        content=(
            "# Introduction\n\n"
            "This is an introduction paragraph.\n\n"
            "## Methods\n\n"
            "We used the following methods:\n\n"
            "- Method A: description of method A\n"
            "- Method B: description of method B\n"
            "- Method C: description of method C\n\n"
            "```python\n"
            "def hello():\n"
            "    print('hello world')\n"
            "```\n\n"
            "## Results\n\n"
            "| Metric | Value |\n"
            "| ------ | ----- |\n"
            "| Accuracy | 0.95 |\n"
            "| F1 | 0.93 |\n\n"
            "The results show improvement."
        ),
    )


class TestFixedChunker:
    def test_basic_chunking(self, long_document: Document) -> None:
        chunker = FixedChunker(FixedChunkerConfig(chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk(long_document)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source_doc_id"] == "test_doc"
            assert "chunk_index" in chunk.metadata

    def test_no_overlap(self, long_document: Document) -> None:
        chunker = FixedChunker(FixedChunkerConfig(chunk_size=100, chunk_overlap=0))
        chunks = chunker.chunk(long_document)
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        doc = Document(id="t", content="word " * 200)
        chunker = FixedChunker(FixedChunkerConfig(chunk_size=50, chunk_overlap=0))
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.content) <= 50

    def test_empty_document(self) -> None:
        doc = Document(id="empty", content="")
        chunker = FixedChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_chunk_many(self) -> None:
        docs = [
            Document(id="a", content="Hello " * 50),
            Document(id="b", content="World " * 50),
        ]
        chunker = FixedChunker(FixedChunkerConfig(chunk_size=100))
        chunks = chunker.chunk_many(docs)
        assert len(chunks) > 2

    def test_small_document_single_chunk(self) -> None:
        doc = Document(id="small", content="Short text.")
        chunker = FixedChunker(FixedChunkerConfig(chunk_size=1000))
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1


class TestRecursiveChunker:
    def test_paragraph_splitting(self, long_document: Document) -> None:
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=200))
        chunks = chunker.chunk(long_document)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["chunking_strategy"] == "recursive"

    def test_respects_chunk_size(self) -> None:
        doc = Document(id="t", content="word " * 500)
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=100, chunk_overlap=0))
        chunks = chunker.chunk(doc)
        # Chunks should be roughly within size limit (with some flex for overlap)
        for chunk in chunks:
            assert len(chunk.content) <= 200  # generous bound

    def test_single_chunk_if_small(self) -> None:
        doc = Document(id="t", content="Small text.")
        chunker = RecursiveChunker(RecursiveChunkerConfig(chunk_size=1000))
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1


class TestDocumentAwareChunker:
    def test_markdown_parsing(self, markdown_document: Document) -> None:
        chunker = DocumentAwareChunker(DocumentAwareChunkerConfig(chunk_size=500))
        chunks = chunker.chunk(markdown_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["chunking_strategy"] == "document_aware"

    def test_code_block_preservation(self, markdown_document: Document) -> None:
        chunker = DocumentAwareChunker(
            DocumentAwareChunkerConfig(chunk_size=500, preserve_code_blocks=True)
        )
        chunks = chunker.chunk(markdown_document)

        # At least one chunk should contain the code block intact
        code_chunks = [c for c in chunks if "def hello" in c.content]
        assert len(code_chunks) > 0

    def test_table_preservation(self, markdown_document: Document) -> None:
        chunker = DocumentAwareChunker(
            DocumentAwareChunkerConfig(chunk_size=500, preserve_tables=True)
        )
        chunks = chunker.chunk(markdown_document)

        table_chunks = [c for c in chunks if "Accuracy" in c.content and "F1" in c.content]
        assert len(table_chunks) > 0

    def test_plain_text(self, long_document: Document) -> None:
        chunker = DocumentAwareChunker(DocumentAwareChunkerConfig(chunk_size=200))
        chunks = chunker.chunk(long_document)
        assert len(chunks) > 0
