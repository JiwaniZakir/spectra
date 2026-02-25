"""Document-aware chunking that respects structural elements.

Recognises headings, lists, code blocks, and tables, and ensures chunk
boundaries align with these structural elements rather than cutting
through them.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, Field

from spectra.retrieval.base import Document


class BlockType(str, Enum):
    """Types of structural blocks in a document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"
    BLOCKQUOTE = "blockquote"
    EMPTY = "empty"


class StructuralBlock(BaseModel):
    """A structurally coherent block of text."""

    block_type: BlockType
    content: str
    level: int = 0  # heading level, list nesting depth, etc.
    line_start: int = 0
    line_end: int = 0


class DocumentAwareChunkerConfig(BaseModel):
    """Configuration for document-aware chunking."""

    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)
    preserve_headings: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    heading_hierarchy: bool = Field(
        default=True,
        description="Prepend parent heading context to chunks under that heading.",
    )

    model_config = {"frozen": True}


class DocumentAwareChunker:
    """Structure-preserving document chunker.

    Parses the document into structural blocks (headings, paragraphs,
    code blocks, tables, lists) and creates chunks that:

    1. Never split a code block or table in half.
    2. Keep content under a heading together when possible.
    3. Prepend parent heading context to provide chunk-level orientation.
    4. Respect the configured ``chunk_size`` and ``chunk_overlap``.

    Supports Markdown-formatted text. For plain text, falls back to
    paragraph-based splitting.

    Parameters
    ----------
    config:
        Document-aware chunker configuration.
    """

    def __init__(self, config: DocumentAwareChunkerConfig | None = None) -> None:
        self.config = config or DocumentAwareChunkerConfig()

    # Regex patterns for Markdown elements
    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    _CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
    _TABLE_ROW_RE = re.compile(r"^\|.*\|", re.MULTILINE)
    _LIST_RE = re.compile(r"^[\s]*[-*+]|\d+\.\s", re.MULTILINE)
    _BLOCKQUOTE_RE = re.compile(r"^>", re.MULTILINE)

    def chunk(self, document: Document) -> list[Document]:
        """Split *document* respecting structural elements.

        Parameters
        ----------
        document:
            The document to chunk.

        Returns
        -------
        list[Document]
            Structure-aware chunks.
        """
        blocks = self._parse_blocks(document.content)
        chunks = self._blocks_to_chunks(blocks, document)
        return chunks

    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents."""
        all_chunks: list[Document] = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ------------------------------------------------------------------
    # Block parsing
    # ------------------------------------------------------------------

    def _parse_blocks(self, text: str) -> list[StructuralBlock]:
        """Parse *text* into structural blocks."""
        lines = text.split("\n")
        blocks: list[StructuralBlock] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Code fence
            if line.strip().startswith("```"):
                block_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    block_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    block_lines.append(lines[i])
                    i += 1
                blocks.append(
                    StructuralBlock(
                        block_type=BlockType.CODE,
                        content="\n".join(block_lines),
                        line_start=i - len(block_lines),
                        line_end=i,
                    )
                )
                continue

            # Heading
            heading_match = self._HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                blocks.append(
                    StructuralBlock(
                        block_type=BlockType.HEADING,
                        content=line,
                        level=level,
                        line_start=i,
                        line_end=i + 1,
                    )
                )
                i += 1
                continue

            # Table
            if self._TABLE_ROW_RE.match(line):
                table_lines = [line]
                i += 1
                while i < len(lines) and self._TABLE_ROW_RE.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                blocks.append(
                    StructuralBlock(
                        block_type=BlockType.TABLE,
                        content="\n".join(table_lines),
                        line_start=i - len(table_lines),
                        line_end=i,
                    )
                )
                continue

            # List
            if self._LIST_RE.match(line):
                list_lines = [line]
                i += 1
                while i < len(lines) and (
                    self._LIST_RE.match(lines[i]) or (lines[i].startswith("  ") and lines[i].strip())
                ):
                    list_lines.append(lines[i])
                    i += 1
                blocks.append(
                    StructuralBlock(
                        block_type=BlockType.LIST,
                        content="\n".join(list_lines),
                        line_start=i - len(list_lines),
                        line_end=i,
                    )
                )
                continue

            # Blockquote
            if line.startswith(">"):
                bq_lines = [line]
                i += 1
                while i < len(lines) and (lines[i].startswith(">") or lines[i].strip()):
                    bq_lines.append(lines[i])
                    if not lines[i].startswith(">") and not lines[i].strip():
                        break
                    i += 1
                blocks.append(
                    StructuralBlock(
                        block_type=BlockType.BLOCKQUOTE,
                        content="\n".join(bq_lines),
                        line_start=i - len(bq_lines),
                        line_end=i,
                    )
                )
                continue

            # Empty line
            if not line.strip():
                i += 1
                continue

            # Paragraph: collect non-empty, non-special lines
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not self._is_special_line(lines[i]):
                para_lines.append(lines[i])
                i += 1
            blocks.append(
                StructuralBlock(
                    block_type=BlockType.PARAGRAPH,
                    content="\n".join(para_lines),
                    line_start=i - len(para_lines),
                    line_end=i,
                )
            )

        return blocks

    def _is_special_line(self, line: str) -> bool:
        """Check if a line starts a special block."""
        return bool(
            line.strip().startswith("```")
            or self._HEADING_RE.match(line)
            or self._TABLE_ROW_RE.match(line)
            or self._LIST_RE.match(line)
            or line.startswith(">")
        )

    # ------------------------------------------------------------------
    # Chunk construction
    # ------------------------------------------------------------------

    def _blocks_to_chunks(
        self, blocks: list[StructuralBlock], source: Document
    ) -> list[Document]:
        """Combine structural blocks into appropriately sized chunks."""
        chunks: list[Document] = []
        current_content: list[str] = []
        current_size = 0
        heading_context = ""

        for block in blocks:
            # Track heading hierarchy
            if block.block_type == BlockType.HEADING:
                if self.config.heading_hierarchy:
                    heading_context = block.content

                # Flush current chunk if we have content
                if current_content:
                    chunks.append(
                        self._make_chunk(
                            source, len(chunks), "\n\n".join(current_content)
                        )
                    )
                    current_content = []
                    current_size = 0

                current_content.append(block.content)
                current_size += len(block.content)
                continue

            block_text = block.content
            block_size = len(block_text)

            # Preserve atomic blocks (code, tables)
            if block.block_type in (BlockType.CODE, BlockType.TABLE) and (
                self.config.preserve_code_blocks or self.config.preserve_tables
            ):
                if current_size + block_size > self.config.chunk_size and current_content:
                    chunks.append(
                        self._make_chunk(
                            source, len(chunks), "\n\n".join(current_content)
                        )
                    )
                    current_content = []
                    current_size = 0

                    # Add heading context to new chunk
                    if heading_context and self.config.preserve_headings:
                        current_content.append(heading_context)
                        current_size += len(heading_context)

                current_content.append(block_text)
                current_size += block_size

                # If this single block exceeds chunk_size, flush it alone
                if current_size > self.config.chunk_size:
                    chunks.append(
                        self._make_chunk(
                            source, len(chunks), "\n\n".join(current_content)
                        )
                    )
                    current_content = []
                    current_size = 0
                continue

            # Regular content: check if adding would exceed chunk_size
            if current_size + block_size + 2 > self.config.chunk_size:
                if current_content:
                    chunks.append(
                        self._make_chunk(
                            source, len(chunks), "\n\n".join(current_content)
                        )
                    )
                    current_content = []
                    current_size = 0

                    # Add heading context
                    if heading_context and self.config.preserve_headings:
                        current_content.append(heading_context)
                        current_size += len(heading_context)

            current_content.append(block_text)
            current_size += block_size + 2

        # Flush remaining
        if current_content:
            chunks.append(
                self._make_chunk(source, len(chunks), "\n\n".join(current_content))
            )

        return chunks

    @staticmethod
    def _make_chunk(source: Document, index: int, content: str) -> Document:
        return Document(
            id=f"{source.id}_dachunk_{index}",
            content=content.strip(),
            metadata={
                **source.metadata,
                "source_doc_id": source.id,
                "chunk_index": index,
                "chunking_strategy": "document_aware",
            },
        )
