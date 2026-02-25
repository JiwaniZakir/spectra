"""Chain-of-Note (CoN) -- Reading-notes-augmented retrieval.

Implements the approach from Yu et al. (2023) where the model generates
sequential reading notes for each retrieved document, building up
structured understanding before answering.
"""

from __future__ import annotations

import logging
from typing import Sequence

from pydantic import BaseModel, Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)
from spectra.retrieval.dense import DenseRetriever, DenseRetrieverConfig
from spectra.utils.embeddings import EmbeddingConfig
from spectra.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class ReadingNote(BaseModel):
    """A structured reading note for a single document."""

    doc_id: str
    relevance: str  # "relevant", "partially_relevant", "irrelevant"
    key_info: str
    connection_to_query: str
    note_type: str = "standard"  # "standard", "conflicting", "complementary"


class ChainOfNoteConfig(RetrieverConfig):
    """Configuration for Chain-of-Note retrieval."""

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    initial_top_k: int = Field(default=10, ge=1)
    notes_batch_size: int = Field(default=5, ge=1)


class ChainOfNoteRetriever(BaseRetriever):
    """Chain-of-Note retrieval with reading-note generation.

    Chain-of-Note (CoN) augments standard retrieval with a note-taking
    phase.  After initial retrieval, the model generates a structured
    reading note for each candidate document that assesses:

    - Whether the document is relevant.
    - What key information it provides.
    - How it connects to the query.

    Documents are then re-scored based on the content of their reading
    notes, effectively performing an LLM-based reranking with explicit
    reasoning traces.

    The approach also handles three scenarios:

    1. **Direct match** -- a document directly answers the query.
    2. **Indirect match** -- a document provides useful context.
    3. **Unknown** -- no retrieved documents are relevant, requiring the
       model to rely on parametric knowledge.

    References
    ----------
    Yu et al., "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented
    Language Models", arXiv:2311.09210, 2023.

    Parameters
    ----------
    config:
        Chain-of-Note configuration.
    """

    strategy_name = "chain_of_note"
    capabilities = frozenset(
        {RetrieverCapability.DENSE, RetrieverCapability.GENERATIVE, RetrieverCapability.RERANKING}
    )

    def __init__(self, config: ChainOfNoteConfig | None = None) -> None:
        self.config: ChainOfNoteConfig = config or ChainOfNoteConfig()
        super().__init__(self.config)
        self._llm = LLMClient(self.config.llm)
        self._inner = DenseRetriever(
            DenseRetrieverConfig(
                top_k=self.config.initial_top_k,
                embedding=self.config.embedding,
            )
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index *documents* in the underlying dense retriever."""
        self._inner.add_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve documents, generate reading notes, and re-score."""
        k = self._effective_top_k(top_k)

        # Step 1: Initial dense retrieval
        candidates = self._inner.retrieve(query, top_k=self.config.initial_top_k)
        if not candidates:
            return []

        # Step 2: Generate reading notes in batches
        notes: list[ReadingNote] = []
        for i in range(0, len(candidates), self.config.notes_batch_size):
            batch = candidates[i : i + self.config.notes_batch_size]
            batch_notes = self._generate_notes(query, batch, notes)
            notes.extend(batch_notes)

        # Step 3: Re-score based on notes
        results: list[RetrievalResult] = []
        for candidate, note in zip(candidates, notes, strict=False):
            adjusted_score = self._compute_note_score(candidate.score, note)
            results.append(
                RetrievalResult(
                    document=candidate.document,
                    score=adjusted_score,
                    strategy=self.strategy_name,
                    metadata={
                        "reading_note": note.model_dump(),
                        "original_score": candidate.score,
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return self._filter_by_threshold(results[:k])

    # ------------------------------------------------------------------
    # Note generation
    # ------------------------------------------------------------------

    def _generate_notes(
        self,
        query: str,
        candidates: list[RetrievalResult],
        previous_notes: list[ReadingNote],
    ) -> list[ReadingNote]:
        """Generate reading notes for a batch of candidates."""
        # Build context from previous notes
        prev_context = ""
        if previous_notes:
            prev_summaries = "\n".join(
                f"- Doc {n.doc_id}: [{n.relevance}] {n.key_info[:100]}"
                for n in previous_notes[-3:]
            )
            prev_context = f"\nPrevious notes:\n{prev_summaries}\n"

        docs_text = "\n---\n".join(
            f"Document {i+1} (ID: {c.document.id}):\n{c.document.content[:500]}"
            for i, c in enumerate(candidates)
        )

        prompt = (
            "You are creating reading notes for retrieved documents. "
            "For each document, provide:\n"
            "1. RELEVANCE: relevant / partially_relevant / irrelevant\n"
            "2. KEY_INFO: The most important information from this document.\n"
            "3. CONNECTION: How this document relates to the query.\n"
            "4. TYPE: standard / conflicting / complementary\n\n"
            f"Query: {query}\n{prev_context}\n"
            f"Documents:\n{docs_text}\n\n"
            "Generate notes for each document, separated by blank lines. "
            "Format each note as:\n"
            "RELEVANCE: ...\nKEY_INFO: ...\nCONNECTION: ...\nTYPE: ...\n"
        )

        response = self._llm.complete(prompt, max_tokens=1024)
        return self._parse_notes(response.content, candidates)

    @staticmethod
    def _parse_notes(
        text: str, candidates: list[RetrievalResult]
    ) -> list[ReadingNote]:
        """Parse the LLM output into structured reading notes."""
        notes: list[ReadingNote] = []
        blocks = text.strip().split("\n\n")

        for i, candidate in enumerate(candidates):
            note = ReadingNote(
                doc_id=candidate.document.id,
                relevance="partially_relevant",
                key_info="",
                connection_to_query="",
                note_type="standard",
            )

            if i < len(blocks):
                block = blocks[i]
                for line in block.splitlines():
                    line_lower = line.strip().lower()
                    if line_lower.startswith("relevance:"):
                        val = line.split(":", 1)[1].strip().lower()
                        if val in ("relevant", "partially_relevant", "irrelevant"):
                            note.relevance = val
                    elif line_lower.startswith("key_info:"):
                        note.key_info = line.split(":", 1)[1].strip()
                    elif line_lower.startswith("connection:"):
                        note.connection_to_query = line.split(":", 1)[1].strip()
                    elif line_lower.startswith("type:"):
                        val = line.split(":", 1)[1].strip().lower()
                        if val in ("standard", "conflicting", "complementary"):
                            note.note_type = val

            notes.append(note)

        return notes

    @staticmethod
    def _compute_note_score(original_score: float, note: ReadingNote) -> float:
        """Adjust the original retrieval score based on the reading note."""
        multiplier = {
            "relevant": 1.2,
            "partially_relevant": 0.8,
            "irrelevant": 0.2,
        }.get(note.relevance, 0.5)

        type_bonus = {
            "complementary": 0.1,
            "standard": 0.0,
            "conflicting": -0.05,
        }.get(note.note_type, 0.0)

        adjusted = original_score * multiplier + type_bonus
        return max(0.0, min(1.0, adjusted))
