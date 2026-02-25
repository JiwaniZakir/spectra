"""Graph-RAG -- Knowledge-graph-augmented retrieval.

Builds a lightweight entity-relation knowledge graph from documents and
traverses it at query time to gather structurally related context in
addition to (or instead of) pure vector similarity.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Sequence

from pydantic import BaseModel, Field

from spectra.retrieval.base import (
    BaseRetriever,
    Document,
    RetrievalResult,
    RetrieverCapability,
    RetrieverConfig,
)
from spectra.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class Triple(BaseModel):
    """A (subject, predicate, object) triple."""

    subject: str
    predicate: str
    object: str
    source_doc_id: str = ""


class GraphRAGConfig(RetrieverConfig):
    """Configuration for Graph-RAG retrieval."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    max_triples_per_doc: int = Field(default=20, ge=1)
    max_hops: int = Field(default=2, ge=1, le=5)
    entity_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class GraphRAGRetriever(BaseRetriever):
    """Knowledge-graph-augmented retrieval.

    Graph-RAG constructs a knowledge graph from ingested documents by
    extracting entity-relation triples.  At query time, entities are
    identified in the query and a local sub-graph is traversed to find
    relevant documents.

    The implementation uses:

    1. **LLM-based triple extraction** during indexing.
    2. **Entity linking** via fuzzy string matching (Jaccard on character
       trigrams).
    3. **Breadth-first graph traversal** to discover related documents.

    Parameters
    ----------
    config:
        Graph-RAG configuration.
    """

    strategy_name = "graph_rag"
    capabilities = frozenset(
        {RetrieverCapability.GRAPH, RetrieverCapability.GENERATIVE}
    )

    def __init__(self, config: GraphRAGConfig | None = None) -> None:
        self.config: GraphRAGConfig = config or GraphRAGConfig()
        super().__init__(self.config)
        self._llm = LLMClient(self.config.llm)
        self._documents: dict[str, Document] = {}
        self._triples: list[Triple] = []
        # entity -> set of doc IDs containing that entity
        self._entity_docs: dict[str, set[str]] = defaultdict(set)
        # Adjacency list: entity -> list of (predicate, related_entity, doc_id)
        self._adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Extract triples from *documents* and build the graph."""
        for doc in documents:
            self._documents[doc.id] = doc
            triples = self._extract_triples(doc)
            for triple in triples:
                self._triples.append(triple)
                subj = self._normalize_entity(triple.subject)
                obj = self._normalize_entity(triple.object)
                self._entity_docs[subj].add(doc.id)
                self._entity_docs[obj].add(doc.id)
                self._adjacency[subj].append((triple.predicate, obj, doc.id))
                self._adjacency[obj].append((triple.predicate, subj, doc.id))

    def _extract_triples(self, doc: Document) -> list[Triple]:
        """Use the LLM to extract entity-relation triples from a document."""
        prompt = (
            "Extract entity-relation triples from the following text. "
            "Return one triple per line in the format: subject | predicate | object\n"
            f"Extract at most {self.config.max_triples_per_doc} triples.\n\n"
            f"Text: {doc.content[:2000]}\n\n"
            "Triples:"
        )
        response = self._llm.complete(prompt, max_tokens=512)
        return self._parse_triples(response.content, doc.id)

    @staticmethod
    def _parse_triples(text: str, doc_id: str) -> list[Triple]:
        """Parse pipe-separated triple lines."""
        triples: list[Triple] = []
        for line in text.strip().splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                triples.append(
                    Triple(
                        subject=parts[0],
                        predicate=parts[1],
                        object=parts[2],
                        source_doc_id=doc_id,
                    )
                )
        return triples

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Identify query entities, traverse the graph, return relevant docs."""
        k = self._effective_top_k(top_k)

        # Step 1: Extract entities from the query
        query_entities = self._extract_query_entities(query)

        # Step 2: Link query entities to graph entities
        linked: list[str] = []
        for entity in query_entities:
            match = self._link_entity(entity)
            if match:
                linked.append(match)

        if not linked:
            # Fallback: return documents whose content overlaps with the query
            return self._fallback_retrieval(query, k)

        # Step 3: BFS traversal
        relevant_doc_ids = self._traverse(linked, max_hops=self.config.max_hops)

        # Step 4: Score documents by number of graph connections
        doc_scores: dict[str, float] = defaultdict(float)
        for doc_id, depth in relevant_doc_ids.items():
            # Closer documents get higher scores
            doc_scores[doc_id] = 1.0 / (1.0 + depth)

        # Normalize
        max_score = max(doc_scores.values()) if doc_scores else 1.0

        results: list[RetrievalResult] = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: -x[1])[:k]:
            if doc_id in self._documents:
                results.append(
                    RetrievalResult(
                        document=self._documents[doc_id],
                        score=score / max_score if max_score > 0 else 0.0,
                        strategy=self.strategy_name,
                        metadata={
                            "linked_entities": linked,
                            "graph_depth": relevant_doc_ids.get(doc_id, -1),
                        },
                    )
                )

        return self._filter_by_threshold(results)

    def _traverse(
        self, start_entities: list[str], max_hops: int
    ) -> dict[str, int]:
        """BFS from *start_entities*, return {doc_id: min_depth}."""
        visited_entities: set[str] = set()
        doc_depths: dict[str, int] = {}
        frontier: list[tuple[str, int]] = [(e, 0) for e in start_entities]

        while frontier:
            entity, depth = frontier.pop(0)
            if entity in visited_entities or depth > max_hops:
                continue
            visited_entities.add(entity)

            for doc_id in self._entity_docs.get(entity, set()):
                if doc_id not in doc_depths or depth < doc_depths[doc_id]:
                    doc_depths[doc_id] = depth

            if depth < max_hops:
                for _, neighbor, _ in self._adjacency.get(entity, []):
                    if neighbor not in visited_entities:
                        frontier.append((neighbor, depth + 1))

        return doc_depths

    # ------------------------------------------------------------------
    # Entity extraction and linking
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity mentions from the query using the LLM."""
        prompt = (
            "Extract the main entities (people, places, organizations, concepts) "
            "from the following query. Return one entity per line.\n\n"
            f"Query: {query}\n\n"
            "Entities:"
        )
        response = self._llm.complete(prompt, max_tokens=128)
        entities = [
            line.strip().strip("-").strip()
            for line in response.content.strip().splitlines()
            if line.strip()
        ]
        return entities

    def _link_entity(self, entity: str) -> str | None:
        """Link a query entity to a graph entity using trigram similarity."""
        entity_norm = self._normalize_entity(entity)
        best_match: str | None = None
        best_sim = 0.0

        for graph_entity in self._entity_docs:
            sim = self._trigram_similarity(entity_norm, graph_entity)
            if sim > best_sim:
                best_sim = sim
                best_match = graph_entity

        if best_sim >= self.config.entity_similarity_threshold:
            return best_match
        return None

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        return entity.lower().strip()

    @staticmethod
    def _trigram_similarity(a: str, b: str) -> float:
        """Jaccard similarity over character trigrams."""
        if not a or not b:
            return 0.0
        trigrams_a = {a[i : i + 3] for i in range(len(a) - 2)}
        trigrams_b = {b[i : i + 3] for i in range(len(b) - 2)}
        if not trigrams_a or not trigrams_b:
            return 1.0 if a == b else 0.0
        intersection = trigrams_a & trigrams_b
        union = trigrams_a | trigrams_b
        return len(intersection) / len(union)

    def _fallback_retrieval(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Simple keyword overlap fallback when no entities match."""
        query_words = set(query.lower().split())
        scored: list[tuple[str, float]] = []
        for doc_id, doc in self._documents.items():
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((doc_id, overlap / len(query_words)))

        scored.sort(key=lambda x: -x[1])
        results: list[RetrievalResult] = []
        for doc_id, score in scored[:top_k]:
            results.append(
                RetrievalResult(
                    document=self._documents[doc_id],
                    score=score,
                    strategy=self.strategy_name,
                    metadata={"fallback": True},
                )
            )
        return results
