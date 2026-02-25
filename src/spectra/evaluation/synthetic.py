"""Synthetic evaluation dataset generation.

Generates question-answer-context triples from a document corpus using
an LLM, enabling evaluation without human-annotated datasets.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Sequence

from pydantic import BaseModel, Field

from spectra.evaluation.metrics import EvaluationSample
from spectra.retrieval.base import Document
from spectra.utils.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


class SyntheticConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    num_questions_per_doc: int = Field(default=3, ge=1, le=20)
    question_types: list[str] = Field(
        default_factory=lambda: ["factual", "reasoning", "comparison", "multi_hop"]
    )
    max_context_length: int = Field(default=2000, ge=100)
    difficulty_levels: list[str] = Field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )
    seed: int = 42

    model_config = {"frozen": True}


class GeneratedQA(BaseModel):
    """A generated question-answer pair."""

    question: str
    answer: str
    context: str
    question_type: str = "factual"
    difficulty: str = "medium"
    source_doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyntheticGenerator:
    """LLM-based synthetic evaluation dataset generator.

    Generates diverse question-answer-context triples from a document
    corpus.  Supports multiple question types (factual, reasoning,
    comparison, multi-hop) and difficulty levels.

    Parameters
    ----------
    config:
        Generation configuration.

    Example
    -------
    >>> gen = SyntheticGenerator()
    >>> docs = [Document(id="1", content="Python was created by Guido van Rossum...")]
    >>> samples = gen.generate(docs, num_samples=5)
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self._llm = LLMClient(self.config.llm)
        self._rng = random.Random(self.config.seed)

    def generate(
        self,
        documents: Sequence[Document],
        num_samples: int | None = None,
    ) -> list[EvaluationSample]:
        """Generate synthetic QA samples from the given documents.

        Parameters
        ----------
        documents:
            Source documents to generate questions from.
        num_samples:
            Total number of samples to generate. If None, generates
            ``num_questions_per_doc * len(documents)`` samples.

        Returns
        -------
        list[EvaluationSample]
            Generated evaluation samples.
        """
        all_qa: list[GeneratedQA] = []

        for doc in documents:
            qas = self._generate_from_document(doc)
            all_qa.extend(qas)

        # Limit to requested number
        if num_samples is not None and len(all_qa) > num_samples:
            self._rng.shuffle(all_qa)
            all_qa = all_qa[:num_samples]

        return [
            EvaluationSample(
                query=qa.question,
                answer=qa.answer,
                contexts=[qa.context],
                ground_truth=qa.answer,
                metadata={
                    "question_type": qa.question_type,
                    "difficulty": qa.difficulty,
                    "source_doc_id": qa.source_doc_id,
                    "synthetic": True,
                },
            )
            for qa in all_qa
        ]

    def _generate_from_document(self, doc: Document) -> list[GeneratedQA]:
        """Generate QA pairs from a single document."""
        context = doc.content[: self.config.max_context_length]
        qas: list[GeneratedQA] = []

        for i in range(self.config.num_questions_per_doc):
            q_type = self._rng.choice(self.config.question_types)
            difficulty = self._rng.choice(self.config.difficulty_levels)

            prompt = self._build_generation_prompt(context, q_type, difficulty)
            response = self._llm.complete(prompt, max_tokens=512, temperature=0.7)
            parsed = self._parse_qa_response(response.content)

            if parsed:
                qas.append(
                    GeneratedQA(
                        question=parsed["question"],
                        answer=parsed["answer"],
                        context=context,
                        question_type=q_type,
                        difficulty=difficulty,
                        source_doc_id=doc.id,
                    )
                )

        return qas

    @staticmethod
    def _build_generation_prompt(
        context: str, question_type: str, difficulty: str
    ) -> str:
        """Build the generation prompt based on question type and difficulty."""
        type_instructions = {
            "factual": "Ask a straightforward factual question that can be answered from the text.",
            "reasoning": "Ask a question that requires reasoning or inference over the text.",
            "comparison": "Ask a question that requires comparing two or more things mentioned in the text.",
            "multi_hop": "Ask a question that requires combining information from multiple parts of the text.",
        }

        difficulty_instructions = {
            "easy": "The question should be simple and directly answerable.",
            "medium": "The question should require some careful reading.",
            "hard": "The question should be challenging and require deep understanding.",
        }

        instruction = type_instructions.get(question_type, type_instructions["factual"])
        diff_note = difficulty_instructions.get(difficulty, "")

        return (
            "Generate a question and its answer based on the following text. "
            f"{instruction} {diff_note}\n\n"
            "Respond in JSON format with keys 'question' and 'answer'.\n\n"
            f"Text:\n{context}\n\n"
            "JSON:"
        )

    @staticmethod
    def _parse_qa_response(text: str) -> dict[str, str] | None:
        """Parse the LLM's JSON response into question/answer."""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
                if "question" in data and "answer" in data:
                    return {
                        "question": str(data["question"]),
                        "answer": str(data["answer"]),
                    }
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def generate_adversarial(
        self,
        documents: Sequence[Document],
        num_samples: int = 10,
    ) -> list[EvaluationSample]:
        """Generate adversarial samples designed to challenge RAG pipelines.

        These include:
        - Questions whose answers are NOT in the provided context.
        - Questions with misleading phrasing.
        - Questions requiring information from multiple documents.

        Parameters
        ----------
        documents:
            Source documents.
        num_samples:
            Number of adversarial samples to generate.
        """
        all_samples: list[EvaluationSample] = []

        for _ in range(num_samples):
            if not documents:
                break

            doc = self._rng.choice(list(documents))
            context = doc.content[: self.config.max_context_length]

            prompt = (
                "Generate a challenging adversarial question about the following text. "
                "The question should be tricky -- it might ask about something NOT in "
                "the text, use misleading phrasing, or require external knowledge.\n\n"
                "Respond in JSON with keys:\n"
                "- 'question': the adversarial question\n"
                "- 'answer': the correct answer\n"
                "- 'is_answerable': whether the text actually contains enough info (true/false)\n\n"
                f"Text:\n{context}\n\n"
                "JSON:"
            )
            response = self._llm.complete(prompt, max_tokens=512, temperature=0.8)
            parsed = self._parse_qa_response(response.content)

            if parsed:
                all_samples.append(
                    EvaluationSample(
                        query=parsed["question"],
                        answer=parsed["answer"],
                        contexts=[context],
                        ground_truth=parsed["answer"],
                        metadata={
                            "question_type": "adversarial",
                            "source_doc_id": doc.id,
                            "synthetic": True,
                        },
                    )
                )

        return all_samples
