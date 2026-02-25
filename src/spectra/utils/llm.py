"""LLM client abstraction using litellm for provider-agnostic completions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout: float = 60.0
    max_retries: int = 3
    api_base: str | None = None
    api_key: str | None = None

    model_config = {"frozen": True}


class LLMResponse(BaseModel):
    """Structured response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Provider-agnostic LLM client backed by litellm.

    Supports OpenAI, Anthropic, Cohere, local models, and any provider that
    litellm supports.  Both synchronous and asynchronous completions are
    provided.

    Parameters
    ----------
    config:
        LLM configuration.  Defaults to ``gpt-4o-mini`` with temperature 0.

    Example
    -------
    >>> client = LLMClient()
    >>> resp = client.complete("What is RAG?")
    >>> print(resp.content)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    # ------------------------------------------------------------------
    # Synchronous
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a single synchronous completion.

        Parameters
        ----------
        prompt:
            The user prompt.
        system:
            Optional system message.
        temperature:
            Override the default temperature.
        max_tokens:
            Override the default max_tokens.
        """
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            ) from exc

        messages = self._build_messages(prompt, system)
        response = litellm.completion(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            num_retries=self.config.max_retries,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )

        return self._parse_response(response)

    def complete_batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
    ) -> list[LLMResponse]:
        """Run multiple completions synchronously."""
        return [self.complete(p, system=system) for p in prompts]

    # ------------------------------------------------------------------
    # Asynchronous
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run a single asynchronous completion.

        Parameters
        ----------
        prompt:
            The user prompt.
        system:
            Optional system message.
        temperature:
            Override the default temperature.
        max_tokens:
            Override the default max_tokens.
        """
        try:
            import litellm
        except ImportError as exc:
            raise ImportError(
                "litellm is required. Install with: pip install litellm"
            ) from exc

        messages = self._build_messages(prompt, system)
        response = await litellm.acompletion(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=self.config.timeout,
            num_retries=self.config.max_retries,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
        )
        return self._parse_response(response)

    async def acomplete_batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        max_concurrency: int = 8,
    ) -> list[LLMResponse]:
        """Run multiple completions concurrently with bounded parallelism."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _limited(prompt: str) -> LLMResponse:
            async with semaphore:
                return await self.acomplete(prompt, system=system)

        return list(await asyncio.gather(*[_limited(p) for p in prompts]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str, system: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        usage = response.usage or {}
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )
