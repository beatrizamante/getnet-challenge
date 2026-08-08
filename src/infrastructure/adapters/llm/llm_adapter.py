import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.domain.ports.LLM_Port import LLMPort
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.config.settings import LLMSettings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0
# Retry on server errors and rate limits; client errors (4xx except 429) are not retried.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMAdapter(LLMPort):
    """LLMPort adapter."""

    def __init__(self, settings: LLMSettings, langfuse: LangfuseAdapter) -> None:
        self._model = OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(base_url=settings.base_url, api_key=settings.llm_api_key),
        )
        self._langfuse = langfuse

    async def complete(self, prompt: str, system: str) -> str:
        """Run a free-text completion and trace the call in Langfuse."""
        trace_id = self._langfuse.trace("llm.complete", {"prompt": prompt, "system": system})
        agent = Agent(self._model, instructions=system, retries=_MAX_RETRIES)
        result = await _with_retry(agent.run, prompt)
        output = str(result.output)
        self._langfuse.span(trace_id, "llm.complete", input_data={"prompt": prompt}, output={"output": output})
        return output

    async def complete_structured(self, prompt: str, schema: type[T]) -> T:
        """Run a structured completion whose output is validated against `schema`."""
        trace_id = self._langfuse.trace(
            "llm.complete_structured", {"prompt": prompt, "schema": schema.__name__}
        )
        agent: Agent[None, T] = Agent(self._model, output_type=schema, retries=_MAX_RETRIES)
        result = await _with_retry(agent.run, prompt)
        self._langfuse.span(
            trace_id,
            "llm.complete_structured",
            input_data={"prompt": prompt},
            output={"output": result.output.model_dump()},
        )
        return result.output


_R = TypeVar("_R")


async def _with_retry(fn: Callable[..., Awaitable[_R]], *args: Any, **kwargs: Any) -> _R:
    """Exponential backoff for transient HTTP errors (rate limits, 5xx)."""
    for attempt in range(_MAX_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(_BASE_DELAY * (2**attempt))
        except httpx.ConnectError:
            if attempt == _MAX_RETRIES - 1:
                raise
            await asyncio.sleep(_BASE_DELAY * (2**attempt))
    raise RuntimeError("_with_retry loop exhausted without returning")  # unreachable
