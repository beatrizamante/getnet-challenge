import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.domain.ports.LLM_Port import LLMPort
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.config.settings import LLMSettings

logger = logging.getLogger(__name__)

# Retry on server errors and rate limits; client errors (4xx except 429) are not retried.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMAdapter(LLMPort):
    """LLMPort backed by LangChain's ChatOpenAI — same model layer LangGraph uses."""

    def __init__(self, settings: LLMSettings, langfuse: LangfuseAdapter) -> None:
        self._model = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.api_key,  # type: ignore[arg-type]
            base_url=settings.base_url,
            max_retries=0,  # retries handled by _with_retry below
        )
        self._langfuse = langfuse
        self._max_retries = settings.max_retries
        self._base_delay = settings.base_delay

    def as_runnable(self) -> ChatOpenAI:
        """Expose the raw model so LangGraph agents can call bind_tools() on it."""
        return self._model

    async def complete(self, prompt: str, system: str) -> str:
        """Run a free-text completion and trace the call in Langfuse."""
        trace_id = self._langfuse.trace("llm.complete", {"prompt": prompt, "system": system})
        messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
        response = await _with_retry(
            self._model.ainvoke,
            messages,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
        )
        output = str(response.content)
        self._langfuse.span(
            trace_id, "llm.complete", input_data={"prompt": prompt}, output={"output": output}
        )
        return output

    async def complete_structured[T: BaseModel](
        self, prompt: str, schema: type[T], system: str = ""
    ) -> T:
        """Run a structured completion whose output is validated against `schema`."""
        trace_id = self._langfuse.trace(
            "llm.complete_structured", {"prompt": prompt, "schema": schema.__name__}
        )
        text = await self.complete(prompt=prompt, system=system)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        raw = match.group(0) if match else cleaned
        result = schema.model_validate(json.loads(raw))
        self._langfuse.span(
            trace_id,
            "llm.complete_structured",
            input_data={"prompt": prompt},
            output={"output": result.model_dump()},
        )
        return result


async def _with_retry[R](
    fn: Callable[..., Awaitable[R]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> R:
    """Exponential backoff for transient HTTP errors (rate limits, 5xx)."""
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in _RETRYABLE_STATUS or attempt == max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2**attempt))
        except httpx.ConnectError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2**attempt))
    raise RuntimeError("_with_retry loop exhausted without returning")  # unreachable
