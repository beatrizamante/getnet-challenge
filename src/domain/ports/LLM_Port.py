from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMPort(ABC):
    """Contract for any LLM provider (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str: ...

    @abstractmethod
    async def complete_structured(self, prompt: str, schema: type[T], system: str = "") -> T: ...

    @abstractmethod
    def as_runnable(self) -> Any:
        """Return the underlying LangChain runnable for tool binding in LangGraph agents."""
        ...
