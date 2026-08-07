from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMPort(ABC):
    """Contract for any LLM provider (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def complete(self, prompt: str, system: str) -> str: ...

    @abstractmethod
    async def complete_structured(self, prompt: str, schema: type[T]) -> T: ...
