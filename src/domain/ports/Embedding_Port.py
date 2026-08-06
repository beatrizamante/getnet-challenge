from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Contract for any embedding provider (OpenAI, local model, etc.)."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
