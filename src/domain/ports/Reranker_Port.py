from abc import ABC, abstractmethod

from src.domain.entities.Chunk import Chunk


class RerankerPort(ABC):
    """Contract for any cross-encoder reranker (local model, Cohere, etc.)."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Re-score `chunks` against `query` and return the top `top_n` by relevance."""
        ...
