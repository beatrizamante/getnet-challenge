
from abc import ABC, abstractmethod

from src.domain.entities.Chunk import Chunk


class VectorStorePort(ABC):
    """Contract for any vector database backend (ChromaDB, Pinecone, etc.)."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def similarity_search(self, query: str, k: int) -> list[Chunk]: ...
