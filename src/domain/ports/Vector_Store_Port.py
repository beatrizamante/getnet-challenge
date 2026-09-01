from abc import ABC, abstractmethod

from src.domain.entities.Chunk import Chunk


class VectorStorePort(ABC):
    """Contract for any vector database backend (ChromaDB, Pinecone, etc.)."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the vector store.

        Args:
            chunks: Chunks to persist and index for similarity search.
        """

    @abstractmethod
    async def similarity_search(self, query: str, k: int) -> list[Chunk]:
        """Retrieve chunks most similar to a query.

        Args:
            query: Text used to find semantically similar chunks.
            k: Maximum number of chunks to return.

        Returns:
            Chunks selected by the vector store's similarity ranking.
        """
