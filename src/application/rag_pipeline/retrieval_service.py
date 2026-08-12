import logging

from src.domain.entities.Chunk import Chunk
from src.domain.ports.Vector_Store_Port import VectorStorePort

logger = logging.getLogger(__name__)

_DEFAULT_K = 4


class RagRetrievalService:
    """Retrieves relevant chunks from the vector store for a given query."""

    def __init__(self, vector_store: VectorStorePort, default_k: int = _DEFAULT_K) -> None:
        self._vector_store = vector_store
        self._default_k = default_k

    async def retrieve_chunks(self, query: str, k: int | None = None) -> list[Chunk]:
        return await self._vector_store.similarity_search(query, k or self._default_k)

    async def retrieve(self, query: str, k: int | None = None) -> str:
        """Return a formatted context string from the top-k most similar chunks."""
        chunks = await self.retrieve_chunks(query, k)
        if not chunks:
            return ""
        return _format_context(chunks)


def _format_context(chunks: list[Chunk]) -> str:
    parts = [f"[Source: {c.source}]\n{c.content}" for c in chunks]
    return "\n\n---\n\n".join(parts)
