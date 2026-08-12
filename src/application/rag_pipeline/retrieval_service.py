import logging

from src.domain.entities.Chunk import Chunk
from src.domain.ports.Reranker_Port import RerankerPort
from src.domain.ports.Vector_Store_Port import VectorStorePort

logger = logging.getLogger(__name__)

_DEFAULT_K = 4


class RagRetrievalService:
    """Two-stage retrieval: ChromaDB similarity search → optional cross-encoder reranking."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        default_k: int = _DEFAULT_K,
        reranker: RerankerPort | None = None,
        top_n: int = _DEFAULT_K,
        rerank_factor: int = 4,
    ) -> None:
        self._vector_store = vector_store
        self._default_k = default_k
        self._reranker = reranker
        self._top_n = top_n
        self._rerank_factor = rerank_factor

    async def retrieve_chunks(self, query: str, k: int | None = None) -> list[Chunk]:
        if self._reranker:
            # Fetch more candidates than needed, then rerank to top_n
            candidates_k = self._top_n * self._rerank_factor
            candidates = await self._vector_store.similarity_search(query, candidates_k)
            return await self._reranker.rerank(query, candidates, self._top_n)
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
