import asyncio
import logging

from sentence_transformers import CrossEncoder

from src.domain.entities.Chunk import Chunk
from src.domain.ports.Reranker_Port import RerankerPort

logger = logging.getLogger(__name__)


class CrossEncoderReranker(RerankerPort):
    """RerankerPort backed by a sentence-transformers CrossEncoder.

    Distinct from the bi-encoder used for embeddings: the CrossEncoder reads
    (query, chunk) pairs *jointly*, producing calibrated relevance scores instead of
    comparing independent vectors. Loaded lazily; runs in a thread to keep the event loop free.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading cross-encoder reranker. model=%s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []
        model = self._get_model()
        pairs = [(query, c.content) for c in chunks]
        scores: list[float] = (await asyncio.to_thread(model.predict, pairs)).tolist()
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        logger.debug("Reranked %d → %d chunks. top_score=%.4f", len(chunks), top_n, ranked[0][0])
        return [chunk for _, chunk in ranked[:top_n]]
