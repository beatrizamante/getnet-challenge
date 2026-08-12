import asyncio
import logging

from sentence_transformers import SentenceTransformer

from src.domain.ports.Embedding_Port import EmbeddingPort
from src.infrastructure.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingAdapter(EmbeddingPort):
    """EmbeddingPort backed by a local SentenceTransformer model.

    The model is loaded once on first use; encode() runs in a thread so the
    event loop is never blocked.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._model_name = settings.model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model. model=%s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text; returns a unit-norm vector (cosine-ready)."""
        model = self._get_model()
        result = await asyncio.to_thread(
            model.encode, [text], normalize_embeddings=True, show_progress_bar=False
        )
        return result[0].tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one model call using native batching."""
        if not texts:
            return []
        model = self._get_model()
        result = await asyncio.to_thread(
            model.encode, texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [row.tolist() for row in result]
