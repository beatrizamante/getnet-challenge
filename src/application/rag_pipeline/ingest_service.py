import hashlib
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.domain.entities.Chunk import Chunk
from src.domain.ports.Vector_Store_Port import VectorStorePort

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_OVERLAP = 50


class RagIngestService:
    """Splits raw text into semantically-aware overlapping chunks and upserts them to the vector store."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        self._vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def ingest(
        self,
        text: str,
        source: str,
        metadata: dict[str, str] | None = None,
    ) -> list[str]:
        """Chunk `text`, upsert all chunks, and return their IDs."""
        chunks = self._split(text, source, metadata or {})
        if not chunks:
            return []
        await self._vector_store.upsert(chunks)
        logger.info("Ingested %d chunks from source=%s", len(chunks), source)
        return [c.id for c in chunks]

    def _split(self, text: str, source: str, metadata: dict[str, str]) -> list[Chunk]:
        parts = self._splitter.split_text(text)
        return [
            Chunk(
                # Deterministic ID → upsert is naturally idempotent across re-ingestions
                id=hashlib.sha256(f"{source}:{i}".encode()).hexdigest()[:32],
                content=part,
                source=source,
                metadata={**metadata, "chunk_index": str(i)},
            )
            for i, part in enumerate(parts)
            if part.strip()
        ]
