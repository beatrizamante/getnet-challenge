import logging

from chromadb.api.async_api import AsyncClientAPI

from src.domain.entities.Chunk import Chunk
from src.domain.ports.Embedding_Port import EmbeddingPort
from src.domain.ports.Vector_Store_Port import VectorStorePort

logger = logging.getLogger(__name__)

#NOTE - In more complex programs, for instance an roleplay agent, you'd need to chunk the query as well and make a batch similarity search
class ChromaAdapter(VectorStorePort):
    """VectorStorePort backed by ChromaDB using cosine similarity over HuggingFace embeddings."""

    def __init__(
        self,
        client: AsyncClientAPI | None,  # None when ChromaDB is unavailable at startup
        embedding_port: EmbeddingPort,
        collection_name: str = "getnet_kb",
    ) -> None:
        self._client = client
        self._embedding = embedding_port
        self._collection_name = collection_name
        self._collection = None

    def _check_available(self) -> None:
        if self._client is None:
            raise RuntimeError("ChromaDB is not available. Start the ChromaDB service and restart.")

    async def ping(self) -> bool:
        """Return True if the ChromaDB server is reachable."""
        self._check_available()
        await self._client.heartbeat()  # type: ignore[union-attr]
        return True

    async def _get_collection(self):
        """Lazily resolve the ChromaDB collection, creating it if it does not exist yet."""
        self._check_available()
        if self._collection is None:
            self._collection = await self._client.get_or_create_collection(  # type: ignore[union-attr]
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def upsert(self, chunks: list[Chunk]) -> None:
        """Embed all chunks in a single batch call and upsert them; duplicate IDs overwrite existing entries."""
        self._check_available()
        if not chunks:
            return
        collection = await self._get_collection()
        texts = [c.content for c in chunks]
        embeddings = await self._embedding.embed_batch(texts)
        await collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=texts,
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )

    async def similarity_search(self, query: str, k: int) -> list[Chunk]:
        """Return the k most similar chunks; k is clamped to collection size to avoid a ChromaDB error."""
        collection = await self._get_collection()
        count = await collection.count()
        if count == 0:
            return []
        results = await collection.query(
            query_embeddings=[await self._embedding.embed(query)],  # type: ignore[arg-type]
            n_results=min(k, count),  # ChromaDB errors if n_results > collection size
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0]
        docs = (results["documents"] or [[]])[0]
        metas = (results["metadatas"] or [[]])[0]
        chunks = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            meta_dict = dict(meta)
            source = str(meta_dict.pop("source", ""))
            chunks.append(Chunk(
                id=chunk_id,
                content=doc,
                source=source,
                metadata={key: str(val) for key, val in meta_dict.items()},
            ))
        return chunks
