import chromadb
import fakeredis.aioredis
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.Embedding_Port import EmbeddingPort

_COLLECTION_NAME = "test_collection"
# Fixed-dimension fake embedding — all tests share the same dim (4).
_FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


class _AsyncCollectionWrapper:
    """Makes a sync chromadb.Collection behave like an async collection."""

    def __init__(self, collection) -> None:
        self._col = collection

    async def upsert(self, *, ids, embeddings, documents, metadatas):
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    async def query(self, *, query_embeddings, n_results, include):
        return self._col.query(query_embeddings=query_embeddings, n_results=n_results, include=include)

    async def count(self) -> int:
        return self._col.count()


class _AsyncChromaClient:
    """Wraps sync EphemeralClient for async tests — no server required."""

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()

    async def get_or_create_collection(self, name: str, metadata=None):
        col = self._client.get_or_create_collection(name, metadata=metadata or {})
        return _AsyncCollectionWrapper(col)

    async def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass


@pytest.fixture
def chroma_client():
    """Fresh in-memory ChromaDB client per test — no state leaks between tests."""
    return _AsyncChromaClient()


@pytest_asyncio.fixture(autouse=True)
async def clean_chroma(chroma_client):
    """Ensure the test collection is absent before each test and cleaned up after."""
    await chroma_client.delete_collection(_COLLECTION_NAME)
    yield
    await chroma_client.delete_collection(_COLLECTION_NAME)


@pytest.fixture
def mock_embedding_port() -> EmbeddingPort:
    port = AsyncMock(spec=EmbeddingPort)
    port.embed.return_value = _FAKE_EMBEDDING
    port.embed_batch.side_effect = lambda texts: [_FAKE_EMBEDDING for _ in texts]
    return port


@pytest.fixture
def collection_name() -> str:
    return _COLLECTION_NAME


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------

# Vectors chosen so that:
#   "similar query" has cosine sim ≈ 0.95 with "original query" (above 0.92 threshold)
#   "unrelated query" has cosine sim = 0.0 with "original query"
_SEMANTIC_EMBEDDINGS: dict[str, list[float]] = {
    "original query": [1.0, 0.0, 0.0, 0.0],
    "similar query":  [0.95, 0.31, 0.0, 0.0],   # cos_sim ≈ 0.95 → HIT
    "unrelated query": [0.0, 0.0, 1.0, 0.0],    # cos_sim = 0.0  → MISS
}


@pytest.fixture
def redis_client():
    """Fresh in-memory FakeRedis per test — no state leaks between tests."""
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def mock_semantic_embedding_port() -> CachePort:
    """EmbeddingPort mock that returns distinct vectors per query for similarity testing."""
    port = AsyncMock(spec=EmbeddingPort)
    port.embed.side_effect = lambda text: _SEMANTIC_EMBEDDINGS.get(text, [0.5, 0.5, 0.0, 0.0])
    return port
