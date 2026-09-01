# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock

import pytest

from src.application.rag_pipeline.ingest_service import RagIngestService
from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.entities.Chunk import Chunk
from src.domain.ports.Reranker_Port import RerankerPort
from src.domain.ports.Vector_Store_Port import VectorStorePort


@pytest.fixture
def vector_store():
    return AsyncMock(spec=VectorStorePort)


# --- IngestService ---


class TestRagIngestService:
    async def test_splits_text_and_upserts(self, vector_store):
        service = RagIngestService(vector_store=vector_store, chunk_size=20, chunk_overlap=5)
        ids = await service.ingest("hello world this is a test sentence", source="doc.txt")

        assert len(ids) > 0
        vector_store.upsert.assert_awaited_once()
        chunks: list[Chunk] = vector_store.upsert.call_args[0][0]
        assert all(c.source == "doc.txt" for c in chunks)

    async def test_empty_text_returns_no_ids(self, vector_store):
        service = RagIngestService(vector_store=vector_store)
        ids = await service.ingest("   ", source="blank.txt")

        assert ids == []
        vector_store.upsert.assert_not_called()

    async def test_metadata_attached_to_chunks(self, vector_store):
        service = RagIngestService(vector_store=vector_store, chunk_size=100)
        await service.ingest("some content", source="x.txt", metadata={"lang": "pt"})

        chunks: list[Chunk] = vector_store.upsert.call_args[0][0]
        assert all(c.metadata.get("lang") == "pt" for c in chunks)


# --- RetrievalService ---


class TestRagRetrievalService:
    async def test_retrieve_returns_formatted_context(self, vector_store):
        vector_store.similarity_search.return_value = [
            Chunk(id="1", content="Getnet info", source="site.com"),
        ]
        service = RagRetrievalService(vector_store=vector_store)
        ctx = await service.retrieve("what is Getnet?")

        assert "Getnet info" in ctx
        assert "site.com" in ctx

    async def test_retrieve_returns_empty_string_when_no_chunks(self, vector_store):
        vector_store.similarity_search.return_value = []
        service = RagRetrievalService(vector_store=vector_store)
        ctx = await service.retrieve("something")

        assert ctx == ""

    async def test_retrieve_chunks_delegates_to_vector_store(self, vector_store):
        expected = [Chunk(id="a", content="x", source="s")]
        vector_store.similarity_search.return_value = expected
        service = RagRetrievalService(vector_store=vector_store)
        result = await service.retrieve_chunks("q", k=2)

        assert result == expected
        vector_store.similarity_search.assert_awaited_once_with("q", 2)

    async def test_retrieve_chunks_uses_reranker_when_provided(self, vector_store):
        candidates = [Chunk(id=str(i), content=f"chunk {i}", source="s") for i in range(8)]
        reranked = candidates[:2]
        vector_store.similarity_search.return_value = candidates

        reranker = AsyncMock(spec=RerankerPort)
        reranker.rerank.return_value = reranked

        service = RagRetrievalService(
            vector_store=vector_store, reranker=reranker, top_n=2, rerank_factor=4
        )
        result = await service.retrieve_chunks("q")

        # should fetch top_n * factor = 8 candidates, then rerank to 2
        vector_store.similarity_search.assert_awaited_once_with("q", 8)
        reranker.rerank.assert_awaited_once_with("q", candidates, 2)
        assert result == reranked

    async def test_retrieve_chunks_bypasses_reranker_when_none(self, vector_store):
        chunks = [Chunk(id="1", content="x", source="s")]
        vector_store.similarity_search.return_value = chunks

        service = RagRetrievalService(vector_store=vector_store, reranker=None)
        result = await service.retrieve_chunks("q", k=3)

        vector_store.similarity_search.assert_awaited_once_with("q", 3)
        assert result == chunks
