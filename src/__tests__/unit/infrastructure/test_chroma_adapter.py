import pytest

from src.domain.entities.Chunk import Chunk
from src.infrastructure.adapters.vector_store.chroma_adapter import ChromaAdapter

_SOURCE = "https://getnet.net"


@pytest.fixture
def adapter(chroma_client, mock_embedding_port, collection_name):
    return ChromaAdapter(
        client=chroma_client,
        embedding_port=mock_embedding_port,
        collection_name=collection_name,
    )


async def test_upsert_calls_embed_batch_with_all_contents(adapter, mock_embedding_port):
    chunks = [
        Chunk(id="1", content="Máquina de cartão Getnet", source=_SOURCE),
        Chunk(id="2", content="Pix Getnet", source=_SOURCE),
    ]
    await adapter.upsert(chunks)
    mock_embedding_port.embed_batch.assert_awaited_once_with(
        ["Máquina de cartão Getnet", "Pix Getnet"]
    )


async def test_similarity_search_calls_embed_with_query(adapter):
    await adapter.upsert([Chunk(id="1", content="Getnet card machine", source=_SOURCE)])
    await adapter.similarity_search("card machine", k=1)
    adapter._embedding.embed.assert_awaited_once_with("card machine")


async def test_upsert_and_search_returns_inserted_ids(adapter):
    chunks = [
        Chunk(id="a", content="Content A", source=_SOURCE),
        Chunk(id="b", content="Content B", source=_SOURCE),
        Chunk(id="c", content="Content C", source=_SOURCE),
    ]
    await adapter.upsert(chunks)
    results = await adapter.similarity_search("anything", k=3)
    assert {r.id for r in results} == {"a", "b", "c"}


async def test_upsert_empty_list_is_noop(adapter, mock_embedding_port):
    await adapter.upsert([])
    mock_embedding_port.embed_batch.assert_not_called()


async def test_upsert_idempotent_same_id(adapter):
    chunk = Chunk(id="dup", content="Duplicate content", source=_SOURCE)
    await adapter.upsert([chunk])
    await adapter.upsert([chunk])
    results = await adapter.similarity_search("Duplicate", k=10)
    assert len(results) == 1


async def test_similarity_search_on_empty_collection_returns_empty(adapter, mock_embedding_port):
    results = await adapter.similarity_search("anything", k=5)
    assert results == []
    mock_embedding_port.embed.assert_not_called()


async def test_similarity_search_k_clamped_to_collection_size(adapter):
    chunks = [Chunk(id=str(i), content=f"Content {i}", source=_SOURCE) for i in range(2)]
    await adapter.upsert(chunks)
    results = await adapter.similarity_search("query", k=10)
    assert len(results) == 2


async def test_source_and_metadata_survive_round_trip(adapter):
    chunk = Chunk(
        id="meta-1",
        content="Antecipação de recebíveis",
        source="https://getnet.net/antecipacao",
        metadata={"page_title": "Antecipação"},
    )
    await adapter.upsert([chunk])
    results = await adapter.similarity_search("antecipação", k=1)
    assert results[0].source == "https://getnet.net/antecipacao"
    assert results[0].metadata.get("page_title") == "Antecipação"
