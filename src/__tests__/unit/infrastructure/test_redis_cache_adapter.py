from unittest.mock import patch

import pytest
from redis.exceptions import RedisError

from src.infrastructure.adapters.cache.redis_cache_adapter import RedisCacheAdapter

_THRESHOLD = 0.92
_TTL = 60
_VALUE = '{"answer": "Getnet offers card machines."}'


@pytest.fixture
def adapter(redis_client, mock_semantic_embedding_port):
    return RedisCacheAdapter(
        client=redis_client,
        embedding_port=mock_semantic_embedding_port,
        similarity_threshold=_THRESHOLD,
        default_ttl=_TTL,
    )


async def test_get_returns_none_on_empty_cache(adapter):
    result = await adapter.get("original query")
    assert result is None


async def test_set_and_get_exact_same_query(adapter):
    await adapter.set("original query", _VALUE, _TTL)
    result = await adapter.get("original query")
    assert result == _VALUE


async def test_get_hits_on_semantically_similar_query(adapter):
    await adapter.set("original query", _VALUE, _TTL)
    # "similar query" has cos_sim ≈ 0.95 with "original query" → above 0.92 threshold
    result = await adapter.get("similar query")
    assert result == _VALUE


async def test_get_misses_on_dissimilar_query(adapter):
    await adapter.set("original query", _VALUE, _TTL)
    # "unrelated query" has cos_sim = 0.0 → below threshold
    result = await adapter.get("unrelated query")
    assert result is None


async def test_delete_removes_entry(adapter):
    await adapter.set("original query", _VALUE, _TTL)
    await adapter.delete("original query")
    result = await adapter.get("original query")
    assert result is None


async def test_set_applies_ttl(adapter, redis_client):
    await adapter.set("original query", _VALUE, _TTL)
    # Verify the entry key has a TTL set (> 0), not stored forever
    keys = await redis_client.smembers("semantic_cache:index")
    assert keys, "index should have one entry"
    entry_key = list(keys)[0].decode()
    ttl = await redis_client.ttl(entry_key)
    assert ttl > 0


async def test_get_degrades_gracefully_when_redis_is_down(
    redis_client, mock_semantic_embedding_port
):
    adapter = RedisCacheAdapter(
        client=redis_client,
        embedding_port=mock_semantic_embedding_port,
        similarity_threshold=_THRESHOLD,
    )
    with patch.object(redis_client, "smembers", side_effect=RedisError("connection refused")):
        result = await adapter.get("original query")
    assert result is None


async def test_set_degrades_gracefully_when_redis_is_down(
    redis_client, mock_semantic_embedding_port
):
    adapter = RedisCacheAdapter(
        client=redis_client,
        embedding_port=mock_semantic_embedding_port,
        similarity_threshold=_THRESHOLD,
    )
    with patch.object(redis_client, "hset", side_effect=RedisError("connection refused")):
        await adapter.set("original query", _VALUE, _TTL)  # must not raise


async def test_expired_key_pruned_from_index(adapter, redis_client):
    await adapter.set("original query", _VALUE, ttl=1)
    # Manually delete the entry (simulating TTL expiry) but leave it in the index
    keys = await redis_client.smembers("semantic_cache:index")
    entry_key = list(keys)[0].decode()
    await redis_client.delete(entry_key)

    # get() should clean the stale index entry and return None
    result = await adapter.get("original query")
    assert result is None
    remaining = await redis_client.smembers("semantic_cache:index")
    assert entry_key.encode() not in remaining
