import hashlib
import logging

import numpy as np
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.Embedding_Port import EmbeddingPort

logger = logging.getLogger(__name__)

_INDEX_KEY = "semantic_cache:index"
_PREFIX = "semantic_cache:"
#TODO - For now, the caching is only used for KB retrieval, but for a full on app, we'd need to make an index key for chat history and another for the knowledge base

class RedisCacheAdapter(CachePort):
    """Semantic cache backed by Redis: hits are resolved by embedding cosine similarity, not exact key equality."""

    def __init__(
        self,
        client: aioredis.Redis,
        embedding_port: EmbeddingPort,
        similarity_threshold: float = 0.92,
        default_ttl: int = 3600,
    ) -> None:
        self._redis = client
        self._embedding = embedding_port
        self._threshold = similarity_threshold
        self._default_ttl = default_ttl

    async def get(self, key: str) -> str | None:
        """Return the cached value whose stored embedding is closest to `key`, or None on miss or Redis failure."""
        try:
            query_vec = np.array(await self._embedding.embed(key), dtype=np.float32)
            index_keys = await self._redis.smembers(_INDEX_KEY)
            if not index_keys:
                return None

            best_score = -1.0
            best_value: bytes | str | None = None

            for raw in index_keys:
                entry_key = raw.decode() if isinstance(raw, bytes) else raw
                stored = await self._redis.hgetall(entry_key)
                if not stored:
                    await self._redis.srem(_INDEX_KEY, raw)
                    continue
                emb_bytes = stored.get(b"embedding") or stored.get("embedding")
                value = stored.get(b"value") or stored.get("value")
                if emb_bytes is None or value is None:
                    continue
                raw_bytes = emb_bytes if isinstance(emb_bytes, bytes) else emb_bytes.encode("latin-1")
                stored_vec = np.frombuffer(raw_bytes, dtype=np.float32)
                score = _cosine_similarity(query_vec, stored_vec)
                if score > best_score:
                    best_score = score
                    best_value = value

            if best_score >= self._threshold and best_value is not None:
                return best_value.decode() if isinstance(best_value, bytes) else best_value
            return None
        except RedisError as exc:
            logger.warning("Redis cache get bypassed. error=%s", exc)
            return None

    async def set(self, key: str, value: str, ttl: int = 0) -> None:
        """Embed `key`, store the (embedding, value) pair with TTL, and register it in the index."""
        try:
            emb_bytes = np.array(
                await self._embedding.embed(key), dtype=np.float32
            ).tobytes()
            entry_key = f"{_PREFIX}{_hash(key)}"
            await self._redis.hset(entry_key, mapping={"embedding": emb_bytes, "value": value})
            await self._redis.expire(entry_key, ttl or self._default_ttl)
            await self._redis.sadd(_INDEX_KEY, entry_key)
        except RedisError as exc:
            logger.warning("Redis cache set bypassed. error=%s", exc)

    async def delete(self, key: str) -> None:
        """Remove the entry for `key` from both the hash store and the similarity index."""
        try:
            entry_key = f"{_PREFIX}{_hash(key)}"
            await self._redis.delete(entry_key)
            await self._redis.srem(_INDEX_KEY, entry_key)
        except RedisError as exc:
            logger.warning("Redis cache delete bypassed. error=%s", exc)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
