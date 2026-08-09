import logging

from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.LLM_Port import LLMPort

logger = logging.getLogger(__name__)

_DEFAULT_TTL = 3600


class SemanticCacheService:
    """Cache-aside wrapper: returns a cached LLM answer or invokes the LLM and stores the result."""

    def __init__(self, cache: CachePort, llm: LLMPort, default_ttl: int = _DEFAULT_TTL) -> None:
        self._cache = cache
        self._llm = llm
        self._default_ttl = default_ttl

    async def cached_complete(self, prompt: str, system: str, ttl: int | None = None) -> str:
        cached = await self._cache.get(prompt)
        if cached is not None:
            logger.debug("Semantic cache hit.")
            return cached
        answer = await self._llm.complete(prompt, system)
        await self._cache.set(prompt, answer, ttl or self._default_ttl)
        return answer

    async def get(self, key: str) -> str | None:
        """Semantic similarity lookup — returns cached value or None."""
        return await self._cache.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """Store value under a semantically-indexed key."""
        await self._cache.set(key, value, ttl or self._default_ttl)
