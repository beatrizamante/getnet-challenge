# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock

import pytest

from src.application.caching.semantic_cache_service import SemanticCacheService
from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.LLM_Port import LLMPort


@pytest.fixture
def cache():
    return AsyncMock(spec=CachePort)


@pytest.fixture
def llm():
    return AsyncMock(spec=LLMPort)


@pytest.fixture
def service(cache, llm):
    return SemanticCacheService(cache=cache, llm=llm)


async def test_returns_cached_value_without_calling_llm(service, cache, llm):
    cache.get.return_value = "cached answer"

    result = await service.cached_complete("question", "system")

    assert result == "cached answer"
    llm.complete.assert_not_called()


async def test_calls_llm_on_cache_miss_and_stores_result(service, cache, llm):
    cache.get.return_value = None
    llm.complete.return_value = "llm answer"

    result = await service.cached_complete("question", "system", ttl=60)

    assert result == "llm answer"
    llm.complete.assert_awaited_once_with("question", "system")
    cache.set.assert_awaited_once_with("question", "llm answer", 60)
