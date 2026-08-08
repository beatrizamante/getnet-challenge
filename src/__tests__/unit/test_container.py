# pylint: disable=redefined-outer-name,protected-access
from unittest.mock import MagicMock, patch

import pytest

from src._lib.container import Container
from src.domain.ports.Embedding_Port import EmbeddingPort
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.Search_Port import SearchPort
from src.infrastructure.adapters.cache.redis_cache_adapter import RedisCacheAdapter
from src.infrastructure.adapters.embeddings.huggingface_adapter import HuggingFaceEmbeddingAdapter
from src.infrastructure.adapters.llm.llm_adapter import LLMAdapter
from src.infrastructure.adapters.search.tavily_adapter import TavilySearchAdapter


@pytest.fixture
def container():
    """Fresh container instance per test — avoids singleton state bleed."""
    return Container()


def test_langfuse_adapter_is_singleton(container):
    a = container.langfuse_adapter()
    b = container.langfuse_adapter()
    assert a is b


def test_embedding_port_resolves_to_huggingface_adapter(container):
    adapter = container.embedding_port()
    assert isinstance(adapter, HuggingFaceEmbeddingAdapter)


def test_llm_port_resolves_to_pydanticai_adapter(container):
    with patch("src.infrastructure.adapters.llm.llm_adapter.ChatOpenAI"):
        adapter = container.llm_port()
    assert isinstance(adapter, LLMAdapter)


def test_search_port_resolves_to_tavily_adapter(container):
    adapter = container.search_port()
    assert isinstance(adapter, TavilySearchAdapter)


def test_cache_port_resolves_to_redis_adapter(container):
    adapter = container.cache_port()
    assert isinstance(adapter, RedisCacheAdapter)


def test_override_search_port_with_mock(container):
    """Verify that overriding a provider injects the mock wherever it is consumed."""
    mock_search = MagicMock(spec=SearchPort)

    with container.search_port.override(mock_search):
        resolved = container.search_port()

    assert resolved is mock_search


def test_override_embedding_port_with_mock(container):
    mock_embedding = MagicMock(spec=EmbeddingPort)

    with container.embedding_port.override(mock_embedding):
        # cache_port depends on embedding_port — it should receive the mock
        cache = container.cache_port()
        assert cache._embedding is mock_embedding


def test_override_llm_port_with_mock(container):
    mock_llm = MagicMock(spec=LLMPort)

    with container.llm_port.override(mock_llm):
        resolved = container.llm_port()

    assert resolved is mock_llm
