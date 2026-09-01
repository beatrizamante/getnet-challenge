# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock, patch

import pytest

from src.domain.entities.Search_Result import SearchResult
from src.infrastructure.adapters.search.tavily_adapter import TavilySearchAdapter
from src.infrastructure.config.settings import SearchSettings

_SETTINGS = SearchSettings(api_key="test-key", max_results=3)

_TAVILY_RESPONSE = {
    "results": [
        {
            "title": "Getnet Máquinas",
            "url": "https://getnet.net/maquinas",
            "content": "Conheça as maquininhas.",
            "score": 0.9,
        },
        {
            "title": "Pix Getnet",
            "url": "https://getnet.net/pix",
            "content": "Aceite Pix.",
            "score": 0.8,
        },
    ]
}


@pytest.fixture
def adapter():
    mock_client = AsyncMock()
    mock_client.search.return_value = _TAVILY_RESPONSE
    with patch(
        "src.infrastructure.adapters.search.tavily_adapter.AsyncTavilyClient",
        return_value=mock_client,
    ):
        return TavilySearchAdapter(_SETTINGS), mock_client


async def test_search_returns_list_of_search_results(adapter):
    inst, _ = adapter
    results = await inst.search("getnet card machine")
    assert len(results) == 2
    assert all(isinstance(r, SearchResult) for r in results)


async def test_search_maps_fields_correctly(adapter):
    inst, _ = adapter
    results = await inst.search("getnet")
    assert results[0].title == "Getnet Máquinas"
    assert results[0].url == "https://getnet.net/maquinas"
    assert results[0].snippet == "Conheça as maquininhas."
    assert results[0].score == 0.9


async def test_search_passes_max_results_to_client(adapter):
    inst, mock_client = adapter
    await inst.search("query")
    _, call_kwargs = mock_client.search.call_args
    assert call_kwargs.get("max_results") == _SETTINGS.max_results


async def test_search_returns_empty_on_api_failure():
    mock_client = AsyncMock()
    mock_client.search.side_effect = Exception("network error")
    with patch(
        "src.infrastructure.adapters.search.tavily_adapter.AsyncTavilyClient",
        return_value=mock_client,
    ):
        inst = TavilySearchAdapter(_SETTINGS)
        results = await inst.search("anything")
    assert results == []


async def test_search_handles_empty_results():
    mock_client = AsyncMock()
    mock_client.search.return_value = {"results": []}
    with patch(
        "src.infrastructure.adapters.search.tavily_adapter.AsyncTavilyClient",
        return_value=mock_client,
    ):
        inst = TavilySearchAdapter(_SETTINGS)
        results = await inst.search("no results query")
    assert results == []
