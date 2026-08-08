import logging
from typing import Any

from tavily import AsyncTavilyClient

from src.domain.entities.Search_Result import SearchResult
from src.domain.ports.Search_Port import SearchPort
from src.infrastructure.config.settings import SearchSettings

logger = logging.getLogger(__name__)


class TavilySearchAdapter(SearchPort):
    """SearchPort backed by Tavily — optimised for RAG retrieval over web content."""

    def __init__(self, settings: SearchSettings) -> None:
        self._client = AsyncTavilyClient(api_key=settings.api_key)
        self._max_results = settings.max_results

    async def search(self, query: str) -> list[SearchResult]:
        """Execute a web search and map Tavily results to domain SearchResult objects."""
        try:
            response: dict[str, Any] = await self._client.search(
                query, max_results=self._max_results
            )
            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score"),
                )
                for r in response.get("results", [])
            ]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Tavily search failed, returning empty results. error=%s", exc)
            return []
