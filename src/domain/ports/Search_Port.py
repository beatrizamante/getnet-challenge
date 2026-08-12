from abc import ABC, abstractmethod

from src.domain.entities.Search_Result import SearchResult


class SearchPort(ABC):
    """Contract for any external search provider (Tavily, SerpAPI, etc.)."""

    @abstractmethod
    async def search(self, query: str) -> list[SearchResult]:
        """Search an external source for results relevant to a query.

        Args:
            query: Search query submitted to the provider.

        Returns:
            Search results returned by the provider.
        """
