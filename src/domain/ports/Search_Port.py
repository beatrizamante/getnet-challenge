from abc import ABC, abstractmethod

from src.domain.entities.Search_Result import SearchResult


class SearchPort(ABC):
    """Contract for any external search provider (Tavily, SerpAPI, etc.)."""

    @abstractmethod
    async def search(self, query: str) -> list[SearchResult]: ...
