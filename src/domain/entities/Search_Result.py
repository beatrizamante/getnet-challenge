from typing import TypeAlias

from pydantic import BaseModel, Field


class SearchResultModel(BaseModel):
    """Single result returned by a web or document search."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    score: float | None = None  # relevance score when available


SearchResult: TypeAlias = SearchResultModel
