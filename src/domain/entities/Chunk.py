from typing import TypeAlias

from pydantic import BaseModel, Field


class ChunkModel(BaseModel):
    """Document fragment stored and retrieved from the vector store."""

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)  # origin document or URL
    metadata: dict[str, str] = {}
    embedding: list[float] | None = None


Chunk: TypeAlias = ChunkModel
