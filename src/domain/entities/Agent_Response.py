from typing import TypeAlias

from pydantic import BaseModel, Field


class AgentResponseModel(BaseModel):
    """Structured output returned by a specialized agent."""

    answer: str = Field(min_length=1)
    source_agent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]

AgentResponse: TypeAlias = AgentResponseModel
