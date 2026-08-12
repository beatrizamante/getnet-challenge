from typing import TypeAlias

from pydantic import BaseModel, Field


class AgentResponseModel(BaseModel):
    """Structured output returned by a specialized agent."""

    answer: str = Field(min_length=1)
    source_agent: str = Field(min_length=1)
    # Populated by offline evaluation (deepeval), not at inference time
    confidence: float | None = None
    sources: list[str]

AgentResponse: TypeAlias = AgentResponseModel
