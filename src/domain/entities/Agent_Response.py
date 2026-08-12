from typing import TypeAlias

from pydantic import BaseModel, Field, field_validator


class AgentResponseModel(BaseModel):
    """Structured output returned by a specialized agent."""

    answer: str = Field(min_length=1, max_length=4000)
    source_agent: str = Field(min_length=1)
    # Populated by offline evaluation (deepeval), not at inference time
    confidence: float | None = None
    sources: list[str]

    @field_validator("sources", mode="before")
    @classmethod
    def _cap_sources(cls, v: list) -> list:
        """Limit sources count and individual length to prevent exfiltration via source field."""
        return [s for s in v if isinstance(s, str) and len(s) <= 500][:10]

AgentResponse: TypeAlias = AgentResponseModel
