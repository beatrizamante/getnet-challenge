from pydantic import BaseModel, Field, field_validator

_MAX_ANSWER = 4000


class AgentResponseModel(BaseModel):
    """Structured output returned by a specialized agent."""

    answer: str = Field(min_length=1, max_length=_MAX_ANSWER)
    source_agent: str = Field(min_length=1)
    # Populated by offline evaluation (deepeval), not at inference time
    confidence: float | None = None
    sources: list[str]

    @field_validator("sources", mode="before")
    @classmethod
    def _cap_sources(cls, v: list) -> list:
        """Limit sources count and individual length to prevent exfiltration via source field."""
        return [s for s in v if isinstance(s, str) and len(s) <= 500][:10]

    @classmethod
    def build(
        cls,
        answer: str,
        source_agent: str,
        sources: list[str] | None = None,
    ) -> "AgentResponseModel":
        """Safe constructor: truncates answer and applies a fallback for empty responses."""
        safe_answer = (answer.strip() or "No answer provided.")[:_MAX_ANSWER]
        return cls(answer=safe_answer, source_agent=source_agent, sources=sources or [])


type AgentResponse = AgentResponseModel
