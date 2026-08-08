from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """LLM-generated content of an agent response — confidence is an evaluation metric, not self-reported."""

    answer: str = Field(min_length=1)


class AgentState(TypedDict):
    messages: list[str]
    user_id: str
    route: NotRequired[str | None]
    context: NotRequired[str]
    response: NotRequired[dict | None]
