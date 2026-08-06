from typing import TypeAlias

from pydantic import BaseModel, Field


class ConversationStateModel(BaseModel):
    """LangGraph state that flows through the agent graph."""

    messages: list[str]
    user_id: str = Field(min_length=1)
    route: str | None = None
    context: str

ConversationState: TypeAlias = ConversationStateModel
