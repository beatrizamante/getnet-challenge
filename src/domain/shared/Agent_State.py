from typing import Literal, NotRequired, TypedDict


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict):
    messages: list[str]
    history: NotRequired[list[Turn]]
    user_id: str
    route: NotRequired[str | None]
    context: NotRequired[str]
    response: NotRequired[dict | None]
    session_id: NotRequired[str]  # propagated for Langfuse trace correlation
