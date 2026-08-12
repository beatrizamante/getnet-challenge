from typing import NotRequired, TypedDict


class AgentState(TypedDict):
    messages: list[str]
    user_id: str
    route: NotRequired[str | None]
    context: NotRequired[str]
    response: NotRequired[dict | None]
    session_id: NotRequired[str]  # propagated for Langfuse trace correlation
