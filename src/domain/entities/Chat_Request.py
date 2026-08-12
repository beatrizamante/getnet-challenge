from typing import TypeAlias

from src.domain.entities.User_Message import UserMessageModel


class ChatRequestModel(UserMessageModel):
    """Extends UserMessage (user_id, message) with optional session tracking."""

    session_id: str | None = None


ChatRequest: TypeAlias = ChatRequestModel
