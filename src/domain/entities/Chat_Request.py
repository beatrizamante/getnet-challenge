from src.domain.entities.User_Message import UserMessage


class ChatRequest(UserMessage):
    """Extends UserMessage (user_id, message) with optional session tracking."""

    session_id: str | None = None
