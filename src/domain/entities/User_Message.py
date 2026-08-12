from typing import TypeAlias

from pydantic import BaseModel, Field


class UserMessageModel(BaseModel):
    """Incoming chat payload sent by the user."""
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=100)

UserMessage: TypeAlias = UserMessageModel
