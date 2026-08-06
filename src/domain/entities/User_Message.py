from typing import TypeAlias

from pydantic import BaseModel, Field


class UserMessageModel(BaseModel):
    """Incoming chat payload sent by the user."""

    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


UserMessage: TypeAlias = UserMessageModel
