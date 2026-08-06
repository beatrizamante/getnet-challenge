from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, Field


class UserProfileModel(BaseModel):
    """Account data used by the Customer Support Agent."""

    plan: str = Field(min_length=1)
    machine_model: str = Field(min_length=1)
    status: str = Field(min_length=1)
    joined_at: datetime

UserProfile: TypeAlias = UserProfileModel
