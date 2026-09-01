from datetime import datetime

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Account data used by the Customer Support Agent."""

    plan: str = Field(min_length=1)
    machine_model: str = Field(min_length=1)
    status: str = Field(min_length=1)
    joined_at: datetime
