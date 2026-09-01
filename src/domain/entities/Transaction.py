from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class TransactionModel(BaseModel):
    """Sale record stored and queried by the Financial Agent."""

    id: str = Field(min_length=1)
    amount: int = Field(gt=0)  # value in cents
    status: str = Field(min_length=1)
    created_at: datetime
    settlement_date: datetime

    @model_validator(mode="after")
    def settlement_after_creation(self) -> "TransactionModel":
        if self.settlement_date < self.created_at:
            raise ValueError("settlement_date must not be before created_at")
        return self


type Transaction = TransactionModel
