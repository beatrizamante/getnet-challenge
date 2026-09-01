from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["knowledge", "customer_support", "general_search", "escalate", "off_topic"]


class RouteDecision(BaseModel):
    """Router Agent output. confidence is intentionally absent — the LLM expresses uncertainty via intent=escalate."""

    intent: Intent
    target_agent: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
