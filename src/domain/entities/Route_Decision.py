from typing import TypeAlias

from pydantic import BaseModel, Field


class RouteDecisionModel(BaseModel):
    """Router Agent output used to dispatch to the correct specialized agent."""

    intent: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    target_agent: str = Field(min_length=1)

RouteDecision: TypeAlias = RouteDecisionModel
