import logging

from src.domain.shared.state import AgentState
from src.domain.entities.Route_Decision import RouteDecisionModel
from src.domain.ports.LLM_Port import LLMPort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a router agent for Getnet's customer support system. "
    "Analyse the user message and decide which specialised agent should handle it. "
    "Available agents: "
    "  - 'knowledge'        → product info, services, FAQs, how-to questions. "
    "  - 'customer_support' → account issues, transactions, billing, status checks. "
    "Respond with a JSON object containing 'intent' (string), 'confidence' (float 0–1), "
    "and 'target_agent' (one of the two values above)."
)

class RouterAgent:
    """Classifies incoming messages and populates state['route'] with the target agent name."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""
        decision: RouteDecisionModel = await self._llm.complete_structured(
            prompt=user_message,
            schema=RouteDecisionModel,
            system=_SYSTEM_PROMPT,
        )
        logger.info(
            "Router → target=%s intent=%s",
            decision.target_agent,
            decision.intent
        )
        return {"route": decision.target_agent}
