import logging

from src.domain.entities.Route_Decision import RouteDecision
from src.domain.ports.LLM_Port import LLMPort
from src.domain.shared.Agent_State import AgentState
from src.infrastructure.config.prompt_catalog import PromptCatalog, load_prompt_catalog

logger = logging.getLogger(__name__)


class RouterAgent:
    """Pure classifier: reads a message and returns a routing decision. Never calls other agents."""

    def __init__(self, llm: LLMPort, prompts: PromptCatalog | None = None) -> None:
        self._llm = llm
        self._prompts = prompts or load_prompt_catalog()

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""
        try:
            decision: RouteDecision = await self._llm.complete_structured(
                prompt=user_message,
                schema=RouteDecision,
                system=self._prompts.router_system,
            )
            route = decision.target_agent
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Router failed to parse decision, defaulting to escalate. error=%s", exc)
            route = "escalate"
        logger.info("Router → route=%s", route)
        return {"route": route}
