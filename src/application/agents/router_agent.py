import logging

from src.domain.shared.Agent_State import AgentState
from src.domain.entities.Route_Decision import RouteDecisionModel
from src.domain.ports.LLM_Port import LLMPort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a routing classifier for Getnet's multi-agent customer support system.

Reason internally about which agent should handle the user's message, then output \
ONLY a JSON object with these fields:
  - intent: "knowledge" | "customer_support" | "general_search" | "escalate" | "off_topic"
  - target_agent: same value as intent (used for dispatch)
  - reasoning: one sentence explaining the decision, for logs — not for the user

## Intent definitions
knowledge         → Getnet product/fee/service/how-to questions, answered by internal KB.
customer_support  → Questions about the user's OWN account, transactions, machines, \
or settlement dates — requires user-specific data.
general_search    → Factual questions unrelated to Getnet requiring live web data \
(e.g. exchange rates).
off_topic         → Unrelated to payments or Getnet (weather, sports, etc.).
escalate          → Message is ambiguous, sensitive, or you genuinely can't tell.

## Disambiguation rules (the actual hard part)
- A question phrased in general terms about a product/feature ("how does X work?") \
→ knowledge, even if the underlying topic (Pix, antecipação, maquininha) also shows \
up in customer_support cases.
- A question phrased about the user's own situation ("minha maquininha", "meu \
recebimento", "minha venda") → customer_support, even if it overlaps with a KB topic.
- Anything needing judgment outside Getnet's product scope AND outside general web \
facts (legal/financial advice, fraud, disputes) → escalate. Don't guess.
- When confidence between customer_support and knowledge is low, prefer escalate \
over a wrong guess — a misrouted account question is worse than one extra hop.

## Edge-case examples
User: "When will the money from yesterday's sales be deposited?"
→ {{"intent":"customer_support","target_agent":"customer_support","reasoning":"Requires the user's own transaction and settlement data."}}

User: "How does receivables advance (antecipação) work with Getnet?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"General product explanation, not tied to this user's account."}}

User: "What's the euro exchange rate today?"
→ {{"intent":"general_search","target_agent":"general_search","reasoning":"Real-time financial data outside Getnet's KB scope."}}

User: "uhh i dunno"
→ {{"intent":"escalate","target_agent":"escalate","reasoning":"Message is too ambiguous to classify."}}
"""


class RouterAgent:
    """Pure classifier: reads a message and returns a routing decision. Never calls other agents."""

    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""
        try:
            decision: RouteDecisionModel = await self._llm.complete_structured(
                prompt=user_message,
                schema=RouteDecisionModel,
                system=_SYSTEM_PROMPT,
            )
            route = decision.target_agent
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Router failed to parse decision, defaulting to escalate. error=%s", exc)
            route = "escalate"
        logger.info("Router → route=%s", route)
        return {"route": route}
