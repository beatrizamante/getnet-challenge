import logging

from src.domain.shared.State import AgentState
from src.domain.entities.Route_Decision import RouteDecisionModel
from src.domain.ports.LLM_Port import LLMPort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a routing classifier for Getnet's multi-agent customer support system.
Analyse the user message and respond with a JSON object with these fields:
  - intent: one of "knowledge" | "customer_support" | "general_search" | "escalate" | "off_topic"
  - target_agent: same as intent (used for dispatch)
  - reasoning: one sentence explaining the decision

Intent definitions:
  knowledge        → questions about Getnet products, fees, services, how-to (answered by internal KB)
  customer_support → questions about the user’s own account, transactions, machines, settlement dates
  general_search   → factual questions unrelated to Getnet that require web search (e.g. exchange rates)
  off_topic        → completely unrelated to payments or Getnet (e.g. weather, sports)
  escalate         → use this when the message is ambiguous, sensitive, or you genuinely cannot tell

Few-shot examples (based on real challenge scenarios):
User: "What’s the difference between the Get Clássica and the Get Smart?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"Product comparison answered by Getnet knowledge base."}}

User: "What’s the weather forecast in Porto Alegre tomorrow?"
→ {{"intent":"off_topic","target_agent":"off_topic","reasoning":"Completely unrelated to Getnet or payment services."}}

User: "When will the money from yesterday’s sales be deposited?"
→ {{"intent":"customer_support","target_agent":"customer_support","reasoning":"Requires the user’s own transaction and settlement data."}}

User: "Do I need a bank account to receive my sales via Pix?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"General Pix/product policy question answered by KB."}}

User: "My card machine won’t connect to the internet, what should I do?"
→ {{"intent":"customer_support","target_agent":"customer_support","reasoning":"Device troubleshooting tied to user’s specific machine model."}}

User: "How does receivables advance (antecipação) work with Getnet?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"Product/feature explanation from Getnet KB."}}

User: "What’s the euro exchange rate today?"
→ {{"intent":"general_search","target_agent":"general_search","reasoning":"Real-time financial data not in KB — requires web search."}}

User: "My card machine is showing a transaction decline error."
→ {{"intent":"customer_support","target_agent":"customer_support","reasoning":"Error tied to user’s account and device."}}

User: "How many installments can I split a sale into with the crediário?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"Product policy question answered by Getnet KB."}}

User: "Can I sell through WhatsApp using the Payment Link?"
→ {{"intent":"knowledge","target_agent":"knowledge","reasoning":"Payment Link feature question answered by KB."}}

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
