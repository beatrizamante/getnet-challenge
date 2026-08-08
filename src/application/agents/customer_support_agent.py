import logging

from src.domain.shared.state import AgentOutput, AgentState
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.User_Repository_Port import UserRepositoryPort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Getnet customer support specialist. "
    "Use the user's account profile and recent transaction history to provide personalised, accurate support. "
    "Be empathetic, clear, and concise. Do not invent account details."
)

_TRANSACTION_DAYS = 30

class CustomerSupportAgent:
    """Personalised support agent: fetches user context before calling the LLM."""

    def __init__(self, llm: LLMPort, user_repo: UserRepositoryPort) -> None:
        self._llm = llm
        self._user_repo = user_repo

    async def run(self, state: AgentState) -> dict:
        user_id = state["user_id"]
        user_message = state["messages"][-1] if state.get("messages") else ""

        profile = await self._user_repo.get_profile(user_id)
        transactions = await self._user_repo.get_transactions(user_id, _TRANSACTION_DAYS)

        context_parts: list[str] = []
        if profile:
            context_parts.append(
                f"User Profile:\n"
                f"  - Plan: {profile.plan}\n"
                f"  - Machine: {profile.machine_model}\n"
                f"  - Status: {profile.status}\n"
                f"  - Member since: {profile.joined_at.date()}"
            )
        if transactions:
            tx_lines = [
                f"  • {t.id}: R${t.amount / 100:.2f} ({t.status}) on {t.created_at.date()}"
                for t in transactions[:10]
            ]
            context_parts.append("Recent Transactions:\n" + "\n".join(tx_lines))

        context = "\n\n".join(context_parts)
        prompt = (
            f"User context:\n{context}\n\nUser question: {user_message}"
            if context
            else user_message
        )
        output = await self._llm.complete_structured(prompt, AgentOutput, system=_SYSTEM_PROMPT)
        response = {
            "answer": output.answer,
            "source_agent": "customer_support",
            "sources": [],
        }
        return {"context": context, "response": response}
