import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import create_react_agent

from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.User_Repository_Port import UserRepositoryPort
from src.domain.shared.Application_Errors import UserNotFoundError
from src.domain.shared.State import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Getnet customer support specialist helping user {user_id}. "
    "ALWAYS call get_user_profile(user_id='{user_id}') first before answering. "
    "For transaction or settlement queries, call get_transaction_history. "
    "For a specific transaction settlement date, call get_settlement_estimate. "
    "Never assume facts — retrieve them with the available tools. "
    "Be empathetic, clear, and concise."
)

class CustomerSupportAgent:
    """ReAct agent with 3 tools: profile lookup, transaction history, settlement estimate."""

    def __init__(
        self,
        llm: LLMPort,
        user_repo: UserRepositoryPort,
        langfuse: LangfuseAdapter | None = None,
        _graph: Any = None,
    ) -> None:
        self._langfuse = langfuse
        tools = [
            _make_get_profile_tool(user_repo),
            _make_get_transactions_tool(user_repo),
            _make_get_settlement_tool(user_repo),
        ]
        self._graph = _graph or create_react_agent(llm.as_runnable(), tools=tools)

    async def run(self, state: AgentState) -> dict:
        user_id = state["user_id"]
        user_message = state["messages"][-1] if state.get("messages") else ""
        session_id = str(state.get("session_id", ""))

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT.format(user_id=user_id)),
            HumanMessage(content=user_message),
        ]

        callbacks = []
        if self._langfuse:
            handler = self._langfuse.get_callback_handler(
                user_id=user_id, session_id=session_id, trace_name="customer_support_agent"
            )
            if handler:
                callbacks.append(handler)

        config: RunnableConfig = {"callbacks": callbacks} if callbacks else {}
        result = await self._graph.ainvoke({"messages": messages}, config=config)

        final: AIMessage = result["messages"][-1]
        answer = final.content if isinstance(final.content, str) else str(final.content)
        return {
            "context": "",
            "response": {"answer": answer, "source_agent": "customer_support", "sources": []},
        }


def _make_get_profile_tool(user_repo: UserRepositoryPort):
    @lc_tool
    async def get_user_profile(user_id: str) -> str:
        """Retrieve the account profile (plan, machine model, status, join date) for the given user_id."""
        try:
            p = await user_repo.get_profile(user_id)
            if p is None:
                return f"No account found for user '{user_id}'."
            return (
                f"Plan: {p.plan}\n"
                f"Machine: {p.machine_model}\n"
                f"Status: {p.status}\n"
                f"Member since: {p.joined_at.date()}"
            )
        except UserNotFoundError:
            return f"No account found for user '{user_id}'."

    return get_user_profile


def _make_get_transactions_tool(user_repo: UserRepositoryPort):
    @lc_tool
    async def get_transaction_history(user_id: str, days: int = 7) -> str:
        """Retrieve the user's recent transactions including amount, status, and settlement date."""
        try:
            txs = await user_repo.get_transactions(user_id, days)
        except UserNotFoundError:
            return f"No account found for user '{user_id}'."
        if not txs:
            return "No transactions found in the specified period."
        lines = [
            f"• {t.id}: R${t.amount / 100:.2f} ({t.status}) "
            f"on {t.created_at.date()} → settles {t.settlement_date.date()}"
            for t in txs[:10]
        ]
        return "\n".join(lines)

    return get_transaction_history


def _make_get_settlement_tool(user_repo: UserRepositoryPort):
    @lc_tool
    async def get_settlement_estimate(transaction_id: str, user_id: str) -> str:
        """Get the settlement date for a specific transaction. Pass the transaction_id and the user_id."""
        try:
            txs = await user_repo.get_transactions(user_id, days=365)
        except UserNotFoundError:
            return f"No account found for user '{user_id}'."
        tx = next((t for t in txs if t.id == transaction_id), None)
        if tx is None:
            return f"Transaction '{transaction_id}' not found for user '{user_id}'."
        status_note = "" if tx.status != "pending" else " (still pending — date may change)"
        return (
            f"Transaction: {tx.id}\n"
            f"  Amount:          R${tx.amount / 100:.2f}\n"
            f"  Status:          {tx.status}\n"
            f"  Transaction date: {tx.created_at.date()}\n"
            f"  Settlement date:  {tx.settlement_date.date()}{status_note}"
        )

    return get_settlement_estimate
