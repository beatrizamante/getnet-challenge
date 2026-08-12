from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.application.agents.customer_support_agent import CustomerSupportAgent
from src.application.agents.escalation_agent import EscalationAgent
from src.application.agents.knowledge_agent import KnowledgeAgent
from src.application.agents.router_agent import RouterAgent
from src.domain.entities.Agent_Response import AgentResponseModel
from src.domain.shared.state import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.observability.tracing import traced_node

_OFF_TOPIC_ANSWER = (
    "I'm only able to assist with questions related to Getnet's payment solutions and services. "
    "For other topics, please use a general-purpose search engine."
)


async def _off_topic_node(**_: object) -> dict:  # type: ignore[misc]
    return {"response": {"answer": _OFF_TOPIC_ANSWER, "source_agent": "off_topic", "sources": []}}


async def _formatter_node(state: AgentState) -> dict:
    """Validates and normalises the agent response through AgentResponseModel before the graph exits."""
    raw: dict = state.get("response") or {}
    try:
        validated = AgentResponseModel(
            answer=str(raw.get("answer") or "No answer provided."),
            source_agent=str(raw.get("source_agent") or "unknown"),
            sources=raw.get("sources") or [],
        )
    except Exception:  # pylint: disable=broad-except
        validated = AgentResponseModel(
            answer="An error occurred while processing your request.",
            source_agent="unknown",
            sources=[],
        )
    return {"response": validated.model_dump()}


def _select_agent(state: AgentState) -> str:
    route = state.get("route") or "knowledge"
    if route == "general_search":
        return "knowledge"
    if route == "escalate":
        return "escalate"
    if route == "off_topic":
        return "off_topic"
    if route in ("knowledge", "customer_support"):
        return route
    return "escalate"


def build_graph(
    router: RouterAgent,
    knowledge: KnowledgeAgent,
    customer_support: CustomerSupportAgent,
    escalation: EscalationAgent,
    langfuse: LangfuseAdapter | None = None,
) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(AgentState)

    def _wrap(name: str, fn: Any) -> Any:
        return traced_node(langfuse, name)(fn) if langfuse else fn

    graph.add_node("router", _wrap("router", router.run))
    graph.add_node("knowledge", _wrap("knowledge", knowledge.run))
    graph.add_node("customer_support", _wrap("customer_support", customer_support.run))
    graph.add_node("escalate", _wrap("escalate", escalation.run))
    graph.add_node("off_topic", _wrap("off_topic", _off_topic_node))
    graph.add_node("formatter", _wrap("formatter", _formatter_node))

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _select_agent,
        {
            "knowledge": "knowledge",
            "customer_support": "customer_support",
            "escalate": "escalate",
            "off_topic": "off_topic",
        },
    )
    # All agent nodes funnel through the formatter before the graph exits
    graph.add_edge("knowledge", "formatter")
    graph.add_edge("customer_support", "formatter")
    graph.add_edge("escalate", "formatter")
    graph.add_edge("off_topic", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
