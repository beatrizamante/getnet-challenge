from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.application.agents.customer_support_agent import CustomerSupportAgent
from src.application.agents.escalation_agent import EscalationAgent
from src.application.agents.knowledge_agent import KnowledgeAgent
from src.application.agents.router_agent import RouterAgent
from src.domain.entities.Agent_Response import AgentResponseModel
from src.domain.shared.Agent_State import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.observability.tracing import traced_node


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
    off_topic_answer: str = "I'm only able to assist with questions related to Getnet's payment solutions and services. For other topics, please use a general-purpose search engine.",
) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(AgentState)

    async def _off_topic_node(_: AgentState) -> dict:
        return {"response": {"answer": off_topic_answer, "source_agent": "off_topic", "sources": []}}
    #NOTE - While we are adding a string object with a 'conversation memory' between agents here, it was done for simplicity sake, but there are better and more economic methods of doing it so.
    #NOTE - A summarizer is one of them. A small model running on the local server just to summarize the conversation.
    #NOTE - Another way is by summarizing the nth State into one paragraph (trim), while keeping the rest
    #NOTE - Using RAG for long chat contexts
    #NOTE - Different models for different agents (reasoning for router and knowledge, a mini model for customer support
    #NOTE - Semantic compressing for tokens is another one

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
    graph.add_edge("knowledge", "formatter")
    graph.add_edge("customer_support", "formatter")
    graph.add_edge("escalate", "formatter")
    graph.add_edge("off_topic", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()

#NOTE - For a bigger system, you could manage the memory to avoid exorbitant token costs and more compact states that fi context windows better, avoiding the Lost in the Middle effect
#NOTE - Fot this app specifically, there's no reason to use long term memory since it's basically a Q&A bot, but for more complex agents, long term memory is advisable
