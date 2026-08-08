from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.application.agents.customer_support_agent import CustomerSupportAgent
from src.application.agents.knowledge_agent import KnowledgeAgent
from src.application.agents.router_agent import RouterAgent
from src.domain.shared.state import AgentState


def _select_agent(state: AgentState) -> str:
    return state.get("route") or "knowledge"


def build_graph(
    router: RouterAgent,
    knowledge: KnowledgeAgent,
    customer_support: CustomerSupportAgent,
) -> CompiledStateGraph:
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("router", router.run)
    graph.add_node("knowledge", knowledge.run)
    graph.add_node("customer_support", customer_support.run)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _select_agent,
        {"knowledge": "knowledge", "customer_support": "customer_support"},
    )
    graph.add_edge("knowledge", END)
    graph.add_edge("customer_support", END)

    return graph.compile()
