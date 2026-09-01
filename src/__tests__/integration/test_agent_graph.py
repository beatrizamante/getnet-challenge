import pytest

from src.application.agents.graph import build_graph
from src.domain.entities.Agent_Response import AgentResponse
from src.domain.shared.Agent_State import AgentState


class _MockRouter:
    def __init__(self, route: str) -> None:
        self._route = route

    async def run(self, _: AgentState) -> dict:
        return {"route": self._route}


class _MockAgent:
    def __init__(self, source: str) -> None:
        self._source = source

    async def run(self, _: AgentState) -> dict:
        return {
            "context": "",
            "response": AgentResponse.build(
                answer=f"Resposta do agente {self._source}.", source_agent=self._source
            ).model_dump(),
        }


def _compile(route: str):
    return build_graph(
        router=_MockRouter(route),  # type: ignore[arg-type]
        knowledge=_MockAgent("knowledge"),  # type: ignore[arg-type]
        customer_support=_MockAgent("customer_support"),  # type: ignore[arg-type]
        escalation=_MockAgent("escalate"),  # type: ignore[arg-type]
        langfuse=None,
    )


@pytest.mark.asyncio
async def test_graph_routes_to_knowledge_agent() -> None:
    """Router returning 'knowledge' causes the knowledge agent to produce the final response."""
    graph = _compile("knowledge")
    state: AgentState = {"messages": ["Como funciona o Pix?"], "user_id": "u1", "session_id": "s1"}

    result = await graph.ainvoke(state)

    assert result["response"]["source_agent"] == "knowledge"


@pytest.mark.asyncio
async def test_graph_routes_to_escalation_agent() -> None:
    """Router returning 'escalate' causes the escalation agent to produce the final response."""
    graph = _compile("escalate")
    state: AgentState = {"messages": ["..."], "user_id": "u1", "session_id": "s1"}

    result = await graph.ainvoke(state)

    assert result["response"]["source_agent"] == "escalate"
