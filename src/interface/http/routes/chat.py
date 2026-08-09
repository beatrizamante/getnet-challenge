import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from src._lib.container import Container, get_container
from src.application.agents.escalation_agent import EscalationAgent
from src.application.caching.semantic_cache_service import SemanticCacheService
from src.application.guardrails.input_guardrail import InputGuardrail
from src.application.guardrails.output_guardrail import OutputGuardrail
from src.domain.entities.Agent_Response import AgentResponseModel
from src.domain.shared.State import AgentState

# Responses from these agents are user-agnostic and safe to cache
_CACHEABLE_AGENTS = {"knowledge", "off_topic", "general_search"}

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str | None = None


def _graph(container: Annotated[Container, Depends(get_container)]):
    return container.agent_graph()


def _escalation(container: Annotated[Container, Depends(get_container)]) -> EscalationAgent:
    return container.escalation_agent()


def _input_guard(container: Annotated[Container, Depends(get_container)]) -> InputGuardrail:
    return container.input_guardrail()


def _output_guard(container: Annotated[Container, Depends(get_container)]) -> OutputGuardrail:
    return container.output_guardrail()


def _cache(container: Annotated[Container, Depends(get_container)]) -> SemanticCacheService:
    return container.semantic_cache_service()


@router.post("/chat", response_model=AgentResponseModel)
async def chat(
    body: ChatRequest,
    background: BackgroundTasks,
    graph=Depends(_graph),
    escalation: EscalationAgent = Depends(_escalation),
    input_guard: InputGuardrail = Depends(_input_guard),
    output_guard: OutputGuardrail = Depends(_output_guard),
    cache: SemanticCacheService = Depends(_cache),
) -> AgentResponseModel:
    """Run the agent orchestration graph and return a structured response."""
    guard_result = await input_guard.check(body.message)
    if guard_result.blocked:
        return AgentResponseModel(
            answer=guard_result.safe_response,
            source_agent="guardrail",
            sources=[],
        )

    # Semantic cache: skip graph for user-agnostic questions already answered
    cached_json = await cache.get(body.message)
    if cached_json:
        return AgentResponseModel.model_validate_json(cached_json)

    state: AgentState = {
        "messages": [body.message],
        "user_id": body.user_id,
        "session_id": body.session_id or str(uuid.uuid4()),
    }
    result = await graph.ainvoke(state)
    raw: dict = result.get("response") or {}
    answer = str(raw.get("answer") or "")
    source_agent = str(raw.get("source_agent") or "unknown")
    context: str = result.get("context") or ""

    if source_agent == "knowledge" and context:
        answer = await output_guard.apply(
            question=body.message, answer=answer, context=context
        )

    response = AgentResponseModel(
        answer=answer,
        source_agent=source_agent,
        sources=raw.get("sources") or [],
    )

    # Cache user-agnostic responses — customer_support and escalate are never cached
    if source_agent in _CACHEABLE_AGENTS:
        await cache.set(body.message, response.model_dump_json())

    if source_agent == "escalate":
        background.add_task(
            escalation.log_escalation,
            user_id=body.user_id,
            message=body.message,
            reason="router classified as escalate",
        )

    return response
