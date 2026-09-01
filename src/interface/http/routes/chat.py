import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends

from src._lib.container import Container, get_container
from src.application.agents.escalation_agent import EscalationAgent
from src.application.caching.conversation_history_service import ConversationHistoryService
from src.application.caching.semantic_cache_service import SemanticCacheService
from src.application.guardrails.input_guardrail import InputGuardrail
from src.application.guardrails.output_guardrail import OutputGuardrail
from src.domain.entities.Agent_Response import AgentResponseModel
from src.domain.entities.Chat_Request import ChatRequest
from src.domain.shared.Agent_State import AgentState

_CACHEABLE_AGENTS = {"knowledge", "off_topic", "general_search"}

router = APIRouter(tags=["chat"])

#NOTE - PYBREAKER future addition for production circuit-breaking
#TODO - tool max attempts/repetitions
#TODO - try/catch errors back to the model (if applicable)

async def _graph(container: Annotated[Container, Depends(get_container)]):
    return await container.agent_graph.async_()

async def _escalation(container: Annotated[Container, Depends(get_container)]) -> EscalationAgent:
    return container.escalation_agent()

def _input_guard(container: Annotated[Container, Depends(get_container)]) -> InputGuardrail:
    return container.input_guardrail()

def _output_guard(container: Annotated[Container, Depends(get_container)]) -> OutputGuardrail:
    return container.output_guardrail()

def _cache(container: Annotated[Container, Depends(get_container)]) -> SemanticCacheService:
    return container.semantic_cache_service()

def _history_service(container: Annotated[Container, Depends(get_container)]) -> ConversationHistoryService:
    return container.history_service()


@router.post("/chat", response_model=AgentResponseModel)
async def chat(
    body: ChatRequest,
    background: BackgroundTasks,
    graph=Depends(_graph),
    escalation: EscalationAgent = Depends(_escalation),
    input_guard: InputGuardrail = Depends(_input_guard),
    output_guard: OutputGuardrail = Depends(_output_guard),
    cache: SemanticCacheService = Depends(_cache),
    history_service: ConversationHistoryService = Depends(_history_service)
) -> AgentResponseModel:
    """Run the agent orchestration graph and return a structured response."""
    guard_result = await input_guard.check(body.message)
    if guard_result.blocked:
        return AgentResponseModel.build(
            answer=guard_result.safe_response,
            source_agent="guardrail",
        )

    cached_json = await cache.get(body.message)
    if cached_json:
        return AgentResponseModel.model_validate_json(cached_json)

    history = await history_service.get(body.session_id or "")

    state: AgentState = {
        "messages": [body.message],
        "history": history,
        "user_id": body.user_id,
        "session_id": body.session_id or str(uuid.uuid4()),
    }
    result = await graph.ainvoke(state)
    raw: dict = result.get("response") or {}
    answer = str(raw.get("answer") or "")

    await history_service.append(state["session_id"], "user", body.message)
    await history_service.append(state["session_id"], "assistant", answer)
    source_agent = str(raw.get("source_agent") or "unknown")
    context: str = result.get("context") or ""

    if source_agent == "knowledge" and context:
        answer = await output_guard.apply(
            question=body.message, answer=answer, context=context
        )

    response = AgentResponseModel.build(
        answer=answer,
        source_agent=source_agent,
        sources=raw.get("sources") or [],
    )

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
