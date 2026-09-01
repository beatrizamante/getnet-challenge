import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request

from src._lib.container import Container, get_container
from src.application.agents.escalation_agent import EscalationAgent
from src.application.caching.conversation_history_service import ConversationHistoryService
from src.application.caching.semantic_cache_service import SemanticCacheService
from src.application.guardrails.input_guardrail import InputGuardrail
from src.application.guardrails.output_guardrail import OutputGuardrail
from src.domain.entities.Agent_Response import AgentResponse
from src.domain.entities.Chat_Request import ChatRequest
from src.domain.shared.Agent_State import AgentState
from src.interface.http.middleware.auth import TokenClaims, require_user
from src.interface.http.middleware.rate_limit import limiter

_CACHEABLE_AGENTS = {"knowledge", "off_topic", "general_search"}

router = APIRouter(tags=["chat"])

# NOTE - PYBREAKER future addition for production circuit-breaking
# TODO - try/catch errors back to the model (if applicable)


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


def _history_service(
    container: Annotated[Container, Depends(get_container)],
) -> ConversationHistoryService:
    return container.history_service()


@router.post("/chat", response_model=AgentResponse)
@limiter.limit("20/minute")
async def chat(
    _: Request,  # required by slowapi to resolve the rate-limit key
    body: ChatRequest,
    background: BackgroundTasks,
    claims: TokenClaims = Depends(require_user),
    graph=Depends(_graph),
    escalation: EscalationAgent = Depends(_escalation),
    input_guard: InputGuardrail = Depends(_input_guard),
    output_guard: OutputGuardrail = Depends(_output_guard),
    cache: SemanticCacheService = Depends(_cache),
    history_service: ConversationHistoryService = Depends(_history_service),
) -> AgentResponse:
    """Run the agent orchestration graph and return a structured response."""
    user_id = claims.sub
    message = body.message
    session_id = body.session_id or str(uuid.uuid4())

    guard_result = await input_guard.check(message)
    if guard_result.blocked:
        return AgentResponse.build(
            answer=guard_result.safe_response,
            source_agent="guardrail",
        )

    cached_json = await cache.get(message)
    if cached_json:
        return AgentResponse.model_validate_json(cached_json)

    history = await history_service.get(session_id)

    state: AgentState = {
        "messages": [message],
        "history": history,
        "user_id": user_id,
        "session_id": session_id,
    }
    result = await graph.ainvoke(state)
    raw: dict = result.get("response") or {}
    answer = str(raw.get("answer") or "")

    await history_service.append_exchange(session_id, message, answer)
    source_agent = str(raw.get("source_agent") or "unknown")
    context: str = result.get("context") or ""

    if source_agent == "knowledge" and context:
        answer = await output_guard.apply(question=body.message, answer=answer, context=context)

    response = AgentResponse.build(
        answer=answer,
        source_agent=source_agent,
        sources=raw.get("sources") or [],
    )

    if source_agent in _CACHEABLE_AGENTS:
        await cache.set(body.message, response.model_dump_json())

    if source_agent == "escalate":
        background.add_task(
            escalation.log_escalation,
            user_id=user_id,
            message=body.message,
            reason="router classified as escalate",
        )

    return response
