# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from fastapi import FastAPI

from src._lib.container import get_container
from src.application.guardrails.input_guardrail import InputGuardrailResult
from src.interface.http.middleware.auth import TokenClaims, require_user
from src.interface.http.routes.chat import router as chat_router

_DEFAULT_CLAIMS = TokenClaims(sub="u1", role="user")


def _make_mock_container(
    guard_result: InputGuardrailResult,
    graph_result: dict | None = None,
) -> MagicMock:
    container = MagicMock()

    guard = MagicMock()
    guard.check = AsyncMock(return_value=guard_result)
    container.input_guardrail.return_value = guard

    out_guard = MagicMock()
    out_guard.apply = AsyncMock(return_value="passthrough")
    container.output_guardrail.return_value = out_guard

    cache_svc = MagicMock()
    cache_svc.get = AsyncMock(return_value=None)
    cache_svc.set = AsyncMock()
    container.semantic_cache_service.return_value = cache_svc

    hist_svc = MagicMock()
    hist_svc.get = AsyncMock(return_value=[])
    hist_svc.append = AsyncMock()
    container.history_service.return_value = hist_svc

    escalation = MagicMock()
    escalation.log_escalation = AsyncMock()
    container.escalation_agent.return_value = escalation

    default_graph_result = {
        "response": {"answer": "Getnet suporta Pix.", "source_agent": "knowledge", "sources": []},
        "context": "",
    }
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=graph_result or default_graph_result)
    container.agent_graph.async_ = AsyncMock(return_value=mock_graph)

    return container


def make_test_app(container: MagicMock, claims: TokenClaims = _DEFAULT_CLAIMS) -> FastAPI:
    """Minimal FastAPI app — no lifespan, auth stubbed out to the given claims."""
    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_container] = lambda: container
    app.dependency_overrides[require_user] = lambda: claims
    return app


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def passing_guard_container():
    return _make_mock_container(guard_result=InputGuardrailResult(blocked=False))


@pytest.fixture
def blocked_guard_container():
    return _make_mock_container(
        guard_result=InputGuardrailResult(
            blocked=True,
            reason="prompt_injection",
            safe_response="I'm unable to process that request. "
            "I'm here to help with questions about Getnet's payment solutions and services.",
        )
    )
