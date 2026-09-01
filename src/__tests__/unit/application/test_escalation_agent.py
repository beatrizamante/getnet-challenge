import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.agents.escalation_agent import EscalationAgent
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter


@pytest.fixture
def langfuse():
    lf = MagicMock(spec=LangfuseAdapter)
    lf.trace.return_value = "trace-id"
    return lf


@pytest.fixture
def redis_client():
    r = AsyncMock()
    r.lrange.return_value = []
    return r


@pytest.fixture
def agent(langfuse, redis_client):
    return EscalationAgent(langfuse=langfuse, redis_client=redis_client)


async def test_run_returns_handoff_response_immediately(agent):
    state = {"messages": ["I need help"], "user_id": "u1"}
    result = await agent.run(state)

    assert result["response"]["source_agent"] == "escalate"
    assert result["response"]["answer"]
    assert isinstance(result["response"]["sources"], list)


async def test_run_does_not_call_langfuse_or_redis(agent, langfuse, redis_client):
    """run() is synchronous — logging happens separately as a BackgroundTask."""
    await agent.run({"messages": ["help"], "user_id": "u1"})

    langfuse.trace.assert_not_called()
    redis_client.lpush.assert_not_called()


async def test_log_escalation_traces_to_langfuse(agent, langfuse):
    await agent.log_escalation("u1", "unclear message", reason="ambiguous intent")

    langfuse.trace.assert_called_once()
    call_args = langfuse.trace.call_args
    assert call_args[0][0] == "escalation"
    payload = call_args[0][1]
    assert payload["user_id"] == "u1"
    assert payload["reason"] == "ambiguous intent"


async def test_log_escalation_pushes_to_redis(agent, redis_client):
    await agent.log_escalation("u1", "test message", reason="low confidence")

    redis_client.lpush.assert_called_once()
    key, raw_event = redis_client.lpush.call_args[0]
    assert key == "escalations:u1"
    event = json.loads(raw_event)
    assert event["user_id"] == "u1"
    assert event["message"] == "test message"
    assert "timestamp" in event


async def test_log_escalation_trims_and_expires_redis_list(agent, redis_client):
    await agent.log_escalation("u1", "msg")

    redis_client.ltrim.assert_called_once()
    redis_client.expire.assert_called_once()


async def test_get_audit_log_decodes_redis_entries(agent, redis_client):
    stored = json.dumps(
        {"user_id": "u1", "message": "help", "reason": "", "timestamp": "2026-08-08T00:00:00+00:00"}
    )
    redis_client.lrange.return_value = [stored.encode()]

    log = await agent.get_audit_log("u1")

    assert len(log) == 1
    assert log[0]["user_id"] == "u1"
    assert log[0]["message"] == "help"


async def test_get_audit_log_empty_for_unknown_user(agent, redis_client):
    redis_client.lrange.return_value = []
    log = await agent.get_audit_log("ghost")
    assert log == []
