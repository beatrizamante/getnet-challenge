from unittest.mock import MagicMock

import pytest

from src.application.agents.escalation_agent import EscalationAgent


@pytest.mark.asyncio
async def test_escalation_log_persists_to_redis(fake_redis) -> None:
    """log_escalation writes a JSON entry that get_audit_log recovers with all fields."""
    agent = EscalationAgent(langfuse=MagicMock(), redis_client=fake_redis)

    await agent.log_escalation("u1", "preciso de ajuda urgente", "router_escalated")
    log = await agent.get_audit_log("u1")

    assert len(log) == 1
    assert log[0]["user_id"] == "u1"
    assert log[0]["reason"] == "router_escalated"


@pytest.mark.asyncio
async def test_escalation_log_empty_for_new_user(fake_redis) -> None:
    """get_audit_log returns an empty list for a user_id with no prior escalations."""
    agent = EscalationAgent(langfuse=MagicMock(), redis_client=fake_redis)

    log = await agent.get_audit_log("user_sem_historico")

    assert log == []
