import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from src.domain.entities.Agent_Response import AgentResponseModel
from src.domain.shared.Agent_State import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter

logger = logging.getLogger(__name__)

_ESCALATION_KEY = "escalations:{user_id}"


class EscalationAgent:
    """Returns an immediate handoff response; async logging runs separately as a BackgroundTask."""

    def __init__(
        self,
        langfuse: LangfuseAdapter,
        redis_client: aioredis.Redis,
        handoff_answer: str = "I'm transferring you to a human specialist who can better assist with your request. Please hold — someone from the Getnet support team will be with you shortly.",
        audit_ttl: int = 86400 * 30,
        max_entries: int = 100,
    ) -> None:
        self._langfuse = langfuse
        self._redis = redis_client
        self._handoff_answer = handoff_answer
        self._audit_ttl = audit_ttl
        self._max_entries = max_entries

    async def run(self, _: AgentState) -> dict:
        return {
            "response": AgentResponseModel.build(
                answer=self._handoff_answer,
                source_agent="escalate",
            ).model_dump()
        }

    async def log_escalation(self, user_id: str, message: str, reason: str = "") -> None:
        """Fire-and-forget: log escalation event to Langfuse + Redis audit list."""
        self._langfuse.trace(
            "escalation",
            {"user_id": user_id, "message": message, "reason": reason},
        )
        event = json.dumps(
            {
                "user_id": user_id,
                "message": message,
                "reason": reason,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
        )
        key = _ESCALATION_KEY.format(user_id=user_id)
        await self._redis.lpush(key, event)
        await self._redis.ltrim(key, 0, self._max_entries - 1)
        await self._redis.expire(key, self._audit_ttl)
        logger.info("Escalation logged. user_id=%s reason=%s", user_id, reason)

    async def get_audit_log(self, user_id: str) -> list[dict]:
        """Return the stored escalation events for a user (most recent first)."""
        key = _ESCALATION_KEY.format(user_id=user_id)
        raw: list[bytes] = await self._redis.lrange(key, 0, -1)  # type: ignore[misc]
        return [json.loads(entry) for entry in raw]
