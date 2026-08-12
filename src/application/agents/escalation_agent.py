import json
import logging
from datetime import datetime, timezone

import redis.asyncio as aioredis

from src.domain.shared.State import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter

logger = logging.getLogger(__name__)

_HANDOFF_ANSWER = (
    "I'm transferring you to a human specialist who can better assist with your request. "
    "Please hold — someone from the Getnet support team will be with you shortly."
)

_ESCALATION_KEY = "escalations:{user_id}"
_ESCALATION_TTL = 86400 * 30  # keep audit trail for 30 days
_MAX_ENTRIES = 100             # cap list length per user


class EscalationAgent:
    """Returns an immediate handoff response; async logging runs separately as a BackgroundTask."""

    def __init__(self, langfuse: LangfuseAdapter, redis_client: aioredis.Redis) -> None:
        self._langfuse = langfuse
        self._redis = redis_client

    async def run(self, state: AgentState) -> dict:
        return {"response": {"answer": _HANDOFF_ANSWER, "source_agent": "escalate", "sources": []}}

    async def log_escalation(self, user_id: str, message: str, reason: str = "") -> None:
        """Fire-and-forget: log escalation event to Langfuse + Redis audit list."""
        self._langfuse.trace(
            "escalation",
            {"user_id": user_id, "message": message, "reason": reason},
        )
        event = json.dumps({
            "user_id": user_id,
            "message": message,
            "reason": reason,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        })
        key = _ESCALATION_KEY.format(user_id=user_id)
        await self._redis.lpush(key, event)         # type: ignore[misc]
        await self._redis.ltrim(key, 0, _MAX_ENTRIES - 1)
        await self._redis.expire(key, _ESCALATION_TTL)
        logger.info("Escalation logged. user_id=%s reason=%s", user_id, reason)

    async def get_audit_log(self, user_id: str) -> list[dict]:
        """Return the stored escalation events for a user (most recent first)."""
        key = _ESCALATION_KEY.format(user_id=user_id)
        raw: list[bytes] = await self._redis.lrange(key, 0, -1)  # type: ignore[misc]
        return [json.loads(entry) for entry in raw]
