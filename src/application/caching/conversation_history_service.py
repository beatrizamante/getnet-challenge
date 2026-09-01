import json
from typing import Literal
from src.domain.ports.Cache_Port import CachePort
from src.domain.shared.Agent_State import Turn

class ConversationHistoryService:
    def __init__(self, cache: CachePort, session_ttl: int = 3600, max_turns: int = 7) -> None:
        self._cache = cache
        self._session_ttl = session_ttl
        self._max_turns = max_turns

    async def get(self, session_id: str) -> list[Turn]:
        raw = await self._cache.get(f"session:{session_id}")
        return json.loads(raw) if raw else []

    async def append(self, session_id: str, role: Literal["assistant", "user"], content: str) -> None:
        history = await self.get(session_id)
        history.append({"role": role, "content": content})
        history = history[-self._max_turns * 2:]
        await self._cache.set(f"session:{session_id}", json.dumps(history), self._session_ttl)
