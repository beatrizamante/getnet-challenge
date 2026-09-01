import json

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

    async def append_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Persist one full user→assistant turn in a single read-modify-write."""
        history = await self.get(session_id)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        history = history[-self._max_turns * 2 :]
        await self._cache.set(f"session:{session_id}", json.dumps(history), self._session_ttl)
