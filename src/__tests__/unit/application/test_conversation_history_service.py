import json
from unittest.mock import AsyncMock

import pytest

from src.application.caching.conversation_history_service import ConversationHistoryService
from src.domain.ports.Cache_Port import CachePort


def _make_cache() -> AsyncMock:
    return AsyncMock(spec=CachePort)


@pytest.mark.asyncio
async def test_get_returns_empty_history_when_session_is_missing() -> None:
    mock_cache = _make_cache()
    mock_cache.get.return_value = None
    service = ConversationHistoryService(mock_cache)

    history = await service.get("session-1")

    assert history == []
    mock_cache.get.assert_awaited_once_with("session:session-1")


@pytest.mark.asyncio
async def test_append_persists_turn_with_configured_ttl() -> None:
    mock_cache = _make_cache()
    mock_cache.get.return_value = json.dumps([{"role": "user", "content": "Oi"}])
    service = ConversationHistoryService(mock_cache, session_ttl=120, max_turns=2)

    await service.append("session-1", "assistant", "Como posso ajudar?")

    mock_cache.set.assert_awaited_once_with(
        "session:session-1",
        json.dumps([
            {"role": "user", "content": "Oi"},
            {"role": "assistant", "content": "Como posso ajudar?"},
        ]),
        120,
    )


@pytest.mark.asyncio
async def test_append_keeps_only_the_configured_number_of_conversation_turns() -> None:
    mock_cache = _make_cache()
    mock_cache.get.return_value = json.dumps([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ])
    service = ConversationHistoryService(mock_cache, max_turns=2)

    await service.append("session-1", "user", "five")

    stored_history = json.loads(mock_cache.set.await_args.args[1])
    assert stored_history == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"},
    ]
