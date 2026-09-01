# pylint: disable=redefined-outer-name,protected-access
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from src.infrastructure.adapters.llm.llm_adapter import LLMAdapter, _with_retry


class _Answer(BaseModel):
    text: str
    confidence: float


@pytest.fixture
def mock_langfuse():
    lf = MagicMock()
    lf.trace.return_value = "trace-123"
    return lf


@pytest.fixture
def mock_chat_model():
    model = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = '{"text": "ok", "confidence": 0.9}'
    model.ainvoke = AsyncMock(return_value=mock_msg)
    return model


@pytest.fixture
def adapter(mock_langfuse, mock_chat_model):
    instance = LLMAdapter.__new__(LLMAdapter)
    instance._model = mock_chat_model
    instance._langfuse = mock_langfuse
    instance._max_retries = 3
    instance._base_delay = 1.0
    return instance


async def test_complete_returns_string(adapter):
    result = await adapter.complete("What is Getnet?", "You are a helpful assistant.")
    assert isinstance(result, str)


async def test_complete_traces_in_langfuse(adapter, mock_langfuse):
    await adapter.complete("prompt", "system")
    mock_langfuse.trace.assert_called_once_with(
        "llm.complete", {"prompt": "prompt", "system": "system"}
    )
    mock_langfuse.span.assert_called_once()


async def test_complete_structured_returns_schema_instance(adapter):
    result = await adapter.complete_structured("Return an answer.", _Answer)
    assert isinstance(result, _Answer)


async def test_complete_structured_traces_in_langfuse(adapter, mock_langfuse):
    await adapter.complete_structured("prompt", _Answer)
    # complete_structured calls complete() internally — both emit a trace
    calls = [c.args[0] for c in mock_langfuse.trace.call_args_list]
    assert "llm.complete_structured" in calls
    assert "llm.complete" in calls


async def test_complete_structured_calls_with_structured_output(adapter, mock_chat_model):
    result = await adapter.complete_structured("prompt", _Answer)
    assert isinstance(result, _Answer)


async def test_complete_retries_on_rate_limit(mock_langfuse):

    calls = 0

    async def flaky(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            response = MagicMock()
            response.status_code = 429
            raise httpx.HTTPStatusError("rate limited", request=MagicMock(), response=response)
        return MagicMock(content="ok")

    with patch("src.infrastructure.adapters.llm.llm_adapter.asyncio.sleep"):
        result = await _with_retry(flaky)
    assert result.content == "ok"
    assert calls == 3
