# pylint: disable=redefined-outer-name,protected-access
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from src.infrastructure.adapters.llm.llm_adapter import LLMAdapter


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
    mock_msg.content = "Test response from DeepSeek"
    model.ainvoke = AsyncMock(return_value=mock_msg)
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=_Answer(text="ok", confidence=0.9))
    model.with_structured_output = MagicMock(return_value=structured)
    return model


@pytest.fixture
def adapter(mock_langfuse, mock_chat_model):
    instance = LLMAdapter.__new__(LLMAdapter)
    instance._model = mock_chat_model
    instance._langfuse = mock_langfuse
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
    mock_langfuse.trace.assert_called_once_with(
        "llm.complete_structured", {"prompt": "prompt", "schema": "_Answer"}
    )
    mock_langfuse.span.assert_called_once()


async def test_complete_structured_calls_with_structured_output(adapter, mock_chat_model):
    await adapter.complete_structured("prompt", _Answer)
    mock_chat_model.with_structured_output.assert_called_once_with(_Answer)


async def test_complete_retries_on_rate_limit(mock_langfuse):
    import httpx
    from src.infrastructure.adapters.llm.llm_adapter import _with_retry

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
