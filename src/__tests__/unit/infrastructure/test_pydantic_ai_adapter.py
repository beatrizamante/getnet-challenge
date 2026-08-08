import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

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
def adapter(mock_langfuse):
    instance = LLMAdapter.__new__(LLMAdapter)
    instance._model = TestModel()
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
        return MagicMock(output="ok")

    with patch("src.infrastructure.adapters.llm.pydantic_ai_adapter.asyncio.sleep"):
        result = await _with_retry(flaky)
    assert result.output == "ok"
    assert calls == 3
