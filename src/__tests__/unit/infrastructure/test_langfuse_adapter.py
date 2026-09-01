import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.observability.tracing import traced_node
from src.infrastructure.config.settings import LangfuseSettings


@pytest.fixture
def disabled_settings():
    return LangfuseSettings(enabled=False)


@pytest.fixture
def enabled_settings():
    return LangfuseSettings(
        enabled=True,
        public_key="pk-test",
        secret_key="sk-test",
        host="https://cloud.langfuse.com",
    )


def test_trace_is_noop_when_disabled(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    result = adapter.trace("name", {"key": "value"})
    assert result == ""


def test_trace_calls_langfuse_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    mock_client.create_trace_id.return_value = "trace-abc"

    with patch(
        "src.infrastructure.adapters.observability.langfuse_adapter.Langfuse",
        return_value=mock_client,
    ):
        adapter = LangfuseAdapter(enabled_settings)

    result = adapter.trace("llm.complete", {"prompt": "hi"})
    mock_client.create_trace_id.assert_called_once()
    mock_client.start_observation.assert_called_once()
    assert result == "trace-abc"


def test_span_calls_langfuse_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    with patch(
        "src.infrastructure.adapters.observability.langfuse_adapter.Langfuse",
        return_value=mock_client,
    ):
        adapter = LangfuseAdapter(enabled_settings)

    adapter.span("trace-id", "my-span", {"in": 1}, {"out": 2})
    mock_client.start_observation.assert_called_once()


def test_score_calls_langfuse_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    with patch(
        "src.infrastructure.adapters.observability.langfuse_adapter.Langfuse",
        return_value=mock_client,
    ):
        adapter = LangfuseAdapter(enabled_settings)

    adapter.score("trace-id", "relevance", 0.85)
    mock_client.create_score.assert_called_once_with(
        trace_id="trace-id", name="relevance", value=0.85
    )


def test_get_callback_handler_is_noop_when_disabled(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    assert adapter.get_callback_handler("u1", "s1") is None


def test_get_callback_handler_calls_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    mock_client.create_trace_id.return_value = "trace-xyz"

    with patch(
        "src.infrastructure.adapters.observability.langfuse_adapter.Langfuse",
        return_value=mock_client,
    ):
        adapter = LangfuseAdapter(enabled_settings)

    result = adapter.get_callback_handler(user_id="u1", session_id="s1", trace_name="agent.chat")
    mock_client.create_trace_id.assert_called_once()
    assert result is not None


def test_get_callback_handler_degrades_gracefully_on_error(enabled_settings):
    mock_client = MagicMock()
    mock_client.create_trace_id.side_effect = Exception("sdk error")

    with patch(
        "src.infrastructure.adapters.observability.langfuse_adapter.Langfuse",
        return_value=mock_client,
    ):
        adapter = LangfuseAdapter(enabled_settings)

    assert adapter.get_callback_handler("u1", "s1") is None


# --- traced_node ---


def test_traced_node_calls_trace_and_span(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    adapter.trace = MagicMock(return_value="tid")  # type: ignore[method-assign]
    adapter.span = MagicMock()  # type: ignore[method-assign]

    @traced_node(adapter, "router")
    async def fake_node(state):
        return {"route": "knowledge"}

    state = {"messages": ["hi"], "user_id": "u1", "session_id": "s1"}
    result = asyncio.run(fake_node(state))

    adapter.trace.assert_called_once()
    call_kwargs = adapter.trace.call_args
    assert "node.router" in call_kwargs[0]
    adapter.span.assert_called_once()
    assert result == {"route": "knowledge"}


def test_traced_node_records_error_span_on_exception(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    adapter.trace = MagicMock(return_value="tid")  # type: ignore[method-assign]
    adapter.span = MagicMock()  # type: ignore[method-assign]

    @traced_node(adapter, "router")
    async def failing_node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(failing_node({"messages": [], "user_id": "u1"}))

    span_output = (
        adapter.span.call_args[1]["output"]
        if adapter.span.call_args[1]
        else adapter.span.call_args[0][3]
    )
    assert "error" in span_output
