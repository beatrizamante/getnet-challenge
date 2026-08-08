from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
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


def test_span_is_noop_when_disabled(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    adapter.span("trace-id", "name", {}, {})  # must not raise


def test_score_is_noop_when_disabled(disabled_settings):
    adapter = LangfuseAdapter(disabled_settings)
    adapter.score("trace-id", "faithfulness", 0.9)  # must not raise


def test_trace_calls_langfuse_client_when_enabled(enabled_settings):
    mock_trace = MagicMock()
    mock_trace.id = "trace-abc"
    mock_client = MagicMock()
    mock_client.trace.return_value = mock_trace

    with patch("src.infrastructure.adapters.observability.langfuse_adapter.Langfuse", return_value=mock_client):
        adapter = LangfuseAdapter(enabled_settings)

    result = adapter.trace("llm.complete", {"prompt": "hi"})
    mock_client.trace.assert_called_once_with(name="llm.complete", input={"prompt": "hi"})
    assert result == "trace-abc"


def test_span_calls_langfuse_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    with patch("src.infrastructure.adapters.observability.langfuse_adapter.Langfuse", return_value=mock_client):
        adapter = LangfuseAdapter(enabled_settings)

    adapter.span("trace-id", "my-span", {"in": 1}, {"out": 2})
    mock_client.span.assert_called_once_with(
        trace_id="trace-id", name="my-span", input={"in": 1}, output={"out": 2}
    )


def test_score_calls_langfuse_client_when_enabled(enabled_settings):
    mock_client = MagicMock()
    with patch("src.infrastructure.adapters.observability.langfuse_adapter.Langfuse", return_value=mock_client):
        adapter = LangfuseAdapter(enabled_settings)

    adapter.score("trace-id", "relevance", 0.85)
    mock_client.score.assert_called_once_with(trace_id="trace-id", name="relevance", value=0.85)


def test_init_degrades_gracefully_when_langfuse_raises(enabled_settings):
    with patch("src.infrastructure.adapters.observability.langfuse_adapter.Langfuse", side_effect=Exception("unreachable")):
        adapter = LangfuseAdapter(enabled_settings)
    # should be disabled and all methods no-op
    assert adapter.trace("n", {}) == ""
