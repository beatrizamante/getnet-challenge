import logging
from typing import Any

from langfuse import Langfuse  # noqa: F401 — imported so tests can patch it

from src.infrastructure.config.settings import LangfuseSettings

logger = logging.getLogger(__name__)


class LangfuseAdapter:
    """Thin wrapper over the Langfuse SDK; becomes a no-op when LANGFUSE_ENABLED=false."""

    def __init__(self, settings: LangfuseSettings) -> None:
        self._enabled = settings.enabled
        self._client: Any = None
        if self._enabled:
            try:
                self._client = Langfuse(
                    public_key=settings.public_key,
                    secret_key=settings.secret_key,
                    host=settings.host,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Langfuse client init failed, disabling tracing. error=%s", exc)
                self._enabled = False

    def trace(self, name: str, payload: dict[str, Any]) -> str:
        """Open a new trace and return its ID, or '' when tracing is disabled."""
        if not self._enabled or self._client is None:
            return ""
        client: Any = self._client
        t = client.trace(name=name, input=payload)  # pylint: disable=no-member
        return t.id

    def span(self, trace_id: str, name: str, input_data: dict[str, Any], output: dict[str, Any]) -> None:
        """Record a child span on an existing trace."""
        if not self._enabled or self._client is None or not trace_id:
            return
        client: Any = self._client
        client.span(trace_id=trace_id, name=name, input=input_data, output=output)  # pylint: disable=no-member

    def score(self, trace_id: str, name: str, value: float) -> None:
        """Attach a numeric score to a trace (e.g. faithfulness, relevance)."""
        if not self._enabled or self._client is None or not trace_id:
            return
        client: Any = self._client
        client.score(trace_id=trace_id, name=name, value=value)  # pylint: disable=no-member

    def flush(self) -> None:
        """Block until all pending events are sent; call on app shutdown."""
        if self._enabled and self._client is not None:
            client: Any = self._client
            client.flush()
