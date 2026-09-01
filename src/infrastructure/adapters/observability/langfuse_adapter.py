import logging
from typing import Any

from langfuse import Langfuse  # noqa: F401 — imported so tests can patch it
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

from src.infrastructure.config.settings import LangfuseSettings

logger = logging.getLogger(__name__)


class LangfuseAdapter:
    """Thin wrapper over the Langfuse SDK v4; becomes a no-op when LANGFUSE_ENABLED=false."""

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
        try:
            client: Any = self._client
            safe = {
                k: (v[:500] + "\u2026" if isinstance(v, str) and len(v) > 500 else v)
                for k, v in payload.items()
            }
            trace_id: str = client.create_trace_id()
            client.start_observation(
                name=name, input=safe, trace_context=TraceContext(trace_id=trace_id)
            )
            return trace_id
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Langfuse trace failed. error=%s", exc)
            return ""

    def span(
        self, trace_id: str, name: str, input_data: dict[str, Any], output: dict[str, Any]
    ) -> None:
        """Record a child span on an existing trace."""
        if not self._enabled or self._client is None or not trace_id:
            return
        try:
            client: Any = self._client
            client.start_observation(
                name=name,
                input=input_data,
                output=output,
                trace_context=TraceContext(trace_id=trace_id),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Langfuse span failed. error=%s", exc)

    def score(self, trace_id: str, name: str, value: float) -> None:
        """Attach a numeric score to a trace."""
        if not self._enabled or self._client is None or not trace_id:
            return
        try:
            client: Any = self._client
            client.create_score(trace_id=trace_id, name=name, value=value)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Langfuse score failed. error=%s", exc)

    def get_callback_handler(
        self, user_id: str = "", session_id: str = "", trace_name: str = ""
    ) -> Any:
        """Return a LangChain CallbackHandler that automatically traces LLM+tool calls as child spans."""
        if not self._enabled or self._client is None:
            return None
        try:
            client: Any = self._client
            trace_id: str = client.create_trace_id()
            ctx: dict[str, Any] = {"trace_id": trace_id}
            if user_id:
                ctx["user_id"] = user_id
            if session_id:
                ctx["session_id"] = session_id
            return CallbackHandler(trace_context=TraceContext(**ctx))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Failed to create Langfuse callback handler. error=%s", exc)
            return None

    def flush(self) -> None:
        """Block until all pending events are sent; call on app shutdown."""
        if self._enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Langfuse flush failed. error=%s", exc)
