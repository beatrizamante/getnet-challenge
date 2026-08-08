import functools
import time
from collections.abc import Callable
from typing import Any

from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter


def traced_node(langfuse: LangfuseAdapter, node_name: str) -> Callable:
    """Wraps a LangGraph node function with a Langfuse trace+span, propagating user_id/session_id."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            user_id = str(state.get("user_id", ""))
            session_id = str(state.get("session_id", ""))
            messages = state.get("messages", [])

            trace_id = langfuse.trace(
                f"node.{node_name}",
                {"user_id": user_id, "session_id": session_id, "messages": messages},
            )
            start = time.monotonic()
            try:
                result = await fn(state)
                langfuse.span(
                    trace_id,
                    f"node.{node_name}",
                    input_data={"user_id": user_id, "session_id": session_id},
                    output={"latency_ms": int((time.monotonic() - start) * 1000)},
                )
                return result
            except Exception as exc:
                langfuse.span(
                    trace_id, f"node.{node_name}",
                    input_data={"user_id": user_id},
                    output={"error": str(exc)},
                )
                raise

        return wrapper
    return decorator
