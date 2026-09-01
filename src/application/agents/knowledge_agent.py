import json
import logging
import re
from contextvars import ContextVar
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool

from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.entities.Agent_Response import AgentResponse
from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.Search_Port import SearchPort
from src.domain.shared.Agent_State import AgentState, Turn
from src.domain.shared.constants import CONTEXT_CHUNK_SEPARATOR, REACT_MAX_ITERATIONS
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.config.prompt_catalog import PromptCatalog, load_prompt_catalog

logger = logging.getLogger(__name__)

_retrieved_ctx: ContextVar[list[str]] = ContextVar("retrieved_context")


class KnowledgeAgent:
    """LangGraph ReAct agent — graph compiled once; per-request context via ContextVar."""

    def __init__(
        self,
        llm: LLMPort,
        retrieval: RagRetrievalService,
        search: SearchPort,
        cache: CachePort,
        langfuse: LangfuseAdapter | None = None,
        kb_cache_ttl: int = 1800,
        prompts: PromptCatalog | None = None,
        _graph: Any = None,
    ) -> None:
        self._langfuse = langfuse
        prompt_catalog = prompts or load_prompt_catalog()
        tools = [
            _make_retrieve_tool(retrieval, cache, kb_cache_ttl),
            _make_web_search_tool(search),
        ]
        self._graph = _graph or create_agent(
            llm.as_runnable(),
            tools=tools,
            system_prompt=prompt_catalog.knowledge_system,
        )

    def _build_messages(self, history: list[Turn], current: str) -> list:
        msgs = []
        for turn in history:
            cls = HumanMessage if turn["role"] == "user" else AIMessage
            msgs.append(cls(content=turn["content"]))
        msgs.append(HumanMessage(content=current))
        return msgs

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""
        user_id = str(state.get("user_id", ""))
        session_id = str(state.get("session_id", ""))

        token = _retrieved_ctx.set([])
        try:
            callbacks = []
            if self._langfuse:
                handler = self._langfuse.get_callback_handler(
                    user_id=user_id, session_id=session_id, trace_name="knowledge_agent"
                )
                if handler:
                    callbacks.append(handler)

            config: RunnableConfig = {
                "callbacks": callbacks,
                "recursion_limit": REACT_MAX_ITERATIONS,
            }
            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=user_message)]}, config=config
            )

            final: AIMessage = result["messages"][-1]
            answer = final.content if isinstance(final.content, str) else str(final.content)
            sources = _extract_sources(result["messages"])
            context = CONTEXT_CHUNK_SEPARATOR.join(_retrieved_ctx.get())
        finally:
            _retrieved_ctx.reset(token)

        return {
            "context": context,
            "response": AgentResponse.build(
                answer=answer,
                source_agent="knowledge",
                sources=sources,
            ).model_dump(),
        }


def _make_retrieve_tool(
    retrieval: RagRetrievalService,
    cache: CachePort,
    kb_cache_ttl: int = 1800,
):
    @lc_tool
    async def retrieve_from_kb(query: str) -> str:
        """Search Getnet's knowledge base for product and service information."""
        cache_key = f"kb:{query.lower().strip()}"
        cached = await cache.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                if "context" in data:
                    logger.debug("KB retrieval cache hit.")
                    _retrieved_ctx.get().append(data["context"])
                    return data["context"]
            except (json.JSONDecodeError, KeyError):
                pass

        chunks = await retrieval.retrieve_chunks(query)
        if not chunks:
            return ""

        context = CONTEXT_CHUNK_SEPARATOR.join(f"[Source: {c.source}]\n{c.content}" for c in chunks)
        sources = list(dict.fromkeys(c.source for c in chunks))
        await cache.set(
            cache_key, json.dumps({"context": context, "sources": sources}), kb_cache_ttl
        )
        _retrieved_ctx.get().append(context)
        return context

    return retrieve_from_kb


def _make_web_search_tool(search: SearchPort):
    @lc_tool
    async def web_search(query: str) -> str:
        """Search the web for questions not covered by the Getnet knowledge base."""
        try:
            results = await search.search(query)
            if not results:
                return "No results found."
            return "\n\n".join(
                f"[{r.title}]({r.url})\n{_sanitize_snippet(r.snippet)}" for r in results[:3]
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Web search failed. error=%s", exc)
            return ""

    return web_search


def _sanitize_snippet(text: str, max_chars: int = 400) -> str:
    """Strip HTML tags and cap length — reduces indirect prompt injection surface."""
    clean = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", clean).strip()[:max_chars]


def _extract_sources(messages: list) -> list[str]:
    """Parse [Source: url] labels out of tool response messages."""
    sources: list[str] = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str):
            for line in content.splitlines():
                if line.startswith("[Source: ") and line.endswith("]"):
                    sources.append(line[9:-1])
    return list(dict.fromkeys(sources))
