import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool as lc_tool
from langchain.agents import create_agent

from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.Search_Port import SearchPort
from src.domain.shared.Agent_State import AgentState
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Getnet product specialist. "
    "You have two tools available:\n"
    "  - retrieve_from_kb: search Getnet's knowledge base for product/service info.\n"
    "  - web_search: search the web for general questions not in the knowledge base.\n"
    "Always try retrieve_from_kb first. Only use web_search if the knowledge base "
    "returns no useful information. "
    "Always cite sources using the [Source: ...] labels. "
    "If no source has the answer, say so honestly — never invent facts about Getnet."
)

_KB_CACHE_TTL = 1800  # 30 min


class KnowledgeAgent:
    """LangGraph ReAct agent — the LLM decides which tool to call and when to stop."""

    def __init__(
        self,
        llm: LLMPort,
        retrieval: RagRetrievalService,
        search: SearchPort,
        cache: CachePort,
        langfuse: LangfuseAdapter | None = None,
        _graph: Any = None,
    ) -> None:
        self._llm = llm
        self._langfuse = langfuse
        self._retrieval = retrieval
        self._search = search
        self._cache = cache
        self._graph_override = _graph

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""
        user_id = str(state.get("user_id", ""))
        session_id = str(state.get("session_id", ""))

        retrieved_context: list[str] = []
        tools = [
            _make_retrieve_tool(self._retrieval, self._cache, retrieved_context),
            _make_web_search_tool(self._search),
        ]
        graph = self._graph_override or create_agent(
            self._llm.as_runnable(),
            tools=tools,
            system_prompt=_SYSTEM_PROMPT,
        )

        callbacks = []
        if self._langfuse:
            handler = self._langfuse.get_callback_handler(
                user_id=user_id, session_id=session_id, trace_name="knowledge_agent"
            )
            if handler:
                callbacks.append(handler)

        config: RunnableConfig = {"callbacks": callbacks} if callbacks else {}
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_message)]}, config=config
        )

        final: AIMessage = result["messages"][-1]
        answer = final.content if isinstance(final.content, str) else str(final.content)
        sources = _extract_sources(result["messages"])
        context = "\n\n---\n\n".join(retrieved_context)

        return {
            "context": context,
            "response": {"answer": answer, "source_agent": "knowledge", "sources": sources},
        }


def _make_retrieve_tool(retrieval: RagRetrievalService, cache: CachePort, context_sink: list[str]):
    @lc_tool
    async def retrieve_from_kb(query: str) -> str:
        """Search Getnet's knowledge base for product and service information."""
        cache_key = f"kb:{query}"
        cached = await cache.get(cache_key)
        if cached:
            try:
                data = json.loads(cached)
                if "context" in data:
                    logger.debug("KB retrieval cache hit.")
                    context_sink.append(data["context"])
                    return data["context"]
            except (json.JSONDecodeError, KeyError):
                pass

        chunks = await retrieval.retrieve_chunks(query)
        if not chunks:
            return ""

        context = "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.content}" for c in chunks)
        sources = list(dict.fromkeys(c.source for c in chunks))
        await cache.set(cache_key, json.dumps({"context": context, "sources": sources}), _KB_CACHE_TTL)
        context_sink.append(context)
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
    import re
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
