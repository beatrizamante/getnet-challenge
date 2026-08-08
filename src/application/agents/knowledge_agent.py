import logging

from src.domain.shared.state import AgentOutput, AgentState
from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.Search_Port import SearchPort

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a knowledgeable Getnet assistant. "
    "Use the provided context to answer questions about Getnet products and services accurately and concisely. "
    "If the context does not contain a clear answer, say so and advise the user to contact support."
)

class KnowledgeAgent:
    """RAG-powered agent that combines vector store retrieval with optional web search."""

    def __init__(
        self,
        llm: LLMPort,
        retrieval: RagRetrievalService,
        search: SearchPort,
    ) -> None:
        self._llm = llm
        self._retrieval = retrieval
        self._search = search

    async def run(self, state: AgentState) -> dict:
        user_message = state["messages"][-1] if state.get("messages") else ""

        chunks = await self._retrieval.retrieve_chunks(user_message)
        search_context = await self._web_search(user_message)

        rag_context = (
            "\n\n---\n\n".join(f"[Source: {c.source}]\n{c.content}" for c in chunks)
            if chunks
            else ""
        )
        context = "\n\n".join(filter(None, [rag_context, search_context]))
        prompt = f"Context:\n{context}\n\nQuestion: {user_message}" if context else user_message

        output = await self._llm.complete_structured(prompt, AgentOutput, system=_SYSTEM_PROMPT)
        response = {
            "answer": output.answer,
            "source_agent": "knowledge",
            "sources": [c.source for c in chunks],
        }
        return {"context": context, "response": response}

    async def _web_search(self, query: str) -> str:
        try:
            results = await self._search.search(query)
            if not results:
                return ""
            parts = [f"[{r.title}]({r.url})\n{r.snippet}" for r in results[:3]]
            return "\n\n".join(parts)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Web search failed, skipping. error=%s", exc)
            return ""
