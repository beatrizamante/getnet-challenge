# pylint: disable=redefined-outer-name
from langchain_core.messages import AIMessage, ToolMessage
from unittest.mock import AsyncMock
import json
from datetime import datetime, timezone


from src.application.agents.router_agent import RouterAgent
from src.application.agents.knowledge_agent import KnowledgeAgent, _make_retrieve_tool, _retrieved_ctx
from src.application.agents.customer_support_agent import (
    CustomerSupportAgent,
    _make_get_profile_tool,
    _make_get_settlement_tool,
    _make_get_transactions_tool,
)
from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.entities.User_Profile import UserProfile
from src.domain.entities.Chunk import Chunk
from src.domain.entities.Route_Decision import RouteDecisionModel
from src.domain.ports.Cache_Port import CachePort
from src.domain.ports.LLM_Port import LLMPort
from src.domain.entities.Transaction import Transaction
from src.domain.ports.Search_Port import SearchPort
from src.domain.shared.Application_Errors import UserNotFoundError
from src.domain.ports.User_Repository_Port import UserRepositoryPort


def _decision(intent, target_agent):
    return RouteDecisionModel(intent=intent, target_agent=target_agent, reasoning="test")


class TestRouterAgent:
    # --- routing logic ---

    async def test_routes_knowledge_intent(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("knowledge", "knowledge")
        result = await RouterAgent(llm=llm).run({"messages": ["What is Get Smart?"], "user_id": "u1"})
        assert result == {"route": "knowledge"}

    async def test_routes_customer_support_intent(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("customer_support", "customer_support")
        result = await RouterAgent(llm=llm).run({"messages": ["My deposit is late"], "user_id": "u1"})
        assert result == {"route": "customer_support"}

    async def test_routes_general_search_intent(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("general_search", "general_search")
        result = await RouterAgent(llm=llm).run({"messages": ["Euro rate today"], "user_id": "u1"})
        assert result == {"route": "general_search"}

    async def test_routes_off_topic_intent(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("off_topic", "off_topic")
        result = await RouterAgent(llm=llm).run({"messages": ["Weather in SP?"], "user_id": "u1"})
        assert result == {"route": "off_topic"}

    async def test_routes_escalate_when_llm_classifies_as_escalate(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("escalate", "escalate")
        result = await RouterAgent(llm=llm).run({"messages": ["uhh"], "user_id": "u1"})
        assert result == {"route": "escalate"}

    async def test_handles_empty_messages(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision("knowledge", "knowledge")
        state = await RouterAgent(llm=llm).run({"messages": [], "user_id": "u1"})
        assert "route" in state

    # --- 10 challenge scenarios (mocked LLM returns expected classification) ---

    async def _route(self, message: str, intent: str, target: str) -> str:
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = _decision(intent, target)
        result = await RouterAgent(llm=llm).run({"messages": [message], "user_id": "cliente1988"})
        return result["route"]

    async def test_scenario_get_classica_vs_smart(self):
        route = await self._route(
            "What's the difference between the Get Clássica and the Get Smart?",
            "knowledge", "knowledge"
        )
        assert route == "knowledge"

    async def test_scenario_weather(self):
        route = await self._route(
            "What's the weather forecast in Porto Alegre tomorrow?",
            "off_topic", "off_topic"
        )
        assert route == "off_topic"

    async def test_scenario_deposit_timing(self):
        route = await self._route(
            "When will the money from yesterday's sales be deposited?",
            "customer_support", "customer_support"
        )
        assert route == "customer_support"

    async def test_scenario_pix_bank_account(self):
        route = await self._route(
            "Do I need a bank account to receive my sales via Pix?",
            "knowledge", "knowledge"
        )
        assert route == "knowledge"

    async def test_scenario_machine_no_internet(self):
        route = await self._route(
            "My card machine won't connect to the internet, what should I do?",
            "customer_support", "customer_support"
        )
        assert route == "customer_support"

    async def test_scenario_antecipacao(self):
        route = await self._route(
            "How does receivables advance (antecipação) work with Getnet?",
            "knowledge", "knowledge"
        )
        assert route == "knowledge"

    async def test_scenario_euro_rate(self):
        route = await self._route(
            "What's the euro exchange rate today?",
            "general_search", "general_search"
        )
        assert route == "general_search"

    async def test_scenario_decline_error(self):
        route = await self._route(
            "My card machine is showing a transaction decline error.",
            "customer_support", "customer_support"
        )
        assert route == "customer_support"

    async def test_scenario_crediario_installments(self):
        route = await self._route(
            "How many installments can I split a sale into with the crediário?",
            "knowledge", "knowledge"
        )
        assert route == "knowledge"

    async def test_scenario_whatsapp_payment_link(self):
        route = await self._route(
            "Can I sell through WhatsApp using the Payment Link?",
            "knowledge", "knowledge"
        )
        assert route == "knowledge"


class TestKnowledgeAgent:
    """Tests inject a mock graph via _graph= to bypass create_agent entirely."""

    def _make_mock_graph(self, answer: str = "answer", sources: list[str] | None = None):
        mock_graph = AsyncMock()
        messages = [AIMessage(content=answer)]
        if sources:
            # simulate a tool message with [Source: url] lines so _extract_sources works
            tool_msg = ToolMessage(content="\n".join(f"[Source: {s}]" for s in sources), tool_call_id="t1")
            messages = [tool_msg] + messages
        mock_graph.ainvoke.return_value = {"messages": messages}
        return mock_graph

    def _make_agent(self, graph):
        llm = AsyncMock(spec=LLMPort)
        retrieval = AsyncMock(spec=RagRetrievalService)
        search = AsyncMock(spec=SearchPort)
        cache = AsyncMock(spec=CachePort, **{"get.return_value": None}) # type: ignore
        return KnowledgeAgent(llm=llm, retrieval=retrieval, search=search, cache=cache, _graph=graph)

    async def test_run_returns_answer_from_graph(self):
        graph = self._make_mock_graph(answer="Getnet accepts Visa.", sources=["getnet.com.br"])
        agent = self._make_agent(graph)
        result = await agent.run({"messages": ["Does Getnet accept Visa?"], "user_id": "u1"})

        assert result["response"]["source_agent"] == "knowledge"
        assert result["response"]["answer"] == "Getnet accepts Visa."
        assert "getnet.com.br" in result["response"]["sources"]
        graph.ainvoke.assert_called_once()

    async def test_run_empty_sources_when_no_tool_messages(self):
        graph = self._make_mock_graph(answer="I don't know.")
        agent = self._make_agent(graph)
        result = await agent.run({"messages": ["question"], "user_id": "u1"})

        assert result["response"]["sources"] == []

    async def test_run_handles_empty_messages(self):
        graph = self._make_mock_graph(answer="answer")
        agent = self._make_agent(graph)
        result = await agent.run({"messages": [], "user_id": "u1"})

        assert "answer" in result["response"]

    async def test_retrieve_tool_caches_result(self):
        """Tests the retrieve_from_kb tool function independently."""
        retrieval = AsyncMock(spec=RagRetrievalService)
        retrieval.retrieve_chunks.return_value = [
            Chunk(id="1", content="info", source="getnet.com.br")
        ]
        cache = AsyncMock(spec=CachePort)
        cache.get.return_value = None

        tool = _make_retrieve_tool(retrieval, cache)
        token = _retrieved_ctx.set([])
        try:
            result = await tool.ainvoke({"query": "payment methods"})
        finally:
            _retrieved_ctx.reset(token)

        assert "info" in result
        cache.set.assert_called_once()
        stored = json.loads(cache.set.call_args[0][1])
        assert "getnet.com.br" in stored["sources"]

    async def test_retrieve_tool_returns_cached_without_chromadb(self):
        retrieval = AsyncMock(spec=RagRetrievalService)
        cache = AsyncMock(spec=CachePort)
        cache.get.return_value = json.dumps({"context": "cached ctx", "sources": []})

        tool = _make_retrieve_tool(retrieval, cache)
        token = _retrieved_ctx.set([])
        try:
            result = await tool.ainvoke({"query": "q"})
        finally:
            _retrieved_ctx.reset(token)

        assert result == "cached ctx"
        retrieval.retrieve_chunks.assert_not_called()


class TestCustomerSupportAgent:
    def _make_mock_graph(self, answer: str = "answer"):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"messages": [AIMessage(content=answer)]}
        return mock_graph

    def _make_agent(self, graph, user_repo=None):
        llm = AsyncMock(spec=LLMPort)
        user_repo = user_repo or AsyncMock(spec=UserRepositoryPort)
        return CustomerSupportAgent(llm=llm, user_repo=user_repo, _graph=graph)

    async def test_run_returns_answer_from_graph(self):
        graph = self._make_mock_graph("Your plan is Get Smart.")
        agent = self._make_agent(graph)
        result = await agent.run({"messages": ["What is my plan?"], "user_id": "u1"})

        assert result["response"]["source_agent"] == "customer_support"
        assert result["response"]["answer"] == "Your plan is Get Smart."
        graph.ainvoke.assert_called_once()

    async def test_run_injects_user_id_into_system_message(self):
        graph = self._make_mock_graph("ok")
        agent = self._make_agent(graph)
        await agent.run({"messages": ["help"], "user_id": "cliente1988"})

        messages_sent = graph.ainvoke.call_args[0][0]["messages"]
        system_content = messages_sent[0].content
        assert "cliente1988" in system_content

    async def test_get_profile_tool_returns_formatted_profile(self):
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_profile.return_value = UserProfile(
            plan="Get Smart", machine_model="Smart 2",
            status="active", joined_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        tool = _make_get_profile_tool(user_repo)
        result = await tool.ainvoke({"user_id": "u1"})

        assert "Get Smart" in result
        assert "Smart 2" in result
        assert "active" in result

    async def test_get_profile_tool_handles_unknown_user(self):
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_profile.side_effect = UserNotFoundError("ghost")
        tool = _make_get_profile_tool(user_repo)
        result = await tool.ainvoke({"user_id": "ghost"})

        assert "No account found" in result

    async def test_get_transactions_tool_formats_list(self):
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_transactions.return_value = [
            Transaction(
                id="TX-001", amount=15000, status="settled",
                created_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
                settlement_date=datetime(2026, 8, 8, tzinfo=timezone.utc),
            )
        ]
        tool = _make_get_transactions_tool(user_repo)
        result = await tool.ainvoke({"user_id": "u1", "days": 7})

        assert "TX-001" in result
        assert "R$150.00" in result
        assert "settled" in result

    async def test_get_transactions_tool_empty(self):
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_transactions.return_value = []
        tool = _make_get_transactions_tool(user_repo)
        result = await tool.ainvoke({"user_id": "u1", "days": 7})

        assert "No transactions" in result

    async def test_get_settlement_tool_returns_date(self):
        tx = Transaction(
            id="TX-001", amount=50000, status="settled",
            created_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
            settlement_date=datetime(2026, 8, 7, tzinfo=timezone.utc),
        )
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_transactions.return_value = [tx]
        tool = _make_get_settlement_tool(user_repo)
        result = await tool.ainvoke({"transaction_id": "TX-001", "user_id": "u1"})

        assert "2026-08-07" in result
        assert "R$500.00" in result

    async def test_get_settlement_tool_unknown_transaction(self):
        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_transactions.return_value = []
        tool = _make_get_settlement_tool(user_repo)
        result = await tool.ainvoke({"transaction_id": "TX-999", "user_id": "u1"})

        assert "not found" in result
