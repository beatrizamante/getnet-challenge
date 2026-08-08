# pylint: disable=redefined-outer-name
from unittest.mock import AsyncMock

from src.application.agents.router_agent import RouterAgent
from src.application.agents.knowledge_agent import KnowledgeAgent
from src.application.agents.customer_support_agent import CustomerSupportAgent
from src.domain.shared.state import AgentOutput
from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.entities.Chunk import Chunk
from src.domain.entities.Route_Decision import RouteDecisionModel
from src.domain.entities.User_Profile import UserProfile
from src.domain.ports.LLM_Port import LLMPort
from src.domain.ports.Search_Port import SearchPort
from src.domain.ports.User_Repository_Port import UserRepositoryPort


class TestRouterAgent:
    async def test_run_returns_route_from_llm(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = RouteDecisionModel(
            intent="product question",
            confidence=0.9,
            target_agent="knowledge",
        )
        agent = RouterAgent(llm=llm)
        state = await agent.run({"messages": ["What are Getnet's plans?"], "user_id": "u1"})

        assert state == {"route": "knowledge"}

    async def test_run_handles_empty_messages(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = RouteDecisionModel(
            intent="unknown", confidence=0.5, target_agent="knowledge"
        )
        agent = RouterAgent(llm=llm)
        state = await agent.run({"messages": [], "user_id": "u1"})

        assert "route" in state


class TestKnowledgeAgent:
    async def test_run_builds_response_with_rag_and_search(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = AgentOutput(answer="Here is the info.")

        retrieval = AsyncMock(spec=RagRetrievalService)
        retrieval.retrieve_chunks.return_value = [
            Chunk(id="1", content="Getnet accepts cards.", source="getnet.com.br")
        ]

        search = AsyncMock(spec=SearchPort)
        search.search.return_value = []

        agent = KnowledgeAgent(llm=llm, retrieval=retrieval, search=search)
        result = await agent.run({"messages": ["Does Getnet accept Visa?"], "user_id": "u1"})

        assert result["response"]["source_agent"] == "knowledge"
        assert result["response"]["answer"] == "Here is the info."
        assert "confidence" not in result["response"]
        assert "getnet.com.br" in result["response"]["sources"]

    async def test_run_continues_if_search_fails(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = AgentOutput(answer="answer")

        retrieval = AsyncMock(spec=RagRetrievalService)
        retrieval.retrieve_chunks.return_value = []

        search = AsyncMock(spec=SearchPort)
        search.search.side_effect = Exception("Tavily down")

        agent = KnowledgeAgent(llm=llm, retrieval=retrieval, search=search)
        result = await agent.run({"messages": ["question"], "user_id": "u1"})

        assert result["response"]["answer"] == "answer"


class TestCustomerSupportAgent:
    async def test_run_includes_profile_in_prompt(self):
        from datetime import datetime, timezone

        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = AgentOutput(answer="Your plan is Basic.")

        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_profile.return_value = UserProfile(
            plan="Basic",
            machine_model="Mini",
            status="active",
            joined_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        user_repo.get_transactions.return_value = []

        agent = CustomerSupportAgent(llm=llm, user_repo=user_repo)
        result = await agent.run({"messages": ["What's my plan?"], "user_id": "u42"})

        assert result["response"]["source_agent"] == "customer_support"
        assert "confidence" not in result["response"]
        prompt_used = llm.complete_structured.call_args[0][0]
        assert "Basic" in prompt_used

    async def test_run_without_profile_sends_raw_message(self):
        llm = AsyncMock(spec=LLMPort)
        llm.complete_structured.return_value = AgentOutput(answer="No info found.")

        user_repo = AsyncMock(spec=UserRepositoryPort)
        user_repo.get_profile.return_value = None
        user_repo.get_transactions.return_value = []

        agent = CustomerSupportAgent(llm=llm, user_repo=user_repo)
        await agent.run({"messages": ["Help me"], "user_id": "u99"})

        prompt_used = llm.complete_structured.call_args[0][0]
        assert prompt_used == "Help me"
