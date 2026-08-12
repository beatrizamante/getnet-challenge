
from typing import Any
import chromadb
import redis.asyncio as aioredis
from dependency_injector import containers, providers

from src.application.agents.customer_support_agent import CustomerSupportAgent
from src.application.agents.escalation_agent import EscalationAgent
from src.application.agents.graph import build_graph
from src.application.agents.knowledge_agent import KnowledgeAgent
from src.application.agents.router_agent import RouterAgent
from src.application.caching.semantic_cache_service import SemanticCacheService
from src.application.guardrails.input_guardrail import InputGuardrail
from src.application.guardrails.output_guardrail import OutputGuardrail
from src.application.rag_pipeline.ingest_service import RagIngestService
from src.application.rag_pipeline.retrieval_service import RagRetrievalService
from src.domain.ports.Reranker_Port import RerankerPort
from src.config.logger import setup_logging
from src.infrastructure.adapters.cache.redis_cache_adapter import RedisCacheAdapter
from src.infrastructure.adapters.embeddings.huggingface_adapter import HuggingFaceEmbeddingAdapter
from src.infrastructure.adapters.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.adapters.llm.llm_adapter import LLMAdapter
from src.infrastructure.adapters.llm.deepseek_judge import DeepSeekJudgeModel
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.queue.arq_adapter import ArqQueueAdapter
from src.infrastructure.adapters.search.tavily_adapter import TavilySearchAdapter
from src.infrastructure.adapters.user_repository.mock_user_repository import MockUserRepository
from src.infrastructure.adapters.vector_store.chroma_adapter import ChromaAdapter
from src.infrastructure.config.settings import Settings, get_settings
from src.infrastructure.adapters.scraper.getnet_scraper import GetnetScraper


def _make_langfuse(settings: Settings) -> LangfuseAdapter:
    return LangfuseAdapter(settings.langfuse)


def _make_embedding(settings: Settings) -> HuggingFaceEmbeddingAdapter:
    return HuggingFaceEmbeddingAdapter(settings.embedding)


def _make_llm(settings: Settings, langfuse: LangfuseAdapter) -> LLMAdapter:
    return LLMAdapter(settings.llm, langfuse)


def _make_search(settings: Settings) -> TavilySearchAdapter:
    return TavilySearchAdapter(settings.search)


def _make_queue(settings: Settings) -> ArqQueueAdapter:
    return ArqQueueAdapter(settings.redis)


def _make_redis_client(settings: Settings) -> aioredis.Redis:
    return aioredis.Redis.from_url(settings.redis.url)


def _make_cache(
    settings: Settings,
    redis_client: aioredis.Redis,
    embedding: HuggingFaceEmbeddingAdapter,
) -> RedisCacheAdapter:
    return RedisCacheAdapter(
        client=redis_client,
        embedding_port=embedding,
        similarity_threshold=settings.redis.cache_similarity_threshold,
        default_ttl=settings.redis.ttl,
    )


async def _chroma_client_resource(settings: Settings):
    """Lazy-connect async resource: yields None on startup failure so the app boots without ChromaDB."""
    import logging
    _logger = logging.getLogger(__name__)
    try:
        client = await chromadb.AsyncHttpClient(
            host=settings.chroma.host, port=settings.chroma.port
        )
        _logger.info("ChromaDB connected. host=%s port=%d", settings.chroma.host, settings.chroma.port)
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning("ChromaDB unavailable at startup — vector store disabled. error=%s", exc)
        client = None
    yield client


def _make_vector_store(
    settings: Settings,
    chroma_client: Any,
    embedding: HuggingFaceEmbeddingAdapter,
) -> ChromaAdapter:
    return ChromaAdapter(
        client=chroma_client,
        embedding_port=embedding,
        collection_name=settings.chroma.collection_name,
    )


def _make_ingest_service(settings: Settings, vector_store: ChromaAdapter) -> RagIngestService:
    return RagIngestService(
        vector_store=vector_store,
        chunk_size=settings.ingestion.chunk_size,
        chunk_overlap=settings.ingestion.chunk_overlap,
    )


def _make_retrieval_service(settings: Settings, vector_store: ChromaAdapter) -> RagRetrievalService:
    reranker: RerankerPort | None = None
    if settings.reranker.enabled:
        reranker = CrossEncoderReranker(model_name=settings.reranker.model)
    return RagRetrievalService(
        vector_store=vector_store,
        reranker=reranker,
        top_n=settings.reranker.top_n,
        rerank_factor=settings.reranker.factor,
    )


def _make_semantic_cache(cache: RedisCacheAdapter, llm: LLMAdapter) -> SemanticCacheService:
    return SemanticCacheService(cache=cache, llm=llm)


def _make_escalation_agent(
    langfuse: LangfuseAdapter, redis_client: aioredis.Redis
) -> EscalationAgent:
    return EscalationAgent(langfuse=langfuse, redis_client=redis_client)


def _make_input_guardrail(llm: LLMAdapter) -> InputGuardrail:
    return InputGuardrail(llm=llm)


def _make_output_guardrail(settings: Settings, langfuse: LangfuseAdapter) -> OutputGuardrail:
    judge = DeepSeekJudgeModel(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model_name=settings.guardrail.model,
    )
    return OutputGuardrail(judge=judge, langfuse=langfuse, threshold=settings.guardrail.faithfulness_threshold)


def _make_user_repo() -> MockUserRepository:
    return MockUserRepository()


def _make_scraper(settings: Settings) -> GetnetScraper:
    return GetnetScraper(max_concurrent=settings.ingestion.max_concurrent)


def _make_router_agent(llm: LLMAdapter) -> RouterAgent:
    return RouterAgent(llm=llm)


def _make_knowledge_agent(
    llm: LLMAdapter,
    retrieval: RagRetrievalService,
    search: TavilySearchAdapter,
    cache: RedisCacheAdapter,
    langfuse: LangfuseAdapter,
) -> KnowledgeAgent:
    return KnowledgeAgent(llm=llm, retrieval=retrieval, search=search, cache=cache, langfuse=langfuse)


def _make_customer_support_agent(
    llm: LLMAdapter,
    user_repo: MockUserRepository,
    langfuse: LangfuseAdapter,
) -> CustomerSupportAgent:
    return CustomerSupportAgent(llm=llm, user_repo=user_repo, langfuse=langfuse)


def _make_agent_graph(
    router: RouterAgent,
    knowledge: KnowledgeAgent,
    customer_support: CustomerSupportAgent,
    escalation: EscalationAgent,
    langfuse: LangfuseAdapter,
) -> Any:
    return build_graph(
        router=router,
        knowledge=knowledge,
        customer_support=customer_support,
        escalation=escalation,
        langfuse=langfuse,
    )


class Container(containers.DeclarativeContainer):
    """Single source of truth for adapter construction and dependency wiring."""

    settings = providers.Singleton(get_settings)
    logger = providers.Singleton(setup_logging)

    langfuse_adapter = providers.Singleton(_make_langfuse, settings=settings)
    embedding_port = providers.Singleton(_make_embedding, settings=settings)
    llm_port = providers.Singleton(_make_llm, settings=settings, langfuse=langfuse_adapter)
    search_port = providers.Singleton(_make_search, settings=settings)
    queue_port = providers.Singleton(_make_queue, settings=settings)

    redis_client = providers.Singleton(_make_redis_client, settings=settings)
    cache_port = providers.Singleton(
        _make_cache,
        settings=settings,
        redis_client=redis_client,
        embedding=embedding_port,
    )

    _chroma_client = providers.Resource(_chroma_client_resource, settings=settings)
    chroma_client = _chroma_client  # public alias for health checks
    vector_store_port = providers.Singleton(
        _make_vector_store,
        settings=settings,
        chroma_client=_chroma_client,
        embedding=embedding_port,
    )

    ingest_service = providers.Singleton(
        _make_ingest_service, settings=settings, vector_store=vector_store_port
    )
    retrieval_service = providers.Singleton(
        _make_retrieval_service, settings=settings, vector_store=vector_store_port
    )
    semantic_cache_service = providers.Singleton(
        _make_semantic_cache, cache=cache_port, llm=llm_port
    )
    user_repo = providers.Singleton(_make_user_repo)
    scraper = providers.Singleton(_make_scraper, settings=settings)

    router_agent = providers.Singleton(_make_router_agent, llm=llm_port)
    knowledge_agent = providers.Singleton(
        _make_knowledge_agent,
        llm=llm_port,
        retrieval=retrieval_service,
        search=search_port,
        cache=cache_port,
        langfuse=langfuse_adapter,
    )
    customer_support_agent = providers.Singleton(
        _make_customer_support_agent, llm=llm_port, user_repo=user_repo, langfuse=langfuse_adapter
    )
    escalation_agent = providers.Singleton(
        _make_escalation_agent, langfuse=langfuse_adapter, redis_client=redis_client
    )
    input_guardrail = providers.Singleton(_make_input_guardrail, llm=llm_port)
    output_guardrail = providers.Singleton(
        _make_output_guardrail, settings=settings, langfuse=langfuse_adapter
    )
    agent_graph = providers.Singleton(
        _make_agent_graph,
        router=router_agent,
        knowledge=knowledge_agent,
        customer_support=customer_support_agent,
        escalation=escalation_agent,
        langfuse=langfuse_adapter,
    )


class ContainerSingleton:
    _instance: Container | None = None

    @classmethod
    def get_instance(cls) -> Container:
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance  # type: ignore[return-value]


def get_container() -> Container:
    return ContainerSingleton.get_instance()
