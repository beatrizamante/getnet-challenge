
from typing import Any
import chromadb
import redis.asyncio as aioredis
from dependency_injector import containers, providers

from src.config.logger import setup_logging
from src.infrastructure.adapters.cache.redis_cache_adapter import RedisCacheAdapter
from src.infrastructure.adapters.embeddings.huggingface_adapter import HuggingFaceEmbeddingAdapter
from src.infrastructure.adapters.llm.llm_adapter import LLMAdapter
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.queue.arq_adapter import ArqQueueAdapter
from src.infrastructure.adapters.search.tavily_adapter import TavilySearchAdapter
from src.infrastructure.adapters.vector_store.chroma_adapter import ChromaAdapter
from src.infrastructure.config.settings import Settings, get_settings


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
    """Async resource: opened on container.init_resources(), closed on shutdown."""
    client = await chromadb.AsyncHttpClient(
        host=settings.chroma.host, port=settings.chroma.port
    )
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

class Container(containers.DeclarativeContainer):
    """Single source of truth for adapter construction and dependency wiring."""

    settings = providers.Singleton(get_settings)
    logger = providers.Singleton(setup_logging)

    langfuse_adapter = providers.Singleton(_make_langfuse, settings=settings)
    embedding_port = providers.Singleton(_make_embedding, settings=settings)
    llm_port = providers.Singleton(_make_llm, settings=settings, langfuse=langfuse_adapter)
    search_port = providers.Singleton(_make_search, settings=settings)
    queue_port = providers.Singleton(_make_queue, settings=settings)

    _redis_client = providers.Singleton(_make_redis_client, settings=settings)
    cache_port = providers.Singleton(
        _make_cache,
        settings=settings,
        redis_client=_redis_client,
        embedding=embedding_port,
    )

    _chroma_client = providers.Resource(_chroma_client_resource, settings=settings)
    vector_store_port = providers.Singleton(
        _make_vector_store,
        settings=settings,
        chroma_client=_chroma_client,
        embedding=embedding_port,
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
