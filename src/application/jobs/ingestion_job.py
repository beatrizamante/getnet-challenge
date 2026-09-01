import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from src.application.rag_pipeline.ingest_service import RagIngestService
from src.infrastructure.adapters.observability.langfuse_adapter import LangfuseAdapter
from src.infrastructure.adapters.scraper.getnet_scraper import GetnetScraper

logger = logging.getLogger(__name__)

JOB_NAME = "run_ingestion_pipeline"

_INGEST_KEY_PREFIX = "getnet:ingested:"
_INGEST_TTL = 86400 * 30  # 30 days — avoids re-fetching content that hasn't changed


async def run_ingestion_pipeline(ctx: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """ARQ job: scrape Getnet pages, chunk, embed, and upsert into ChromaDB.

    With force=False, pages already marked as ingested in Redis are skipped.
    Deterministic chunk IDs make repeated runs naturally idempotent in ChromaDB.
    """
    container = ctx["container"]
    scraper: GetnetScraper = container.scraper()
    ingest_service: RagIngestService = await container.ingest_service.async_()
    langfuse: LangfuseAdapter = container.langfuse_adapter()
    redis_client: aioredis.Redis = container.redis_client()
    delay: float = container.settings().ingestion.request_delay

    trace_id = langfuse.trace(JOB_NAME, {"force": force})
    logger.info("Ingestion job started. force=%s", force)

    pages = await scraper.scrape_all()
    stats: dict[str, Any] = {
        "pages_fetched": len(pages),
        "pages_ingested": 0,
        "pages_skipped": 0,
        "chunks_stored": 0,
    }

    for page in pages:
        cache_key = f"{_INGEST_KEY_PREFIX}{hashlib.sha256(page.url.encode()).hexdigest()[:16]}"

        if not force and await redis_client.exists(cache_key):
            logger.debug("Skipping already-ingested url=%s", page.url)
            stats["pages_skipped"] += 1
            continue

        metadata = {
            "page_title": page.title,
            "ingested_at": datetime.now(tz=UTC).isoformat(),
        }
        chunk_ids = await ingest_service.ingest(page.text, source=page.url, metadata=metadata)
        await redis_client.set(cache_key, "1", ex=_INGEST_TTL)

        stats["pages_ingested"] += 1
        stats["chunks_stored"] += len(chunk_ids)
        logger.info("Ingested url=%s chunks=%d", page.url, len(chunk_ids))

        await asyncio.sleep(delay)

    langfuse.span(trace_id, JOB_NAME, input_data={"force": force}, output=stats)
    langfuse.flush()
    logger.info("Ingestion job complete. stats=%s", stats)
    return stats
