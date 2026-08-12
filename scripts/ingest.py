"""One-shot ingestion: scrapes Getnet pages and populates ChromaDB. Run via docker compose run --rm ingest."""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

from src._lib.container import get_container


sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

async def main() -> None:

    container = get_container()
    await container.init_resources()  # type: ignore[misc]
    try:
        scraper = container.scraper()
        ingest_service = container.ingest_service()

        logger.info("Starting Getnet knowledge base ingestion...")
        pages = await scraper.scrape_all()
        logger.info("Scraped %d pages.", len(pages))

        total_chunks = 0
        for page in pages:
            chunk_ids = await ingest_service.ingest(
                page.text,
                source=page.url,
                metadata={
                    "page_title": page.title,
                    "ingested_at": datetime.now(tz=timezone.utc).isoformat(),
                },
            )
            total_chunks += len(chunk_ids)
            logger.info("  ✓ %s → %d chunks", page.url, len(chunk_ids))

        logger.info("Ingestion complete. total_chunks=%d", total_chunks)
    finally:
        await container.shutdown_resources()  # type: ignore[misc]


if __name__ == "__main__":
    asyncio.run(main())
