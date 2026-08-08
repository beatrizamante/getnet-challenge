import os

from arq import cron
from arq.connections import RedisSettings

async def run_ingestion_pipeline(ctx: dict, force: bool = False) -> None:
    """Crawl getnet.net, embed chunks, upsert into ChromaDB."""


async def run_eval_suite(ctx: dict) -> None:
    """Run the DeepEval golden dataset and push scores to Langfuse."""

class WorkerSettings:
    functions = [run_ingestion_pipeline, run_eval_suite]
    cron_jobs = [
        cron(run_eval_suite, hour=2, minute=0),  # nightly at 02:00 UTC
    ]
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        database=int(os.getenv("REDIS_DB", "0")),
    )
