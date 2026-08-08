import os

import arq.connections

from src._lib.container import get_container
from src.application.jobs.ingestion_job import run_ingestion_pipeline


async def _on_startup(ctx: dict) -> None:
    container = get_container()
    await container.init_resources()  # type: ignore[misc]
    ctx["container"] = container


async def _on_shutdown(ctx: dict) -> None:
    container = ctx.get("container")
    if container:
        await container.shutdown_resources()  # type: ignore[misc]


class WorkerSettings:
    redis_settings: arq.connections.RedisSettings = arq.connections.RedisSettings(  # type: ignore[misc]
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        database=int(os.getenv("REDIS_DB", "0")),
    )
    functions = [run_ingestion_pipeline]
    on_startup = _on_startup
    on_shutdown = _on_shutdown
