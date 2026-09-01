import logging
from datetime import datetime
from typing import Any

import arq
import arq.connections

from src.domain.ports.Queue_Port import QueuePort
from src.infrastructure.config.settings import RedisSettings

logger = logging.getLogger(__name__)


class ArqQueueAdapter(QueuePort):
    """QueuePort backed by ARQ — jobs are persisted in Redis and picked up by the arq worker."""

    def __init__(self, settings: RedisSettings) -> None:
        self._arq_settings = arq.connections.RedisSettings(
            host=settings.host,
            port=settings.port,
            password=settings.password or None,
            database=settings.db,
        )
        self._pool: arq.ArqRedis | None = None

    async def _get_pool(self) -> arq.ArqRedis:
        if self._pool is None:
            self._pool = await arq.create_pool(self._arq_settings)
        return self._pool

    async def enqueue(self, job_name: str, **kwargs: Any) -> str | None:
        """Enqueue a job; logs a warning and continues if Redis is unavailable."""
        try:
            pool = await self._get_pool()
            job = await pool.enqueue_job(job_name, **kwargs)
            logger.debug("Job enqueued. job=%s id=%s", job_name, job.job_id if job else None)
            return job.job_id if job else None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "ARQ enqueue skipped (Redis unavailable). job=%s error=%s", job_name, exc
            )
            self._pool = None  # reset so the next call retries the connection
            return None

    async def enqueue_at(self, job_name: str, run_at: datetime, **kwargs: Any) -> None:
        """Enqueue a job deferred until `run_at`."""
        try:
            pool = await self._get_pool()
            await pool.enqueue_job(job_name, _defer_until=run_at, **kwargs)
            logger.debug("Job scheduled. job=%s run_at=%s", job_name, run_at)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "ARQ enqueue_at skipped (Redis unavailable). job=%s error=%s", job_name, exc
            )
            self._pool = None
