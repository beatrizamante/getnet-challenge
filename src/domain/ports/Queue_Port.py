from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class QueuePort(ABC):
    """Contract for any async job queue backend (ARQ, Celery, etc.)."""

    @abstractmethod
    async def enqueue(self, job_name: str, **kwargs: Any) -> str | None:
        """Enqueue a job to run as a worker is available; returns the job ID if available."""

    @abstractmethod
    async def enqueue_at(self, job_name: str, run_at: datetime, **kwargs: Any) -> None:
        """Enqueue a job to run at a specific point in time."""
