# pylint: disable=redefined-outer-name,protected-access
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure.adapters.queue.arq_adapter import ArqQueueAdapter
from src.infrastructure.config.settings import RedisSettings

_SETTINGS = RedisSettings(host="localhost", port=6379, db=0)


@pytest.fixture
async def adapter_with_mock_pool():
    mock_pool = AsyncMock()
    mock_pool.enqueue_job = AsyncMock(return_value=MagicMock())
    with patch(
        "src.infrastructure.adapters.queue.arq_adapter.arq.create_pool", return_value=mock_pool
    ):
        inst = ArqQueueAdapter(_SETTINGS)
        # Trigger lazy pool creation
        await inst._get_pool()
        yield inst, mock_pool


async def test_enqueue_calls_enqueue_job_with_job_name(adapter_with_mock_pool):
    adapter, mock_pool = adapter_with_mock_pool
    await adapter.enqueue("run_ingestion_pipeline", force=True)
    mock_pool.enqueue_job.assert_awaited_once_with("run_ingestion_pipeline", force=True)


async def test_enqueue_at_passes_defer_until(adapter_with_mock_pool):
    adapter, mock_pool = adapter_with_mock_pool
    run_at = datetime(2026, 8, 10, 2, 0, 0, tzinfo=UTC)
    await adapter.enqueue_at("run_eval_suite", run_at)
    mock_pool.enqueue_job.assert_awaited_once_with("run_eval_suite", _defer_until=run_at)


async def test_enqueue_passes_kwargs_to_job(adapter_with_mock_pool):
    adapter, mock_pool = adapter_with_mock_pool
    await adapter.enqueue("run_ingestion_pipeline", force=False, urls=["https://getnet.net"])
    _, call_kwargs = mock_pool.enqueue_job.call_args
    assert call_kwargs["force"] is False
    assert call_kwargs["urls"] == ["https://getnet.net"]


async def test_enqueue_degrades_gracefully_on_redis_failure():
    with patch(
        "src.infrastructure.adapters.queue.arq_adapter.arq.create_pool",
        side_effect=ConnectionError("Redis down"),
    ):
        adapter = ArqQueueAdapter(_SETTINGS)
        await adapter.enqueue("run_ingestion_pipeline")  # must not raise


async def test_enqueue_at_degrades_gracefully_on_redis_failure():
    with patch(
        "src.infrastructure.adapters.queue.arq_adapter.arq.create_pool",
        side_effect=ConnectionError("Redis down"),
    ):
        adapter = ArqQueueAdapter(_SETTINGS)
        run_at = datetime(2026, 8, 10, 2, 0, 0, tzinfo=UTC)
        await adapter.enqueue_at("run_eval_suite", run_at)  # must not raise


async def test_pool_created_lazily():
    with patch("src.infrastructure.adapters.queue.arq_adapter.arq.create_pool") as mock_create:
        mock_create.return_value = AsyncMock()
        adapter = ArqQueueAdapter(_SETTINGS)
        mock_create.assert_not_called()
        await adapter._get_pool()
        mock_create.assert_called_once()


async def test_pool_reused_across_calls(adapter_with_mock_pool):
    adapter, mock_pool = adapter_with_mock_pool
    await adapter.enqueue("run_ingestion_pipeline")
    await adapter.enqueue("run_eval_suite")
    # Pool was already created in the fixture; create_pool should not be called again
    assert adapter._pool is mock_pool
