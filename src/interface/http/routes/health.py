import asyncio
import importlib.metadata

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src._lib.container import Container, get_container
from src.infrastructure.adapters.vector_store.chroma_adapter import ChromaAdapter

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe — returns 200 as long as the process is alive."""
    try:
        version = importlib.metadata.version("getnet-challenge")
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return {"status": "ok", "version": version}


@router.get("/ready")
async def ready(
    container: Annotated[Container, Depends(get_container)],
) -> JSONResponse:
    """Readiness probe — verifies Redis and ChromaDB are reachable."""
    checks: dict[str, str] = {}
    ok = True

    try:
        redis = container.redis_client()
        await asyncio.wait_for(redis.ping(), timeout=2.0)
        checks["redis"] = "ok"
    except Exception as exc:  # pylint: disable=broad-except
        checks["redis"] = f"error: {exc}"
        ok = False

    try:
        chroma: ChromaAdapter = container.vector_store_port()  # type: ignore[assignment]
        await asyncio.wait_for(chroma.ping(), timeout=2.0)
        checks["chromadb"] = "ok"
    except Exception as exc:  # pylint: disable=broad-except
        checks["chromadb"] = f"error: {exc}"
        ok = False

    status_code = 200 if ok else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if ok else "degraded", "checks": checks},
    )
