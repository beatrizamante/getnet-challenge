import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src._lib.container import get_container
from src.domain.shared.Application_Errors import BaseError, NotFoundError
from src.interface.http.routes.admin import router as admin_router
from src.interface.http.routes.chat import router as chat_router
from src.interface.http.routes.health import router as health_router
from src.interface.http.server import make_server

logger = logging.getLogger(__name__)

CONTAINER = get_container()
CONTAINER.logger()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _settings = CONTAINER.settings()
    logger.info("Starting Getnet Challenge. env=%s", _settings.app.env)
    await CONTAINER.init_resources()  # type: ignore[misc]
    await CONTAINER.vector_store_port.async_()
    await CONTAINER.agent_graph.async_()
    CONTAINER.input_guardrail()
    CONTAINER.output_guardrail()
    CONTAINER.semantic_cache_service()
    yield
    await CONTAINER.shutdown_resources()  # type: ignore[misc]
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Getnet Multi-Agent Support",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(health_router)
app.include_router(admin_router)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "http method=%s path=%s status=%s latency_ms=%d",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.exception_handler(NotFoundError)
async def _not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": exc.code, "detail": exc.message})


@app.exception_handler(BaseError)
async def _domain_error_handler(_: Request, exc: BaseError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": exc.code, "detail": exc.message})


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "detail": "An unexpected error occurred."},
    )


def main() -> None:
    """Entrypoint for `uv run getnet-challenge` and `python -m src.main` (run from project root)."""
    server = make_server()
    server.run()


if __name__ == "__main__":
    main()
