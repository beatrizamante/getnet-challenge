from contextlib import asynccontextmanager

from fastapi import FastAPI

from src._lib.container import get_container
from src.interface.http.server import make_server

CONTAINER = get_container()
logger = CONTAINER.logger()

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = CONTAINER.settings()
    logger.info("Settings loaded. env=%s", settings.app.env)
    await CONTAINER.init_resources()  # type: ignore[misc]
    yield
    await CONTAINER.shutdown_resources()  # type: ignore[misc]

app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    logger.info("Starting Getnet Challenge...")
    server = make_server()
    server.run()
