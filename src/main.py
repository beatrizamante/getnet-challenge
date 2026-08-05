from contextlib import asynccontextmanager

from fastapi import FastAPI

from src._lib.container import get_container
from src.interface.http.server import make_server

CONTAINER = get_container()
logger = CONTAINER.logger()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initializes modules on app start instead of waiting for a request"""
    logger.info("Loading model and tokenizer...")
    logger.info("Model and tokenizer ready.")
    yield

app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    logger.info("Starting WatchMe AI Backend...")
    server = make_server()
    server.run()
