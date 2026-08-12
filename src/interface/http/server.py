import logging
import uvicorn

from src.infrastructure.config.settings import AppSettings

def make_server() -> uvicorn.Server:
    logger = logging.getLogger(__name__)
    app_cfg = AppSettings()
    config = uvicorn.Config(
        "src.main:app",
        host=app_cfg.host,
        port=app_cfg.port,
        log_level=str(app_cfg.log_level).lower(),
        use_colors=True,
    )
    server = uvicorn.Server(config)
    logger.info("Server configured. host=%s port=%d", app_cfg.host, app_cfg.port)
    return server
