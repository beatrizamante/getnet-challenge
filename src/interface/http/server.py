import logging
import uvicorn

def make_server():
    """
    Configures and returns a Uvicorn server instance for the FastAPI application.

    The server is set up to run on host '0.0.0.0' and port 5000, with debug-level logging and colored output.
    Logs the configuration process using the 'farmaassistant' logger.

    Returns:
        uvicorn.Server: A configured Uvicorn server instance ready to be started.
    """
    logger = logging.getLogger("farmaassistant")
    logger.info("Configuring server...")

    config = uvicorn.Config("main:app",
                        host="0.0.0.0",
                        port=5000,
                        log_level="debug",
                        use_colors=True
                        )
    server = uvicorn.Server(config)
    logger.info("Server configured successfully")
    return server
