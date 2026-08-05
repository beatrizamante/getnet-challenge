

from src.config.logger import setup_logging

from dependency_injector import containers, providers


class Container(containers.DeclarativeContainer):
    """Dependency injection container for application services and configuration."""

    logger = providers.Singleton(setup_logging)

class ContainerSingleton:
    """Singleton wrapper for the DI container"""
    _instance: Container | None = None

    @classmethod
    def get_instance(cls) -> Container:
        """Get the singleton container instance"""
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance # type: ignore


def get_container() -> Container:
    """Get the global container instance"""
    return ContainerSingleton.get_instance()
