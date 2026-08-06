from abc import ABC, abstractmethod


class CachePort(ABC):
    """Contract for any cache backend (Redis, in-memory, etc.)."""

    @abstractmethod
    async def get(self, key: str) -> str | None: ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...
