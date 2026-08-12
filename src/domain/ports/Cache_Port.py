from abc import ABC, abstractmethod


class CachePort(ABC):
    """Contract for any cache backend (Redis, in-memory, etc.)."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Retrieve a value from the cache by key."""

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int) -> None:
        """Set a value in the cache with an optional TTL (in seconds)."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value from the cache by key."""
