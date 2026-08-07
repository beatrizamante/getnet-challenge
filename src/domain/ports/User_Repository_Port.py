from abc import ABC, abstractmethod

from src.domain.entities.Transaction import Transaction
from src.domain.entities.User_Profile import UserProfile


class UserRepositoryPort(ABC):
    """Contract for retrieving user data (profile and transaction history)."""

    @abstractmethod
    async def get_profile(self, user_id: str) -> UserProfile | None: ...

    @abstractmethod
    async def get_transactions(self, user_id: str, days: int) -> list[Transaction]: ...
