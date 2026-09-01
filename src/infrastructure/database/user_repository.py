from datetime import UTC, datetime, timedelta

from src.domain.entities.Transaction import Transaction
from src.domain.entities.User_Profile import UserProfile
from src.domain.ports.User_Repository_Port import UserRepositoryPort


class InMemoryUserRepository(UserRepositoryPort):
    """Stub implementation backed by in-memory dicts — replace with a real DB adapter."""

    def __init__(
        self,
        profiles: dict[str, UserProfile] | None = None,
        transactions: dict[str, list[Transaction]] | None = None,
    ) -> None:
        self._profiles: dict[str, UserProfile] = profiles or {}
        self._transactions: dict[str, list[Transaction]] = transactions or {}

    async def get_profile(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    async def get_transactions(self, user_id: str, days: int) -> list[Transaction]:
        cutoff = datetime.now(tz=UTC) - timedelta(days=days)
        return [t for t in self._transactions.get(user_id, []) if t.created_at >= cutoff]
