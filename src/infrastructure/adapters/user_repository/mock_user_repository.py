from datetime import datetime, timedelta, timezone

from src.domain.entities.Transaction import Transaction
from src.domain.entities.User_Profile import UserProfile
from src.domain.ports.User_Repository_Port import UserRepositoryPort
from src.domain.shared.Application_Errors import UserNotFoundError

_NOW = datetime(2026, 8, 8, tzinfo=timezone.utc)


def _tx(tx_id: str, amount: int, status: str, days_ago: int, settle_days: int = 1) -> Transaction:
    created = _NOW - timedelta(days=days_ago)
    return Transaction(
        id=tx_id,
        amount=amount,
        status=status,
        created_at=created,
        settlement_date=created + timedelta(days=settle_days),
    )


_PROFILES: dict[str, UserProfile] = {
    "cliente1988": UserProfile(
        plan="Get Smart",
        machine_model="Smart 2",
        status="active",
        joined_at=datetime(2020, 3, 15, tzinfo=timezone.utc),
    ),
    "cliente2024": UserProfile(
        plan="Get Clássica",
        machine_model="Clássica Plus",
        status="active",
        joined_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
    ),
    "cliente_suspenso": UserProfile(
        plan="Get Smart",
        machine_model="Smart 2",
        status="suspended",
        joined_at=datetime(2019, 6, 1, tzinfo=timezone.utc),
    ),
    "cliente_novo": UserProfile(
        plan="Get Clássica",
        machine_model="Clássica",
        status="active",
        joined_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    ),
    "empresa_xpto": UserProfile(
        plan="Get Smart Pro",
        machine_model="Smart Pro",
        status="active",
        joined_at=datetime(2018, 11, 30, tzinfo=timezone.utc),
    ),
}

_TRANSACTIONS: dict[str, list[Transaction]] = {
    "cliente1988": [
        _tx("TX-1988-001", 15000, "settled",   days_ago=1,  settle_days=1),
        _tx("TX-1988-002", 32500, "settled",   days_ago=2,  settle_days=1),
        _tx("TX-1988-003",  8900, "settled",   days_ago=3,  settle_days=1),
        _tx("TX-1988-004", 12000, "pending",   days_ago=0,  settle_days=1),
        _tx("TX-1988-005", 50000, "settled",   days_ago=10, settle_days=1),
        _tx("TX-1988-006",  4500, "refunded",  days_ago=15, settle_days=1),
        _tx("TX-1988-007", 27800, "settled",   days_ago=20, settle_days=1),
    ],
    "cliente2024": [
        _tx("TX-2024-001", 9900,  "settled",  days_ago=1, settle_days=1),
        _tx("TX-2024-002", 5000,  "pending",  days_ago=0, settle_days=1),
    ],
    "cliente_suspenso": [],
    "cliente_novo": [],
    "empresa_xpto": [
        _tx("TX-XPTO-001", 500000, "settled",  days_ago=1,  settle_days=1),
        _tx("TX-XPTO-002", 320000, "settled",  days_ago=3,  settle_days=1),
        _tx("TX-XPTO-003", 180000, "pending",  days_ago=0,  settle_days=1),
    ],
}

class MockUserRepository(UserRepositoryPort):
    """Deterministic seed-based repository — designed for development and integration tests."""

    async def get_profile(self, user_id: str) -> UserProfile:
        profile = _PROFILES.get(user_id)
        if profile is None:
            raise UserNotFoundError(user_id)
        return profile

    async def get_transactions(self, user_id: str, days: int) -> list[Transaction]:
        if user_id not in _PROFILES:
            raise UserNotFoundError(user_id)
        cutoff = _NOW - timedelta(days=days)
        return [t for t in _TRANSACTIONS.get(user_id, []) if t.created_at >= cutoff]
