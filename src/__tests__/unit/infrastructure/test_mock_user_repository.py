from datetime import timezone

import pytest

from src.domain.shared.Application_Errors import UserNotFoundError
from src.infrastructure.adapters.user_repository.mock_user_repository import MockUserRepository


@pytest.fixture
def repo():
    return MockUserRepository()


async def test_get_profile_returns_known_user(repo):
    profile = await repo.get_profile("cliente1988")
    assert profile.plan == "Get Smart"
    assert profile.status == "active"


async def test_get_profile_raises_for_unknown_user(repo):
    with pytest.raises(UserNotFoundError) as exc_info:
        await repo.get_profile("nao_existe")
    assert exc_info.value.user_id == "nao_existe"


async def test_get_profile_seeded_for_all_five_users(repo):
    user_ids = ["cliente1988", "cliente2024", "cliente_suspenso", "cliente_novo", "empresa_xpto"]
    for uid in user_ids:
        profile = await repo.get_profile(uid)
        assert profile is not None, f"Missing seed for {uid}"


async def test_get_transactions_returns_within_days(repo):
    txs = await repo.get_transactions("cliente1988", days=7)
    assert len(txs) > 0
    for tx in txs:
        assert tx.settlement_date >= tx.created_at


async def test_get_transactions_filters_by_days(repo):
    recent = await repo.get_transactions("cliente1988", days=5)
    all_time = await repo.get_transactions("cliente1988", days=365)
    assert len(all_time) >= len(recent)


async def test_get_transactions_raises_for_unknown_user(repo):
    with pytest.raises(UserNotFoundError):
        await repo.get_transactions("nao_existe", days=30)


async def test_get_transactions_empty_for_new_user(repo):
    txs = await repo.get_transactions("cliente_novo", days=30)
    assert txs == []


async def test_transaction_amounts_are_in_cents(repo):
    txs = await repo.get_transactions("empresa_xpto", days=365)
    assert all(t.amount > 0 for t in txs)
    # sanity: amounts are in cents (R$500,00 = 50000)
    assert any(t.amount >= 10000 for t in txs)


async def test_suspended_user_profile_is_accessible(repo):
    profile = await repo.get_profile("cliente_suspenso")
    assert profile.status == "suspended"
