import pytest
from pydantic import ValidationError

from src.infrastructure.config.settings import (
    ConversationSettings,
    GuardrailSettings,
    LLMSettings,
    RedisSettings,
)


@pytest.mark.parametrize(
    ("settings_class", "values"),
    [
        (ConversationSettings, {"session_ttl": 0}),
        (GuardrailSettings, {"faithfulness_threshold": 1.1}),
        (LLMSettings, {"base_delay": 0}),
        (RedisSettings, {"port": 65536}),
    ],
)
def test_rejects_invalid_operational_values(settings_class: type, values: dict) -> None:
    with pytest.raises(ValidationError):
        settings_class(**values)
