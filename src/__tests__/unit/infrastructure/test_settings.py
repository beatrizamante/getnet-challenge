import pytest
from pydantic import ValidationError

from src.infrastructure.config.settings import (
    AppSettings,
    ConversationSettings,
    GuardrailSettings,
    IngestionSettings,
    LangfuseSettings,
    LLMSettings,
    RedisSettings,
    SearchSettings,
    Settings,
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


def test_rejects_chunk_overlap_gte_chunk_size() -> None:
    with pytest.raises(ValidationError, match="INGEST_CHUNK_OVERLAP"):
        IngestionSettings(chunk_size=64, chunk_overlap=64)


def test_rejects_langfuse_enabled_without_credentials() -> None:
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY"):
        LangfuseSettings(enabled=True, public_key="", secret_key="")


def test_rejects_production_without_llm_api_key() -> None:
    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        Settings(
            app=AppSettings(env="production"),
            llm=LLMSettings(api_key=""),
            search=SearchSettings(api_key="sk-search"),
            redis=RedisSettings(password="secret"),
        )


def test_rejects_production_without_search_api_key() -> None:
    with pytest.raises(ValidationError, match="SEARCH_API_KEY"):
        Settings(
            app=AppSettings(env="production"),
            llm=LLMSettings(api_key="sk-llm"),
            search=SearchSettings(api_key=""),
            redis=RedisSettings(password="secret"),
        )


def test_rejects_production_without_redis_password() -> None:
    with pytest.raises(ValidationError, match="REDIS_PASSWORD"):
        Settings(
            app=AppSettings(env="production"),
            llm=LLMSettings(api_key="sk-llm"),
            search=SearchSettings(api_key="sk-search"),
            redis=RedisSettings(password=""),
        )


def test_valid_production_settings_pass() -> None:
    s = Settings(
        app=AppSettings(env="production"),
        llm=LLMSettings(api_key="sk-llm"),
        search=SearchSettings(api_key="sk-search"),
        redis=RedisSettings(password="secret"),
    )
    # pylint: disable=no-member
    assert s.app.env == "production"
