from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: str = Field(default="openai")  # openai | anthropic | gemini
    api_key: str = Field(default="")
    llm_model: str = Field(default="deepseek-chat")
    base_url: str = Field(default="https://api.deepseek.com")
    max_retries: int = Field(default=3, ge=0)
    base_delay: float = Field(default=1.0, gt=0)


class ChromaDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=8000, ge=1, le=65535)
    collection_name: str = Field(default="getnet-docs")


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: str = Field(default="")
    db: int = Field(default=0, ge=0)
    ttl: int = Field(default=3600, gt=0)
    cache_similarity_threshold: float = Field(default=0.92, ge=0, le=1)

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_", env_file=".env", extra="ignore")

    secret_key: str = Field(default="")
    public_key: str = Field(default="")
    host: str = Field(default="https://cloud.langfuse.com")
    enabled: bool = Field(default=False)

    @model_validator(mode="after")
    def _credentials_required_when_enabled(self) -> "LangfuseSettings":
        if self.enabled and (not self.public_key or not self.secret_key):
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required "
                "when LANGFUSE_ENABLED=true"
            )
        return self


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HF_", env_file=".env", extra="ignore")

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARCH_", env_file=".env", extra="ignore")

    api_key: str = Field(default="")
    max_results: int = Field(default=5, gt=0)


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=".env", extra="ignore")

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    request_delay: float = Field(default=1.5, ge=0)
    max_concurrent: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _overlap_lt_chunk_size(self) -> "IngestionSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"INGEST_CHUNK_OVERLAP ({self.chunk_overlap}) must be "
                f"less than INGEST_CHUNK_SIZE ({self.chunk_size})"
            )
        return self


class GuardrailSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GUARDRAIL_", env_file=".env", extra="ignore")

    model: str = Field(default="deepseek-v4-flash")
    faithfulness_threshold: float = Field(default=0.7, ge=0, le=1)
    enabled: bool = Field(default=True)
    judge_timeout: float = Field(default=10.0, gt=0)
    safe_rejection: str = Field(
        default=(
            "I'm unable to process that request. "
            "I'm here to help with questions about Getnet's payment solutions and services."
        )
    )
    disclaimer: str = Field(
        default=(
            "\n\n\u26a0\ufe0f Note: this response may not be fully grounded in Getnet's official documentation. "
            "Please verify with official support."
        )
    )


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER_", env_file=".env", extra="ignore")

    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=4, gt=0)
    factor: int = Field(default=4, gt=0)
    enabled: bool = Field(default=False)
    default_k: int = Field(default=4, gt=0)


class ConversationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONV_", env_file=".env", extra="ignore")

    session_ttl: int = Field(default=3600, gt=0)
    max_turns: int = Field(default=7, gt=0)


class EscalationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESCALATION_", env_file=".env", extra="ignore")

    audit_ttl: int = Field(default=86400 * 30, gt=0)
    max_entries: int = Field(default=100, gt=0)
    handoff_answer: str = Field(
        default=(
            "I'm transferring you to a human specialist who can better assist with your request. "
            "Please hold \u2014 someone from the Getnet support team will be with you shortly."
        )
    )


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    kb_cache_ttl: int = Field(default=1800, gt=0)
    off_topic_answer: str = Field(
        default=(
            "I'm only able to assist with questions related to Getnet's payment solutions and services. "
            "For other topics, please use a general-purpose search engine."
        )
    )


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    prompts_file: str = Field(default="")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    guardrail: GuardrailSettings = Field(default_factory=GuardrailSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    escalation: EscalationSettings = Field(default_factory=EscalationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    @model_validator(mode="after")
    def _production_requirements(self) -> "Settings":  # pylint: disable=no-member
        # pylint: disable=no-member
        if self.app.env != "production":
            return self
        errors: list[str] = []
        if not self.llm.api_key:
            errors.append("LLM_API_KEY is required in production")
        if not self.search.api_key:
            errors.append("SEARCH_API_KEY is required in production")
        if not self.redis.password:
            errors.append("REDIS_PASSWORD is required in production")
        if errors:
            raise ValueError("\n".join(errors))
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
