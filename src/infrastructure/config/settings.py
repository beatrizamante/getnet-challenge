from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: str = Field(default="openai")  # openai | anthropic | gemini
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="deepseek-chat")
    base_url: str = Field(default="https://api.deepseek.com")

class ChromaDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHROMA_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    collection_name: str = Field(default="getnet-docs")

class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    password: str = Field(default="")
    db: int = Field(default=0)
    ttl: int = Field(default=3600)
    cache_similarity_threshold: float = Field(default=0.92)

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


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HF_", env_file=".env", extra="ignore")

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARCH_", env_file=".env", extra="ignore")

    api_key: str = Field(default="")
    max_results: int = Field(default=5)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: str = Field(default="development")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    log_level: str = Field(default="INFO")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    chroma: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
