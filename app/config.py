from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def resolve_endpoint(
    specific_key: str | None,
    specific_base: str | None,
    openrouter_key: str | None,
    openai_key: str | None,
    prefer: str = "openrouter",
) -> tuple[str | None, str | None]:
    """Pick (api_key, base_url). A dedicated key always wins; then the `prefer`red shared provider.

    An OpenRouter key always goes to OpenRouter's URL, never to OpenAI's default endpoint.
    """
    if specific_key:
        if specific_key.startswith("sk-or-") and not specific_base:
            return specific_key, OPENROUTER_BASE_URL  # never send an OpenRouter key to OpenAI's endpoint
        return specific_key, specific_base
    order = ("openai", "openrouter") if prefer == "openai" else ("openrouter", "openai")
    for provider in order:
        if provider == "openrouter" and openrouter_key:
            return openrouter_key, specific_base or OPENROUTER_BASE_URL
        if provider == "openai" and openai_key:
            return openai_key, specific_base
    return None, specific_base


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True, env_ignore_empty=True)

    app_env: str = "dev"
    api_key: SecretStr

    # Chat model for the diagnosis answer. Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, ...).
    # Key order: LLM_API_KEY > OPENAI_API_KEY (direct OpenAI) > OPENROUTER_API_KEY.
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None  # e.g. https://openrouter.ai/api/v1 ; unset = OpenAI
    llm_model: str | None = None  # required (in .env) whenever an LLM key is set

    # Shared provider keys, used when LLM_API_KEY / EMBEDDING_API_KEY are not set.
    # The chat model prefers OpenAI; embeddings prefer OpenRouter.
    openrouter_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    rate_limit_per_minute: int = 30

    # Master switch for the vector half of hybrid retrieval. Off = BM25-style keyword ranking only.
    # Embedding endpoint is configured separately from the chat model (any OpenAI-compatible API).
    embeddings_enabled: bool = False
    embedding_api_key: SecretStr | None = None  # falls back to OPENAI_API_KEY
    embedding_base_url: str | None = None  # e.g. https://openrouter.ai/api/v1 ; unset = OpenAI
    embedding_model: str | None = None  # required (in .env) when EMBEDDINGS_ENABLED=true
    embedding_dimensions: int | None = None  # only sent to models that support shortening; else leave unset
    embedding_max_distance: float = 0.5  # cosine distance cutoff for the cross-DTC fallback tier

    # DTC manual service (the other system). Unset base URL = manual step disabled.
    manual_api_base_url: str | None = None  # e.g. https://manual-service.internal ; https required outside dev
    manual_api_user: str | None = None
    manual_api_password: SecretStr | None = None

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433

    @model_validator(mode="after")
    def _models_come_from_env(self) -> "Settings":
        if self.effective_llm_key is not None and not self.llm_model:
            raise ValueError("LLM_MODEL must be set in .env when an LLM key is configured")
        if self.embeddings_enabled and not self.embedding_model:
            raise ValueError("EMBEDDING_MODEL must be set in .env when EMBEDDINGS_ENABLED=true")
        if self.manual_api_base_url:
            if not self.is_dev and not self.manual_api_base_url.startswith("https://"):
                raise ValueError("MANUAL_API_BASE_URL must use https outside dev")
            if bool(self.manual_api_user) != bool(self.manual_api_password):
                raise ValueError("MANUAL_API_USER and MANUAL_API_PASSWORD must be set together")
        return self

    @property
    def llm_endpoint(self) -> tuple[str | None, str | None]:
        return resolve_endpoint(
            _secret(self.llm_api_key),
            self.llm_base_url,
            _secret(self.openrouter_api_key),
            _secret(self.openai_api_key),
            prefer="openai",
        )

    @property
    def embedding_endpoint(self) -> tuple[str | None, str | None]:
        return resolve_endpoint(
            _secret(self.embedding_api_key),
            self.embedding_base_url,
            _secret(self.openrouter_api_key),
            _secret(self.openai_api_key),
        )

    @property
    def effective_llm_key(self) -> str | None:
        return self.llm_endpoint[0]

    @property
    def effective_embedding_key(self) -> str | None:
        return self.embedding_endpoint[0]

    @property
    def is_dev(self) -> bool:
        return self.app_env == "dev"

    @property
    def db_conninfo(self) -> str:
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"user={self.postgres_user} dbname={self.postgres_db} "
            f"password={self.postgres_password.get_secret_value()}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
