from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True, env_ignore_empty=True)

    app_env: str = "dev"
    api_key: SecretStr
    rate_limit_per_minute: int = 30

    log_level: str = "INFO"
    log_dir: str = "logs"

    demo_passcode: SecretStr | None = None
    demo_trust_network: bool = False

    # Chat + embedding model (OpenAI-compatible). LLM_BASE_URL empty = OpenAI itself.
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_reasoning_effort: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536

    # Retrieval tuning.
    retrieval_top_k: int = 30
    retrieval_keep: int = 15
    similarity_threshold: float = 0.60  # cosine similarity, matches dropped below this

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.llm_api_key is not None and not self.llm_model:
            raise ValueError("LLM_MODEL must be set in .env when LLM_API_KEY is set")
        return self

    @property
    def llm_enabled(self) -> bool:
        return self.llm_api_key is not None and bool(self.llm_model)

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
