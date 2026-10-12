from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True, env_ignore_empty=True)

    app_env: str = "dev"
    api_key: SecretStr
    rate_limit_per_minute: int = 30

    # Chat model (OpenAI). LLM_BASE_URL is only for a custom OpenAI-compatible endpoint; empty = OpenAI.
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None

    # DTC manual service. Empty base URL = manual step skipped. Include the route prefix, e.g. http://host:3001/api
    manual_api_base_url: str | None = None
    manual_api_user: str | None = None
    manual_api_password: SecretStr | None = None

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5433

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        if self.llm_api_key is not None and not self.llm_model:
            raise ValueError("LLM_MODEL must be set in .env when LLM_API_KEY is set")
        if self.manual_api_base_url:
            if not self.is_dev and not self.manual_api_base_url.startswith("https://"):
                raise ValueError("MANUAL_API_BASE_URL must use https outside dev")
            if bool(self.manual_api_user) != bool(self.manual_api_password):
                raise ValueError("MANUAL_API_USER and MANUAL_API_PASSWORD must be set together")
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
