from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from the project .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    paysim_dataset_path: str = "PaySim_Dataset.zip"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    max_query_rows: int = Field(default=200, ge=1, le=1000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
