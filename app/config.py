from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    model_path: str = "model/model.pkl"
    model_version: str = "1.0.0"
    model_source: str = "local"  # local | gcs

    gcs_bucket: str = ""
    gcs_model_key: str = ""

    feature_names_path: str = "model/feature_names.json"
    model_metadata_path: str = "model/model_metadata.json"

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    fraud_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_max_size: int = Field(default=100, ge=1, le=1000)
    shap_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
