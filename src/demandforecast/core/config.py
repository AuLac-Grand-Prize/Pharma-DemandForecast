from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_port: int = 8004
    log_level: str = "INFO"

    postgres_dsn: str
    redis_url: str
    cache_ttl_seconds: int = 600

    mlflow_tracking_uri: str = "http://localhost:5000"

    moh_disease_feed_url: str
    moh_api_key: str
    weather_api_url: str
    weather_api_key: str

    internal_api_token: str
    default_horizon_days: int = 30
    default_quantiles: str = "0.1,0.5,0.9"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
