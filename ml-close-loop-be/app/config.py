from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    # PRD §15 requires an `environment` on every deployment row, but no source document names
    # any environment values (the VM environment is explicitly unconfirmed, PRD §26) - a single
    # neutral, overridable default rather than an invented staging/prod ladder.
    deployment_environment: str = "default"

    # Unsloth Studio
    unsloth_studio_url: str = "http://localhost:8888"
    unsloth_api_key: str = ""
    unsloth_default_model: str = "unsloth/Qwen3-0.6B"
    unsloth_models: str = "unsloth/Qwen3-0.6B,unsloth/Qwen3.8-27B,unsloth/Qwen2.5-7B-Instruct"

    # Auth (Phase 9)
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440


settings = Settings()
