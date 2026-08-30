from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    # PRD §15 requires an `environment` on every deployment row, but no source document names
    # any environment values (the VM environment is explicitly unconfirmed, PRD §26) - a single
    # neutral, overridable default rather than an invented staging/prod ladder.
    deployment_environment: str = "default"


settings = Settings()
