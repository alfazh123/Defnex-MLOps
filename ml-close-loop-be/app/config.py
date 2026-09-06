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
    unsloth_models: str = (
        "unsloth/Qwen3-0.6B,unsloth/Qwen3.8-27B,unsloth/Qwen2.5-7B-Instruct"
    )

    # Auth (Phase 9)
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    jwt_refresh_expire_minutes: int = 10080  # 7 days

    # DB connection pool
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800

    # CORS origins (comma-separated)
    cors_origins: str = (
        "http://localhost:3000,http://localhost:8888,http://localhost:5173"
    )

    # Request size limit
    max_request_body_size: int = 1_048_576  # 1MB

    # GPU training lock (issue #33): serializes training across worker processes.
    # `gpu_lock_file` must live on a filesystem all workers share (the ./data volume
    # in docker-compose); flock only guarantees exclusivity among processes that open
    # the same file on the same host.
    gpu_lock_file: str = "data/gpu.lock"
    # Seconds a worker waits for the GPU lock before skipping the poll (run stays PENDING).
    gpu_lock_timeout: int = 300

    # Logging
    debug: bool = False
    log_level: str = "INFO"

    # Promotion eval gate (issue #43, model-promotion-approval-workflow.md §6).
    # All criteria are boolean toggles with documented, non-invented defaults; promote
    # still requires a human trigger (no automatic promotion on numeric thresholds).
    eval_gate_require_eval_set_reference: bool = True
    eval_gate_require_qualitative_majority: bool = True
    eval_gate_require_no_general_regression: bool = True
    eval_gate_require_eval_loss_not_worse: bool = True


settings = Settings()
