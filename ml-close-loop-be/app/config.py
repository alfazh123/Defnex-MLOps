from typing import Literal

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

    # Serving (issue #40)
    # `mock` uses MockServingBackend (tests / no-GPU local dev); `vllm` uses the real
    # VLLMServingBackend against `vllm_url`. docker-compose's GPU serving profile starts vLLM.
    serving_backend: Literal["mock", "vllm"] = "mock"
    vllm_url: str = "http://localhost:8001"
    vllm_api_key: str = ""
    vllm_timeout_seconds: float = 60.0

    # Inference & smoke test (issue #41). Each smoke-test property is explicit and
    # configurable - not a magic number in code. The smoke test runs inside
    # `deployment_service.deploy` after the adapter is loaded but before the alias/
    # pointer moves; a failure aborts the deploy and the alias stays on the old version.
    inference_smoke_enabled: bool = True
    inference_smoke_prompt: str = "Return OK."
    # Minimum length of the generated output for the smoke test to pass. A non-empty
    # output is enough to prove the adapter can generate; tune up for stronger signal.
    inference_smoke_min_chars: int = 1
    # Max tokens for a generation (shared by the smoke test and the inference endpoint).
    inference_max_tokens: int = 128

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

    # Model artifact storage (issue #38). Location where each trained version is kept in an
    # immutable per-version directory. No longer a system temp dir — overridable via env.
    artifact_storage_dir: str = "data/artifacts"

    # Real Unsloth training runner (issue #38). The worker spawns a standalone Unsloth
    # training script in a SEPARATE venv via subprocess (never in the app's own venv —
    # serving and training venvs are kept apart per the project constraint). These knobs
    # let deployment point at the training venv's interpreter and script.
    training_python: str = "python3"
    training_script_path: str = "app/training/run_training.py"
    # Seconds before a runaway training subprocess is killed (run becomes FAILED, not stuck
    # RUNNING). 0 disables the timeout.
    training_timeout_seconds: int = 0

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
