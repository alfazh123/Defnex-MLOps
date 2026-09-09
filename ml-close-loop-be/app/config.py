from typing import Literal, Self

from pydantic import model_validator
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
    # Per-environment vLLM URLs (issue #68, PRD §16.1/§19.4): comma-separated `env:url` pairs,
    # e.g. `staging:http://staging-vllm:8001,production:http://prod-vllm:8001`. Lets a deploy to a
    # named environment target a different host instead of the single `vllm_url` — changing a
    # staging/prod host is a config change, not a code change. Empty (default) falls back to
    # `vllm_url` for every environment (backward compatible).
    vllm_url_by_env: str = ""
    vllm_api_key: str = ""
    vllm_timeout_seconds: float = 60.0
    # Base model the serving stack is actually running (issue #65). The `serving`
    # compose service loads this as its base model; deploy verifies the artifact's
    # recorded base_model matches before moving the pointer (PRD §17.4: a base-model
    # change is a controlled recreate/redeploy, never a silent hot-swap). Empty
    # (default) disables the check so unconfigured/legacy setups keep deploying.
    served_base_model: str = ""

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

    # Job heartbeat / stale detection (issue #60, PRD §10.2-10.4). A RUNNING job
    # persists `heartbeat_at` every `heartbeat_interval_seconds`; a run whose heartbeat
    # (or start time, before the first heartbeat) has not advanced past
    # `stale_threshold_seconds` is reclaimed as STALE by the detector so a crashed
    # worker never leaves it stuck RUNNING.
    heartbeat_interval_seconds: int = 10
    stale_threshold_seconds: int = 60

    # Cross-service GPU coordination (issue #39): before training starts, the serving
    # service is stopped and VRAM verified free, then restarted after training ends —
    # all inside the same #33 GPU lock. `serving_control=mock` (default) disables all
    # of this (the worker never touches serving, preserving pre-#39 behavior for tests
    # and no-GPU dev); `serving_control=shell` runs the stop/start/health-check commands
    # below against the real serving stack.
    serving_control: Literal["mock", "shell"] = "mock"
    serving_stop_cmd: str = ""
    serving_start_cmd: str = ""
    serving_health_cmd: str = ""
    # Per-command timeout so a hung serving stop/start cannot stall the worker forever.
    serving_command_timeout: float = 60.0

    # VRAM verification: the worker waits until `vram_free_threshold_mb` MB are free,
    # polling every `vram_check_poll` seconds, for at most `vram_check_timeout` seconds.
    # `vram_reader=nvidia_smi` reads real free memory via `nvidia-smi`;
    # `vram_reader=mock` (default) assumes the threshold is met so no GPU is touched.
    vram_reader: Literal["mock", "nvidia_smi"] = "mock"
    vram_free_threshold_mb: int = 8192
    vram_check_poll: float = 5.0
    vram_check_timeout: int = 300

    @model_validator(mode="after")
    def _validate_serving_coordination(self) -> Self:
        """No half-configured safety pipeline.

        `serving_control=shell` only activates the real stop/verify/train/restart
        cycle; with default `mock` every field below is inert. Shell mode therefore
        requires its full configuration to be deliberately set — an empty command
        or a mock VRAM reader would silently pretend the safety checks ran while
        the operator believes the whole pipeline is active:
        - the stop/start/health-check commands must all be non-empty;
        - the VRAM reader must be the real `nvidia_smi` (a mock reader that always
          reports the threshold as met fakes the free-VRAM verification);
        - `vram_free_threshold_mb` must be set explicitly — there is no built-in
          default until a governance decision picks an H100 free-VRAM budget.
        """
        if self.serving_control != "shell":
            return self
        missing = [
            name
            for name, value in (
                ("SERVING_STOP_CMD", self.serving_stop_cmd),
                ("SERVING_START_CMD", self.serving_start_cmd),
                ("SERVING_HEALTH_CMD", self.serving_health_cmd),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "SERVING_CONTROL=shell requires non-empty commands; "
                f"missing: {', '.join(missing)}"
            )
        if self.vram_reader != "nvidia_smi":
            raise ValueError(
                "SERVING_CONTROL=shell requires VRAM_READER=nvidia_smi; a mock "
                "reader would fake the free-VRAM check the shell mode promises"
            )
        if "vram_free_threshold_mb" not in self.model_fields_set:
            raise ValueError(
                "SERVING_CONTROL=shell requires VRAM_FREE_THRESHOLD_MB to be set "
                "explicitly (no default until a governance decision exists)"
            )
        if self.vram_free_threshold_mb <= 0:
            raise ValueError("VRAM_FREE_THRESHOLD_MB must be positive")
        return self

    # Model artifact storage (issue #38). Location where each trained version is kept in an
    # immutable per-version directory. No longer a system temp dir — overridable via env.
    artifact_storage_dir: str = "data/artifacts"

    # Artifact backend selection (issue #71, PRD §13.1). `local` uses
    # LocalFilesystemArtifactStorage (dev / CI); `minio` uses MinioArtifactStorage for
    # production object storage (PRD §13.2: PostgreSQL = metadata, MinIO/S3 = bytes).
    artifact_backend: Literal["local", "minio"] = "local"

    # MinIO / S3-compatible object storage settings (issue #71, PRD §13.1).
    # All defaults match the compose-baseline minio service (PRD §32).
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "artifacts"
    minio_secure: bool = False  # True = HTTPS to the MinIO/S3 endpoint

    # Real Unsloth training runner (issue #38). The worker spawns a standalone Unsloth
    # training script in a SEPARATE venv via subprocess (never in the app's own venv —
    # serving and training venvs are kept apart per the project constraint). These knobs
    # let deployment point at the training venv's interpreter and script.
    training_python: str = "python3"
    training_script_path: str = "app/training/run_training.py"
    # Seconds before a runaway training subprocess is killed (run becomes FAILED, not stuck
    # RUNNING). 0 disables the timeout.
    training_timeout_seconds: int = 0

    # SSH / remote connectivity (issue #73)
    ssh_connect_timeout: float = 30.0

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
