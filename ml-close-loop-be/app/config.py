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
    jwt_expire_minutes: int = 15  # short-lived; never checked against revocation (#125)
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

    # Request size limits (issue #242, audit finding T3).
    #
    # `max_request_body_size` guards every route. It used to be the *only* limit, which
    # made the intake endpoint's advertised 100 MB dataset-file cap unreachable: uploads
    # over 1 MB were rejected here before the handler's own 100 MB check could run. The
    # dataset intake routes are now granted a larger body limit derived from
    # `max_upload_file_size` (see `app/main.py`), so both numbers come from here and
    # cannot drift apart.
    max_request_body_size: int = 1_048_576  # 1MB
    # Per-file cap enforced by the dataset intake handler (`app/api/intake.py`).
    max_upload_file_size: int = 104_857_600  # 100MB

    # Validation gate thresholds (issue #241, audit finding T2).
    #
    # The gate used to consider leakage and nothing else, so a file whose every record
    # failed H1-H9 still reported `gate_decision: PASS` and could be committed and
    # trained on. These two ratios make the decision explicit and tunable instead of
    # implicit; see `docs/dataset/validation-gate-policy.md` for the full policy.
    #
    # `validation_gate_fail_ratio`: share of records carrying at least one hard error
    # (H0-H9) at which the gate FAILs. Leakage (H8) fails the gate at any ratio
    # regardless. Default 0.10 = one in ten records failing a hard rule is not a dataset
    # worth training on without a human fixing the source first.
    validation_gate_fail_ratio: float = 0.10

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
    # Maximum times a STALE run can be reclaimed before it is forced to FAILED (P2-6).
    max_stale_retries: int = 3

    # Cross-service GPU coordination (issue #39): before training starts, the serving
    # service is stopped and VRAM verified free, then restarted after training ends —
    # all inside the same #33 GPU lock. `serving_control=mock` (default) disables all
    # of this (the worker never touches serving, preserving pre-#39 behavior for tests
    # and no-GPU dev); `serving_control=shell` runs the stop/start/health-check commands
    # below against the real serving stack.
    serving_control: Literal["mock", "shell", "file_signal"] = "mock"
    serving_stop_cmd: str = ""
    serving_start_cmd: str = ""
    serving_health_cmd: str = ""
    # Per-command timeout so a hung serving stop/start cannot stall the worker forever.
    serving_command_timeout: float = 60.0

    # File-based GPU signaling (Option B1, Phase 2A). Worker writes JSON requests to a
    # shared mount; a host-side systemd service reads them and executes Docker stop/start
    # + nvidia-smi. No docker.sock in any container. `gpu_control_dir` is the path inside
    # the container where request.json/response.json/active.json live (shared via volume mount).
    gpu_control_dir: str = "/models/.gpu-control"
    # How long the worker waits for the host controller to respond before timing out.
    gpu_control_timeout: float = 120.0
    # How often the worker polls response.json for updates.
    gpu_control_poll: float = 1.0

    # VRAM verification: the worker waits until `vram_free_threshold_mb` MB are free,
    # polling every `vram_check_poll` seconds, for at most `vram_check_timeout` seconds.
    # `vram_reader=nvidia_smi` reads real free memory via `nvidia-smi`;
    # `vram_reader=mock` (default) assumes the threshold is met so no GPU is touched.
    vram_reader: Literal["mock", "nvidia_smi"] = "mock"
    vram_free_threshold_mb: int = 8192
    vram_check_poll: float = 5.0
    vram_check_timeout: int = 300

    @model_validator(mode="after")
    def _check_jwt_secret_safe(self) -> Self:
        """Fail fast in production if jwt_secret still has the insecure dev default.

        Only triggers when ``ENVIRONMENT`` is explicitly set to ``production``
        (not just ``debug=False``, which is also the default during tests).
        """
        env = getattr(self, "deployment_environment", "default")
        if env.lower() != "production":
            return self
        if self.jwt_secret == "dev-secret-change-in-production":
            import logging

            logging.getLogger(__name__).critical(
                "FATAL: jwt_secret is the insecure dev default. "
                "Set JWT_SECRET env var before running in production."
            )
            raise SystemExit(1)
        return self

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

    @model_validator(mode="after")
    def _validate_file_signal_coordination(self) -> Self:
        """File-signal mode requires a control directory and explicit VRAM threshold.

        Like shell mode, `file_signal` activates the real stop/verify/train/restart
        cycle via a host-side controller. The worker writes JSON requests to a shared
        mount and the controller reads them, executes Docker commands, and writes
        responses. The control directory must be set and VRAM threshold explicit.
        """
        if self.serving_control != "file_signal":
            return self
        if not self.gpu_control_dir:
            raise ValueError(
                "SERVING_CONTROL=file_signal requires GPU_CONTROL_DIR to be set"
            )
        if "vram_free_threshold_mb" not in self.model_fields_set:
            raise ValueError(
                "SERVING_CONTROL=file_signal requires VRAM_FREE_THRESHOLD_MB "
                "to be set explicitly (no default until a governance decision exists)"
            )
        if self.vram_free_threshold_mb <= 0:
            raise ValueError("VRAM_FREE_THRESHOLD_MB must be positive")
        return self

    # Model artifact storage (issue #38). Location where each trained version is kept in an
    # immutable per-version directory. No longer a system temp dir — overridable via env.
    artifact_storage_dir: str = "data/artifacts"

    # Dataset intake staging + permanent storage (issue #42).
    dataset_storage_dir: str = "data/datasets"

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

    # OpenTelemetry (PRD §28)
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # Alerting webhook (PRD §28.1, issue #84). Empty disables alerts.
    alert_webhook_url: str = ""

    # Promotion eval gate (issue #43, model-promotion-approval-workflow.md §6).
    # All criteria are boolean toggles with documented, non-invented defaults; promote
    # still requires a human trigger (no automatic promotion on numeric thresholds).
    eval_gate_require_eval_set_reference: bool = True
    eval_gate_require_qualitative_majority: bool = True
    eval_gate_require_no_general_regression: bool = True
    eval_gate_require_eval_loss_not_worse: bool = True

    # License tracking (issue #134). `ModelVersion.base_model`/`TrainingRun.base_model` already
    # record which base model a run used, so the license is a config lookup keyed by that string
    # rather than a new DB column duplicating it. Comma-separated `base_model:license` pairs
    # (same shape as `vllm_url_by_env`, `serving.py:150`); an unlisted base model resolves to
    # None (unknown), never a guessed value.
    base_model_licenses: str = "Qwen/Qwen3.8-27B:apache-2.0"
    # Non-commercial license detection for the promotion warning below (issue #134 Open
    # Decision): comma-separated substrings checked case-insensitively against a dataset
    # version's `license`. This is a PLACEHOLDER heuristic pending governance's exact
    # "commercial use" definition - it only ever drives a human-facing warning
    # (`promotion_service._license_warning`), never an automatic block.
    non_commercial_license_patterns: str = "cc-by-nc,noncommercial,non-commercial"

    # Rate limits for write endpoints (P2-3)
    rate_limit_training_create: str = "10/minute"
    rate_limit_intake_validate: str = "10/minute"
    rate_limit_intake_commit: str = "10/minute"
    rate_limit_deploy: str = "5/minute"
    rate_limit_promotion_decision: str = "5/minute"

    # Artifact retention policy (issue #132, PRD §43). PRD §43 only states a negative rule --
    # "artifacts required for rollback must not be deleted merely because a model is no longer
    # active" -- it names no positive minimum-retention number, so these are explicit,
    # operator-overridable defaults, not values sourced from the PRD. They define only the
    # *minimum age before a REJECTED/ARCHIVED/superseded artifact becomes eligible* for deletion
    # consideration; this codebase does not implement an automatic deletion job against them.
    # Full policy: docs/dataset/retention-policy.md.
    retention_rejected_dataset_days: int = 90
    retention_superseded_dataset_days: int = 180
    retention_rejected_model_version_days: int = 90
    retention_archived_model_version_days: int = 365


settings = Settings()
