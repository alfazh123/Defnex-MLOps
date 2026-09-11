from datetime import datetime
from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, field_validator

TrainingRunStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "STALE"]

# Single source of truth for accepted PEFT methods (issue #34). Consumed by both the
# request schema (PeftMethod Literal below) and the Unsloth mapper in
# app/services/unsloth_client.py — never keep a second, divergent list elsewhere.
# rslora is kept because it produces a regular LoRA adapter that the vLLM serving
# path can serve; it only changes the scale factor alpha/sqrt(r) at training time.
SUPPORTED_PEFT_METHODS = ("lora", "qlora", "rslora")

# Values that exist in the ecosystem but cannot be honored by this project's vLLM-only
# serving path (DoRA/QDoRA reparameterization, and the Full Finetuning branch that has no
# PEFT adapter to serve). Rejection is a final governance decision (issue #44, following
# up on the temporary rejection issue #34 introduced) — not contingent on the current
# infrastructure and not expected to change without a new governance decision. They are
# rejected with a specific message rather than silently downgraded to plain LoRA.
_SERVING_UNSUPPORTED_PEFT_METHODS = ("dora", "qdora", "none")

PeftMethod = Literal[SUPPORTED_PEFT_METHODS]

# Fair-use priority queue (issue #135): single source of truth for accepted values,
# consumed by both the request schema below and the claim-order ranking in
# app/services/training_service.py. Default "normal" keeps existing FIFO behavior for
# callers that never set it.
PRIORITY_LEVELS = ("low", "normal", "high")
Priority = Literal[PRIORITY_LEVELS]


class TrainingConfig(BaseModel):
    """Training configuration knobs mapped to Unsloth Studio TrainingStartRequest.

    Loosely typed (`extra="allow"`) per openapi.yaml's own
    `additionalProperties: true` — known properties are documented from
    model-artifact-versioning-lineage.md §6's sample record. Defaults match
    Unsloth Studio defaults where confirmed, otherwise use sensible fallbacks.

    See: https://github.com/unslothai/unsloth/blob/main/studio/backend/models/training.py
    """

    model_config = ConfigDict(extra="allow")

    # PEFT method
    peft_method: PeftMethod = "lora"
    load_in_4bit: bool = False

    @field_validator("peft_method", mode="before")
    @classmethod
    def _reject_serving_unsupported(cls, v: object) -> object:
        if v in _SERVING_UNSUPPORTED_PEFT_METHODS:
            raise ValueError(
                f"peft_method {v!r} is permanently rejected (governance decision, "
                f"issue #44): not supported by this project's vLLM serving path; "
                f"supported values: {', '.join(SUPPORTED_PEFT_METHODS)}"
            )
        return v

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = []
    use_loftq: bool = False

    # Dataset
    hf_dataset: str = ""
    format_type: str = "chatml"
    train_split: str = "train"
    eval_split: str | None = None
    eval_steps: float = 0.0

    # Training hyperparameters
    learning_rate: float | str | None = None
    epochs: int | None = None
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    max_steps: int | None = None
    save_steps: int = 100
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    random_seed: int = 42
    packing: bool = False
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"
    max_seq_length: int = 4096

    # Gradient checkpointing
    gradient_checkpointing: str = "unsloth"

    # Model options
    trust_remote_code: bool = False


class TrainingRunCreateRequest(BaseModel):
    """Request to create a training run (openapi.yaml TrainingRunCreateRequest)."""

    dataset_id: str
    dataset_version: int
    # `model_id` becomes part of the immutable version name `{model_id}-{base_model}-v{N}`
    # (issue #38) and a filesystem directory, so it is restricted to path- and URL-safe
    # characters rather than any free string.
    model_id: str

    @field_validator("model_id", mode="before")
    @classmethod
    def _validate_model_id(cls, v: object) -> object:
        if isinstance(v, str) and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v):
            raise ValueError(
                "model_id must match [A-Za-z0-9][A-Za-z0-9._-]* "
                "(it is embedded in the immutable version name and artifact path)"
            )
        return v

    base_model: str
    training_config: TrainingConfig
    triggered_by: str | None = None
    compute_resource_id: int | None = None
    priority: Priority = "normal"


class TrainingRun(BaseModel):
    """A training run's stored record plus live status/progress (openapi.yaml TrainingRun)."""

    training_run_id: str
    status: TrainingRunStatus
    dataset_id: str
    dataset_version: int
    model_id: str
    base_model: str
    training_config: TrainingConfig
    triggered_by: str | None = None
    created_at: datetime
    current_epoch: int | None = None
    current_step: int | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    model_version: int | None = None
    retry_of: str | None = None


class GpuHoursReportRow(BaseModel):
    """One (triggered_by, model_id) aggregation row of the GPU-hour cost report
    (issue #135) -- computed from existing `TrainingRun.started_at`/`finished_at`,
    no new table. `triggered_by` stands in for "tenant/user" and `model_id` for
    "project" since those are the only attribution fields TrainingRun already has.
    """

    triggered_by: str | None = None
    model_id: str
    run_count: int
    gpu_hours: float
